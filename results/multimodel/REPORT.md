# 三種 dLLM × 解碼模式 × batch size：FOCUS、sliding window 與 unmask 規則

2026-11-17 · GSM8K · 全部實驗都已完成（SDAR 35 組、I-DLM 24 組、DiffusionGemma 28 組）

## TL;DR

- **「連續兩次預測相同就解」在三個模型上的效果完全不同。**
  - SDAR：會掉準確率，從 77.3% 降到 74.6%。
  - I-DLM：沒有損失。它的 ISD 驗證規則本身就是這條規則，而且做得很嚴謹。
  - DiffusionGemma：bs=8 時能加速 25%，準確率不掉。
- **FOCUS 只有在 GPU 是瓶頸、而且 mask 會單調減少時才有用。**
  - SDAR：要 bs ≥ 32 才贏 vanilla，bs=128 時快 1.7 倍；bs=1 時反而 vanilla 最快。
  - DiffusionGemma：每一步都重新評估整個 canvas，所以 FOCUS 沒有好處，sliding window 也一樣。
- **兩個新 mode：**
  - EB sampler（γ=0.5）是 SDAR 每步解出最多 token 的規則，比 baseline 多 25%，準確率掉 1.5 個百分點，大約落在「直接降 threshold」的取捨曲線上。
  - soft force-first 比原本的 force-first 多 1.5 個百分點，速度一樣。
- **I-DLM 的 stride 最好用 N=4。** 放寬驗證最多快 29%，但準確率會從 93.6% 掉到 62.3%。

## 1. 實驗設定

| | SDAR-1.7B-Chat-b32 | I-DLM-8B | DiffusionGemma-26B-A4B-it |
|---|---|---|---|
| 解碼方式 | mask 型 block diffusion，block 內雙向 attention（block 長度 32） | 嚴格 causal 加 logit shift，自我 speculative 的 ISD | 隨機 token 當初始雜訊，256-token canvas，雙向 attention，用 EB sampler |
| 推論引擎 | 本 repo 的 LMDeploy（eager，continuous batching） | 官方 SGLang ISD（我加了 patch） | transformers 5.23，static batching |
| GPU | A6000 | A6000 | A100 80GB |
| 解碼 | greedy | greedy | 官方 sampler（有隨機性） |
| batch size（題數） | 1（24）、8（128）、32（384）、128（全部 1319） | 1（24）、8（96）、32（256）、64（512） | 1（8）、4（32）、8（64） |

共同設定：不開 thinking，prompt 結尾要求把答案寫在 `\boxed{}`，max_new_tokens 為 1024（DiffusionGemma 為 512）。
準確率以 SDAR 在 bs=128（1319 題）、I-DLM 在 bs=64（512 題）、DiffusionGemma 在 bs=8（64 題）的結果為主；其他 batch size 題數較少，準確率只能參考。

## 2. 每種 mode 在三個模型上的對應

| mode | SDAR | I-DLM | DiffusionGemma |
|---|---|---|---|
| FOCUS | 在第 1 層丟掉不重要的 mask | 不適用：mask 只有 N−1 個，而且在 causal attention 下，丟掉 mask 就等於縮小 N | 第 2 層之後，只保留已接受的 token，加上 entropy 最低的 ⌈1.5 × 平均每步新接受數⌉ 個雜訊位置 |
| sliding window | 只送到「解到最右邊的位置 + 4」 | 就是 stride N（N = 2、4、8） | 只送 canvas 的前 W 個位置，W = frontier + 64 |
| 連續兩次相同且 > 0.5 | `--stable-unmask --stable-min-conf 0.5` | 原本的驗證規則「草稿 = argmax」，再加上 `verify_min_prob 0.5` | argmax 跟上一步相同、而且 p ≥ 0.5 就接受 |
| 每步一定解最左邊的 token | `--force-first` | 不需要另外做：每步至少一定輸出 1 個 token | 強制接受最左邊、還沒被接受的位置 |
| 新 A | **EB sampler**（γ = 0.1 或 0.5），取代固定 threshold | top-5 放寬驗證；完全不驗證 | 這就是官方的 baseline |
| 新 B | **soft force-first**：最左邊的 token 只在 p ≥ 0.5 時才強制解 | — | threshold 0.9（改用 SDAR 的規則） |

## 3. SDAR-1.7B（LMDeploy）

| mode | bs=1 | bs=8 | bs=32 | bs=128 tok/s | 準確率（1319 題） | tok/step | 每個輸出 token 的處理量 |
|---|---|---|---|---|---|---|---|
| vanilla | **89** | 421 | 984 | 1421 | 74.5% | 2.39 | 13.4 |
| FOCUS | 66 | 352 | 1090 | 2624 / 2380 | 76.0 / 75.4% | 1.88 | 4.18 |
| FOCUS + window（fw） | 68 | 428 | 1378 | 2361 / 2549 | **77.3%** | 1.97 | 3.79 |
| fw + 連續兩次相同 ≥ 0.5 | 80 | 452 | 1505 | 3050 | 74.6% | 2.23 | 3.27 |
| fw + force-first | 78 | 445 | 1418 | 2995 | 74.1% | 2.06 | 3.38 |
| fw + soft force-first（新 B） | 74 | 455 | 1455 | 2938 | 75.6% | 2.09 | 3.41 |
| fw + EB γ=0.1（新 A） | 71 | 426 | 1280 | 2609 | 76.9% | 2.04 | 3.56 |
| fw + EB γ=0.5（新 A） | 85 | **514** | **1557** | 3035 | 75.8% | **2.46** | 3.03 |

FOCUS 和 fw 各跑了兩次（以 / 分隔）。同一個設定重跑，tok/s 會差約 ±10%，所以比較時以 tok/step 為準。

- **batch size 是決定性的因素。** FOCUS 系列要到 bs ≥ 32 才贏 vanilla（1.4 倍），bs=128 時快 1.7 倍；bs=1 時 vanilla 最快。
  - 原因是 delayed cache 每步都要等 CPU 處理完，vanilla 則每次 CPU 介入可以連跑 4 步。batch 小的時候 GPU 本來就閒，省下的計算換不到時間。
- **新規則都是拿準確率換步數。**
  - 拿早上只降 threshold 的結果當對照：s=0.9 是 75.9%、2.18 tok/step，s=0.8 是 76.3%、2.38 tok/step。
  - EB 0.5（75.8%、2.46）大約在這條取捨曲線上；其他規則都落在曲線下方。
- **加上 0.5 的信心下限，兩條規則都改善了。**
  - stable：沒有下限時是 71.3%，加了下限是 74.6%。
  - force-first：hard 版 74.1%，soft 版 75.6%。

## 4. I-DLM-8B（SGLang ISD）

| 設定 | bs=1 | bs=8 | bs=32 | bs=64 tok/s | 準確率（512 題） | tok / 每條序列每步 | 一輪草稿全部被接受的比例 |
|---|---|---|---|---|---|---|---|
| ISD N=2 | 73 | 514 | 1521 | 1818 | 93.2% | 1.91 | 95% |
| **ISD N=4（官方）** | 119 | 769 | 1510 | 1837 | **93.6%** | 3.15 | 79% |
| ISD N=8 | 120 | 713 | 1037 | 1117 | 93.8% | 3.33 | 19% |
| N=4 + 驗證時 p ≥ 0.5 | 115 | 761 | 1517 | 1799 | 93.9% | 3.12 | 78% |
| N=4 + top-5 放寬驗證 | 129 | 825 | 1631 | 2019 | 86.3% | 3.46 | 87% |
| N=4 不驗證 | **153** | **942** | **1891** | **2362** | 62.3% | 3.98 | 100% |

- **greedy 的驗證是無損的，所以 N=2／4／8 和 p ≥ 0.5 的準確率都一樣。** 「p ≥ 0.5」幾乎沒有作用，因為會被接受的草稿本來就是 argmax，機率幾乎都大於 0.5。
- **N=4 最快。**
  - N=8 每條序列每步要算 15 個位置，一輪草稿全部被接受的比例只有 19%，bs=64 時反而慢了 40%。
  - N=2 在小 batch 時太保守（bs=1：73 tok/s）。
- **少驗證就是拿準確率換速度。** top-5 快 10%、掉 7 個百分點；完全不驗證快 29%、掉 31 個百分點，而且 97/512 題撞到長度上限（輸出開始重複）。

## 5. DiffusionGemma-26B-A4B（transformers）

| mode | bs=1 | bs=4 | bs=8 tok/s | 準確率（64 題，bs=8） | 準確率（8 題，bs=1） | 步數（bs=8） | 深層 / 第 0–1 層的 token 比 |
|---|---|---|---|---|---|---|---|
| EB（官方） | **124** | 172 | 129 | 93.8% | 100% | 363 | 1.00 |
| + 連續兩次相同 ≥ 0.5 | 117 | 171 | **162** | 95.3% | 100% | **277** | 1.00 |
| + force-first | 115 | 158 | 149 | 95.3% | 100% | 314 | 1.00 |
| threshold 0.9（新 B） | 116 | **175** | 160 | 95.3% | 100% | 284 | 1.00 |
| + FOCUS | 26 | 89 | 124 | 93.8% | **12%** | 468 | 0.84 |
| + FOCUS（aging） | 36 | | 126 | 96.9% | 100% | 371 | 0.93 |
| + FOCUS + window64 | 28 | 88 | 107 | 96.9% | 62% | 483 | 0.97 |
| + FOCUS + window64（aging） | 42 | | 130 | 93.8% | 100% | 381 | 0.99 |
| + FOCUS + window64 + 連續兩次相同 ≥ 0.5 | 39 | 107 | 142 | 92.2% | 88% | 351 | 0.97 |
| + FOCUS + window64 + force-first | 28 | 101 | 109 | 93.8% | 75% | 479 | 0.97 |

- **在 bs=8 時，「連續兩次相同 ≥ 0.5」和 threshold 0.9 是不花成本的加速**：步數少 22–24%，tok/s 多 24–26%，準確率沒掉。
  - 在 bs=1 和 bs=4 時沒有加速。單條序列本來就大約 9 步就收斂了，這些規則加速的主要是 static batch 裡最慢的那條序列。
- **FOCUS 和 window 在每種 batch size 下都更慢。**
  - 這個 sampler 每一步都重新評估整個 canvas，接受過的 token 也可能被換掉，而且要等整個 canvas 都收斂才會停。少算的位置只會拖慢收斂。
- **原始版 FOCUS 在 bs=1 時失效（12%）。**
  - 原因：重要度用的是「上一步的 entropy」，被丟掉的位置不會重新計算，所以一直被丟、永遠收斂不了。
  - 修正：加上 aging（連續被丟 2 步就強制保留一步）之後，準確率回到 100%（8 題），但速度還是慢 3 倍。
  - 原版 FOCUS 的重要度是在同一次 forward 裡新算的，沒有這個問題。
- **bs=8 比 bs=4 慢，是 static batching 造成的。** 每一批都要等最慢的那條 canvas 收斂。

## 6. 跨模型的結論

1. **一條規則有沒有用，取決於「第二次預測」算不算獨立的證據。**
   - SDAR：block 內是雙向 attention，又有 delayed cache，兩次預測的 context 幾乎一樣，所以重複並不代表正確。
   - I-DLM：驗證時是在已接受的前文條件下，用 causal attention 重新算一次，是真正的驗證。
   - DiffusionGemma：沒被接受的位置每步都會重新換成雜訊，在不同的雜訊下還能維持同一個 argmax，比較可靠。這一點是推測，還沒有額外驗證。
2. **FOCUS 和 window 要同時滿足兩個條件才有效：**
   - 瓶頸在 GPU，也就是 batch 夠大。
   - 還沒解出的位置只會越來越少，也就是 mask 型的模型。
   DiffusionGemma 不滿足第二個條件；SDAR 在小 batch 時不滿足第一個條件。
3. **在 SDAR 上，目前沒有一條規則勝過「直接降 threshold」。** 要真正加速 SDAR，還是要解決 CPU 的瓶頸，例如之前做過的 CUDA graph、合併 prefill、micro-batch 交替。

## 7. 限制

- 小 batch 和 DiffusionGemma 的題數少（8–64 題），準確率的雜訊約 ±5–10 個百分點。tok/s 重跑會差約 ±10%。
- DiffusionGemma 跑在 transformers 的 static batching 上，而且是 A100，所以它的 tok/s 不能直接跟另外兩個模型比。
- DiffusionGemma 版的 FOCUS 用 entropy 當重要度，不是原版 FOCUS 的 attention 分數。
- I-DLM 用的是 greedy、non-thinking；官方報告的 95.0 用的是 thinking 加 sampling。

## 8. 程式碼與重現方式

- **SDAR**
  - 新規則：`lmdeploy/pytorch/strategies/dllm/unmasking.py`（`entropy_bound`、`force_first_min_conf`），對應參數 `--entropy-bound`、`--force-first-min-conf`。
  - sweep：`results/multimodel/sdar_sweep.sh`。
- **I-DLM**
  - patch：`exps/multimodel/patch_idlm.py`（加上 `verify_min_prob` 和統計檔輸出）。
  - client：`exps/multimodel/idlm_client.py`。
  - sweep：`results/multimodel/idlm_sweep.sh`。
- **DiffusionGemma**
  - mode 實作：`exps/multimodel/dg_modes.py`。
  - 評估腳本：`exps/multimodel/dg_eval.py`。
- **測試**
  - `results/multimodel/test_new_modes.py`：EB 規則與參考實作一致，soft force-first 正確。
  - `results/multimodel/test_dg_tiny.py`：tiny model 上，什麼都不丟時與原版 `generate` bit-for-bit 一致；window 前綴與模型原本的 forward 一致；沒算到的位置絕不會被接受。
- **表格**：`python3 results/multimodel/summarize.py`。

## 9. 追加：GPU 端多步（`--gpu-multistep`）試驗

**結論：不值得繼續往研究方向做。** 兩個門檻都沒過，量到的上限也不夠高。

- **做法**：把 delayed cache + window 決定下一步處理哪些位置的邏輯搬到 GPU（`strategies/dllm/multistep.py`）。每次 CPU 介入可以連跑 K 步，CPU 的簿記和下一步的 forward 平行處理。目前還不支援 FOCUS。
- **正確性**：
  - CPU 對照測試 86,400 個 sequence-step 全部一致。
  - bs=1 時 K=1 和 K=4 的 16 題輸出完全相同，處理的 token 數也一樣。
- **結果**：SDAR-1.7B，A6000（Kentucky，Xeon w9-3475X），eager 模式，單位 tok/s。

  | 模式 | bs=1 | bs=8 | bs=32 | bs=128 | 準確率 (1319 題) |
  |---|---|---|---|---|---|
  | vanilla（修正前） | 142 | 699 | 1246 | 1502 | 74.5% |
  | vanilla（修正後） | 150 | 1014 | 1495 | 1690 | 74.1% |
  | FOCUS + window，K=1 | 111 | 677 | 2090 | 3534 | 77.5% |
  | delayed + window，K=1 | 119 | 682 | 2323 | 3505 | 77.5% |
  | delayed + window，K=4（修正後） | 128 | 656 | 2296 | **4158** | 77.6% |

  - bs=128 時，K=4 是 K=1 的 1.19 倍，沒有達到 1.3 倍。
  - bs≤8 時，K=4 還是輸給 vanilla（656 對 1014，128 對 150）。
- **為什麼上限不高**：bs≈124 的 steady state 下，K=1 的 GPU 本來就有 77% 的時間在跑 kernel（eager 和 graph 都一樣），K=4 只拉高到 81–83%。所以就算把 host 開銷全部消掉，上限也只有約 1.3 倍。
  - 之前估「57% 忙碌、上限 1.76 倍」的是 FOCUS 版本，GPU 閒置主要是 FOCUS 在第 1 層的 host sync 造成的。
  - 不開 FOCUS 時本來就沒有那個閒置；而 FOCUS + window 的 tok/s 並沒有比 delayed + window 高。
- **順便找到的 engine bug（值得保留）**：每 round 跑多個 loop 時，送回應的工作會被餓死。
  - 主迴圈去拿 output 時 output 通常已經好了，所以不會讓出 event loop；而 `forward_event` 在 round 中途就被清掉。結果完成的請求送不回 client，batch 一路降到只剩 2 條序列，才一次補進 125 條。
  - 這個問題也影響 vanilla，因為它預設每 round 跑 4 個 loop。修正後 vanilla 在 bs=8 快了 45%、bs=128 快了 12%。**所以前面各節 SDAR 表格裡的 vanilla 數字都偏低。** 例如 bs=8 時，FOCUS + window 其實輸 vanilla 33%，不是打平。
  - 修正：`engine/engine_loop.py` 的 `_main_loop_get_outputs`，在 `forward_event` 還沒清掉的那幾個 loop 裡讓出一次 event loop。另外 `inputs_maker.prefetch_next_inputs` 在開 multistep 時不預先排 decode，這一項是試驗性的，效果沒有單獨量。
- **還沒修的小問題**：`ModelInputs.step()` 每一步都把 `max_kv_seqlen` 加上一個 block，即使那一步沒有 block 退休，所以會多估。目前不影響正確性。
- **資料**：
  - `results/multimodel/multistep_eager/`：修正前的完整 sweep 與一致性檢查。
  - `results/multimodel/multistep_eager_fix/`：修正後的結果。
  - `results/multimodel/multistep_prof/`：step profile 與 GPU 忙碌比例。
  - 腳本：`results/multimodel/multistep_bench.sh`。
