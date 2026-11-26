# 重現指南：dLLM 解碼模式 × batch size（multimodel-exp）

完整結果與分析見 [results/multimodel/REPORT.md](results/multimodel/REPORT.md)。

## 版本

| tag | commit | 對應結果 |
|---|---|---|
| `results-v1` | `9f7bdd11` | REPORT 第 3–8 節，以及第 9 節的「修正前」 |
| `results-v2` | 本 commit（含 `5c080435` 的 engine 修正） | REPORT 第 9 節的「修正後」 |

`results-v1` 的 vanilla 數字受第 9 節那個 engine bug 影響而偏低，在小 batch 尤其明顯。

## 方法與 commit 對照

| 方法 | commit | 開啟方式（`exps/gsm8k_eval.py` 參數） |
|---|---|---|
| FOCUS token eviction（原 repo 既有） | — | `--focus` |
| delayed cache 上的 sliding window | `35121389` | `--window-k 4`，搭配 `--delayed-cache` 或 `--focus` |
| infra：host 簿記批次化、融合的 unmask kernel、LM head 只算 mask | `72c8279f` | `--fast-host --fast-unmask --masked-logits` |
| infra：FOCUS 的 CUDA graph | `4d903939` | 拿掉 `--eager` |
| FOCUS 保留量與觸發門檻選項、step profiling | `72c8279f` | `--focus-min-keep 4 --focus-evict-gate 1`；環境變數 `DLLM_STEP_PROFILE=out.json` |
| infra：prefill 合併、decode micro-batch | `ec04648c` | `--prefill-min-waiting 16`、`--decode-micro-batch 64` |
| unmask 規則：每步強制解最左邊的 mask（force-first） | `e7aea791` | `--force-first`，可加門檻 `--force-first-min-conf 0.5` |
| unmask 規則：連續兩次預測相同就解（stable） | `e7aea791` | `--stable-unmask`，可加 `--stable-min-conf 0.5`、`--stable-ctx N` |
| unmask 規則：entropy bound（EB sampler） | `e7aea791` | `--entropy-bound 0.5` |
| unmask 規則：error budget | `e7aea791` | `--unmask-budget B` |
| GPU 端 multistep | `95ceb952` | `--delayed-cache --window-k 4 --gpu-multistep 4`（不支援 FOCUS） |
| I-DLM 的 ISD 驗證門檻 | `9f7bdd11` | `exps/multimodel/patch_idlm.py`（`verify_min_prob`） |
| DiffusionGemma 各模式（EB、stable、force-first、FOCUS-entropy、window、aging） | `9f7bdd11` | `exps/multimodel/dg_modes.py`；`dg_eval.py --runs <mode>:<bs>:<num>` |
| engine 修正：回應被餓死 | `5c080435` | 預設開啟，沒有參數 |

## 環境

- **GPU**：SDAR 和 I-DLM 用 RTX A6000 48GB；DiffusionGemma 用 A100 80GB。
- **Docker image**：`pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel`。repo 預設放在 `/workspace/FOCUS-sliding-window`，放在別處就設 `REPO`。
- **共同安裝**：

```bash
pip install -r requirements/runtime_cuda.txt
DISABLE_TURBOMIND=1 pip install -e .
pip install -U "huggingface_hub[cli]" "brotli>=1.2" pandas pyarrow   # brotli < 1.2 會讓 hf download 失敗
hf download openai/gsm8k main/test-00000-of-00001.parquet --repo-type dataset --local-dir /workspace/data/gsm8k
```

- **SDAR**：`hf download JetLM/SDAR-1.7B-Chat-b32`
- **I-DLM**：`apt-get install -y libnuma1`，然後執行 `bash results/multimodel/idlm_sweep.sh setup`。這一步會 clone I-DLM 的 SGLang、建立 venv、套用 patch 並下載模型。
- **DiffusionGemma**：`pip install transformers==5.23.0 accelerate`，然後 `hf download google/diffusiongemma-26B-A4B-it`

## 重現每張表

所有腳本都會跳過已經有 `.json` 的組合，而 repo 裡已經 commit 了原本的結果，所以**一定要把 `OUT` 設成新目錄**。資料不在 `/workspace/data` 時，另外設 `GSM8K`。

| 結果 | tag | 指令 |
|---|---|---|
| infra 優化累加 ablation 與 nsys 時間拆解（見 [results/gsm_opt/README.md](results/gsm_opt/README.md)） | `ec04648c` 或之後 | `OUT=results/repro/gsm_opt bash results/gsm_opt/run_opt.sh` |
| SDAR：8 種模式 × bs 1/8/32/128（第 3 節） | `results-v1` | `OUT=results/repro/sdar bash results/multimodel/sdar_sweep.sh` |
| I-DLM：ISD 設定 × bs 1/8/32/64（第 4 節） | `results-v1` | `OUT=$PWD/results/repro/idlm bash results/multimodel/idlm_sweep.sh` |
| DiffusionGemma：模式 × bs 1/4/8（第 5 節） | `results-v1` | `OUT=results/repro/dg bash results/multimodel/dg_sweep.sh` |
| force-first 和 stable 單獨與合併（FOCUS + window） | `results-v1` | `OUT=results/repro/uf bash results/unmask_feat/run.sh` |
| GPU multistep 修正前（第 9 節） | `results-v1` | `OUT=results/repro/ms_v1 bash results/multimodel/multistep_bench.sh eager` |
| GPU multistep 與 vanilla 修正後（第 9 節） | `results-v2` | `OUT=results/repro/ms_v2 bash results/multimodel/multistep_bench.sh eager` |
| GPU 忙碌比例（第 9 節） | `results-v2` | 見下方 |

量 GPU 忙碌比例（K=1 對 K=4）：

```bash
DLLM_STEP_PROFILE_KERNELS=300:120 DLLM_STEP_PROFILE=prof.json python exps/gsm8k_eval.py \
  --model JetLM/SDAR-1.7B-Chat-b32 --data $GSM8K --out results/repro/prof --tag k4 --eager --slack 1.0 \
  --num 384 --concurrency 128 --delayed-cache --window-k 4 --fast-host --fast-unmask --masked-logits --gpu-multistep 4
# prof.json 的 kernels 欄位：gpu_kernel_s / wall_s = GPU 忙碌比例；拿掉 --gpu-multistep 就是 K=1
```

產生表格：`python results/multimodel/summarize.py results/repro`，會讀取其中的 `sdar/`、`idlm/`、`dg/`。

## 注意

- **雜訊**：tok/s 重跑大約差 ±10%。小 batch 和 DiffusionGemma 題數少，準確率大約差 ±5–10 個百分點。
- **與當時程式碼的差異**：第 3–8 節的結果是在程式還在修改時跑的，`results-v1` 是最接近當時的版本。新功能預設關閉，關閉時的輸出與舊版一致，有 A/A 對照確認過。
- **其他結果目錄**：`results/unmask_feat2/`（stable 加門檻、`stable_ctx`）和 `results/unmask_feat3/`（error budget、EB）是另一輪探索，各自用目錄內的 `run.sh` 重現，報告沒有收錄。
- **只 commit 了摘要**：`results/` 被 `.gitignore` 排除，commit 裡只有每次 run 的 `.json` 摘要，沒有原始輸出。
