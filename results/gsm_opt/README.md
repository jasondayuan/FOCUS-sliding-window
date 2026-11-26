# FOCUS + window 的 infra 優化

模型 SDAR-1.7B-Chat-b32，資料 GSM8K 全部 1319 題，bs=128，單張 RTX A6000，greedy 解碼。
解碼設定：FOCUS + window k=4（`--focus-min-keep 4 --focus-evict-gate 1`），threshold 0.9 × slack 0.9。

## 結論

這組 dLLM serving 卡在 CPU，GPU 大部分時間在等 host。從下往上依序疊了 4 項優化，throughput 從 1911 提升到 3337–3546 tok/s，是原本的 **1.75–1.86 倍**，準確率不變。

## 累加 ablation

每一列都包含上面各列的優化。

| run | 加上的優化 | 參數 | commit | tok/s | 準確率 |
|---|---|---|---|---|---|
| `ref_fw4_s09` | 基準（eager） | — | — | 1911 | 76.8% |
| `b_flags` | host 簿記批次化、融合的 unmask Triton kernel、LM head 只算 mask | `--fast-host --fast-unmask --masked-logits` | `72c8279f` | 2184（+14%） | 77.3% |
| `o1_graph` | CUDA graph（FOCUS eviction 之後的層） | 拿掉 `--eager` | `4d903939` | 2426（+11%） | 76.8% |
| `o2_prefill16` | prefill 合併：等到 16 個請求再一起 prefill | `--prefill-min-waiting 16` | `ec04648c` | 2832（+17%） | 76.5% |
| `o3_mb64` | decode micro-batch：兩組 64 條序列交替，一組做 host 工作時另一組在 forward | `--decode-micro-batch 64` | `ec04648c` | 3337（+18%）；重跑 3546 | 77.3% |

`o3_mb64` 拿掉了 `--masked-logits`。它的 `nonzero` 會讓 host 等 forward 跑完，和 micro-batch 的重疊效果衝突。

**重跑**：先切到 `ec04648c` 或之後的 commit，然後執行 `OUT=results/repro/gsm_opt bash results/gsm_opt/run_opt.sh`。這個腳本是從 `.json` 裡記錄的參數還原的，原本的腳本沒有保留。在 A6000 上每個 run 大約 2–4 分鐘。

## 時間花在哪裡（nsys，`../prof_a6000/`）

取 steady state 下約 30 個 decode step 量測。

| profile | 設定 | 每步週期 | GPU 忙碌 | GPU 閒置時 host 在做什麼（ms/step） |
|---|---|---|---|---|
| `fw4_s09` | 基準（eager） | 146 ms | 38% | `update_running` 23.6、decode forward 21.2、`create_inputs` 20.7、prefill 16.6 |
| `best_k16` | 加上 3 個參數 + CUDA graph + prefill 合併 | 83 ms | 57% | `update_running` 13.2、`create_inputs` 7.5、decode forward 4.6 |
| `mb64` | 再加 micro-batch 64（一步是一個約 64 條序列的 micro-batch） | 49 ms | 61% | decode forward 5.3、prefill 3.6、`update_running` 3.6 |

- `update_running` 和 `create_inputs` 是每步在 host 上更新 delayed cache / FOCUS 狀態、組下一步輸入的時間。
- 做完這 4 項，剩下的閒置主要來自 host 的簿記和 FOCUS 第 1 層的 host sync。後來試過把簿記搬到 GPU（GPU multistep，`95ceb952`），只多了 19%。詳見 `../multimodel/REPORT.md` 第 9 節。

**重跑 profiling**：需要 Nsight Systems 2025.5。

- **nsys 時間拆解**：`nsys_run.sh`（eager）、`nsys_run_graph.sh`（CUDA graph）、`nsys_run_mb.sh`（micro-batch）。用法是 `bash nsys_run_graph.sh <name> <gsm8k_eval 參數>`。
- **不用 nsys 的計時**：`timing_run.sh`。
- **分析**：`analyze_nsys.py` 會產生 `<name>.breakdown.txt`。
- `run_prof.py` 和 `prof_patch.py` 負責在程式裡插入 NVTX 標記。

腳本裡的路徑寫死在 `/workspace/prof`，請自行修改。

## 注意

- tok/s 重跑大約差 ±10%，例如 `o3_mb64` 兩次是 3337 和 3546。
- 這裡用的是 slack 0.9，和 `results/multimodel/` 的 slack 1.0 不同，兩邊的 tok/s 不能直接比較。
