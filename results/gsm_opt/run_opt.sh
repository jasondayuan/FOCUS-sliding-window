# Infra ablation on FOCUS + window (SDAR-1.7B, GSM8K, all 1319, bs=128, RTX A6000): each run adds one optimization.
# Rebuilt from the args stored in the .json results (the original script was not kept). Needs commit ec04648c or later.
# Usage: OUT=results/repro/gsm_opt bash results/gsm_opt/run_opt.sh
cd ${REPO:-/workspace/FOCUS-sliding-window}
D=${GSM8K:-/workspace/data/gsm8k/main/test-00000-of-00001.parquet}
M=JetLM/SDAR-1.7B-Chat-b32
OUT=${OUT:-results/gsm_opt}
mkdir -p $OUT
FW="--focus --window-k 4 --focus-min-keep 4 --focus-evict-gate 1 --slack 0.9"
run() {
  tag=$1; shift
  [ -f $OUT/$tag.json ] && { echo "$(date +%T) SKIP $tag"; return; }
  echo "$(date +%T) START $tag"
  timeout 1800 python exps/gsm8k_eval.py --model $M --data $D --out $OUT --tag $tag --concurrency 128 $FW "$@" \
    > $OUT/$tag.out 2>&1
  echo "$(date +%T) END $tag rc=$?"
}
run ref_fw4_s09 --eager
run b_flags --eager --fast-host --fast-unmask --masked-logits
run o1_graph --fast-host --fast-unmask --masked-logits
run o2_prefill16 --fast-host --fast-unmask --masked-logits --prefill-min-waiting 16
# no --masked-logits with micro-batching: its row selection (nonzero) blocks the host until the forward is done
run o3_mb64 --fast-host --fast-unmask --prefill-min-waiting 16 --decode-micro-batch 64
echo "$(date +%T) ALL_DONE"
