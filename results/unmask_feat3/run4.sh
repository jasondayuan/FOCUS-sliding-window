# GSM8K, SDAR-1.7B, sliding window (k=4) on the delayed cache, s=1.0 unless noted.
# 0: does FOCUS help (window only / + FOCUS as tuned / + FOCUS defaults), eager and optimized stack
# A: unmasking rules (eager). B: window size on the optimized stack.
cd ${REPO:-/workspace/FOCUS-sliding-window}
D=${GSM8K:-/workspace/data/gsm8k/main/test-00000-of-00001.parquet}
M=JetLM/SDAR-1.7B-Chat-b32
OUT=${OUT:-results/unmask_feat3}
FW="--focus --focus-min-keep 4 --focus-evict-gate 1"   # FOCUS as tuned for the window (barely evicts)
FD="--focus"                                           # FOCUS defaults (alpha 1.5, no min-keep / gate)
WO="--delayed-cache"                                   # window only, no FOCUS
OPT="--fast-unmask --fast-host --prefill-min-waiting 16 --decode-micro-batch 64"
run() {
  tag=$1; shift
  echo "$(date +%T) START $tag"
  timeout 1800 python exps/gsm8k_eval.py --model $M --data $D --out $OUT --tag $tag --slack 1.0 "$@" > $OUT/$tag.out 2>&1
  echo "$(date +%T) END $tag rc=$?"
}
run W_O_k6 $WO --window-k 6 $OPT
run W_O_k8 $WO --window-k 8 $OPT
run W_O_adapt $WO --window-k 4 --window-adapt --window-k-max 8 $OPT
run O_s0.9 $FW --window-k 4 $OPT --slack 0.9
run O_s0.8 $FW --window-k 4 $OPT --slack 0.8
echo "$(date +%T) ALL_DONE"
