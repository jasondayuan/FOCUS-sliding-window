# controls: the pre-feature code (A/B identity check) and a confidence-threshold sweep without the new rules
D=${GSM8K:-/workspace/data/gsm8k/main/test-00000-of-00001.parquet}
M=JetLM/SDAR-1.7B-Chat-b32
OUT=${OUT:-/workspace/FOCUS-sliding-window/results/unmask_feat}
run() {
  repo=$1; tag=$2; shift 2
  echo "$(date +%T) START $tag"
  cd $repo && PYTHONPATH=$repo timeout 1800 python exps/gsm8k_eval.py --model $M --data $D --out $OUT --tag $tag --eager \
    --focus --window-k 4 --focus-min-keep 4 --focus-evict-gate 1 --slack 1.0 "$@" > $OUT/$tag.out 2>&1
  echo "$(date +%T) END $tag rc=$?"
}
run /workspace/orig base_orig
run /workspace/FOCUS-sliding-window base_s0.9 --slack 0.9
run /workspace/FOCUS-sliding-window base_s0.8 --slack 0.8
run /workspace/FOCUS-sliding-window base_s0.7 --slack 0.7
echo "$(date +%T) ALL_DONE"
