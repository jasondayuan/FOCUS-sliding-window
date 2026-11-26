# SDAR-1.7B (LMDeploy, eager, greedy, GSM8K): modes x batch size
cd ${REPO:-/workspace/FOCUS-sliding-window}
D=${GSM8K:-/workspace/data/gsm8k/main/test-00000-of-00001.parquet}
M=JetLM/SDAR-1.7B-Chat-b32
OUT=${OUT:-results/multimodel/sdar}
mkdir -p $OUT
python results/multimodel/test_new_modes.py || { echo TEST_FAIL; exit 1; }
python results/unmask_feat/test_unmask.py || { echo TEST_FAIL; exit 1; }
FW="--focus --window-k 4 --focus-min-keep 4 --focus-evict-gate 1"
declare -A MODES=(
  [vanilla]=""
  [focus]="--focus"
  [fw]="$FW"
  [fw_st05]="$FW --stable-unmask --stable-min-conf 0.5"
  [fw_ff]="$FW --force-first"
  [fw_ff05]="$FW --force-first --force-first-min-conf 0.5"
  [fw_eb01]="$FW --entropy-bound 0.1"
  [fw_eb05]="$FW --entropy-bound 0.5"
)
ORDER="fw fw_st05 fw_ff fw_ff05 fw_eb01 fw_eb05 focus vanilla"
run() {
  tag=$1; shift
  [ -f $OUT/$tag.json ] && { echo "$(date +%T) SKIP $tag"; return; }
  echo "$(date +%T) START $tag"
  timeout 2400 python exps/gsm8k_eval.py --model $M --data $D --out $OUT --tag $tag --eager --slack 1.0 "$@" \
    > $OUT/$tag.out 2>&1
  echo "$(date +%T) END $tag rc=$?"
}
run smoke_eb --num 16 --concurrency 8 $FW --entropy-bound 0.1
grep -q '"accuracy"' $OUT/smoke_eb.json || { echo SMOKE_FAIL; tail -30 $OUT/smoke_eb.out; exit 1; }
for bs in 128 32 8 1; do
  case $bs in 128) N=0;; 32) N=384;; 8) N=128;; 1) N=24;; esac
  for m in $ORDER; do
    run ${m}_bs$bs --concurrency $bs --num $N ${MODES[$m]}
  done
done
echo "$(date +%T) ALL_DONE"
