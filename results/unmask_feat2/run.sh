# GSM8K, SDAR-1.7B, FOCUS + window (fw4_mk4_gate1, s=1.0, eager): filters on rule 2 (stable unmask), with controls
cd ${REPO:-/workspace/FOCUS-sliding-window}
D=${GSM8K:-/workspace/data/gsm8k/main/test-00000-of-00001.parquet}
M=JetLM/SDAR-1.7B-Chat-b32
OUT=${OUT:-results/unmask_feat2}
run() {
  tag=$1; shift
  echo "$(date +%T) START $tag"
  timeout 1800 python exps/gsm8k_eval.py --model $M --data $D --out $OUT --tag $tag --eager \
    --focus --window-k 4 --focus-min-keep 4 --focus-evict-gate 1 --slack 1.0 "$@" > $OUT/$tag.out 2>&1
  echo "$(date +%T) END $tag rc=$?"
}
run smoke --num 32 --stable-unmask --stable-ctx 1 --stable-min-conf 0.5
grep -q '"accuracy"' $OUT/smoke.json || { echo SMOKE_FAIL; tail -30 $OUT/smoke.out; exit 1; }
run base
run st --stable-unmask
run st_c0.5 --stable-unmask --stable-min-conf 0.5
run st_c0.7 --stable-unmask --stable-min-conf 0.7
run st_n1 --stable-unmask --stable-ctx 1
run st_n2 --stable-unmask --stable-ctx 2
run st_n1_c0.5 --stable-unmask --stable-ctx 1 --stable-min-conf 0.5
run base_s0.9 --slack 0.9
run base_s0.8 --slack 0.8
run base_r2
echo "$(date +%T) ALL_DONE"
