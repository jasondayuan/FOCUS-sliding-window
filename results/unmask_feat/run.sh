# GSM8K, SDAR-1.7B, FOCUS + window (fw4_mk4_gate1, s=1.0, eager = the original version) +- the two unmasking rules
cd ${REPO:-/workspace/FOCUS-sliding-window}
D=${GSM8K:-/workspace/data/gsm8k/main/test-00000-of-00001.parquet}
M=JetLM/SDAR-1.7B-Chat-b32
OUT=${OUT:-results/unmask_feat}
run() {
  tag=$1; shift
  echo "$(date +%T) START $tag"
  timeout 1800 python exps/gsm8k_eval.py --model $M --data $D --out $OUT --tag $tag --eager \
    --focus --window-k 4 --focus-min-keep 4 --focus-evict-gate 1 --slack 1.0 "$@" > $OUT/$tag.out 2>&1
  echo "$(date +%T) END $tag rc=$?"
}
run smoke_ff_st --num 32 --force-first --stable-unmask
grep -q '"accuracy"' $OUT/smoke_ff_st.json || { echo SMOKE_FAIL; tail -30 $OUT/smoke_ff_st.out; exit 1; }
run base
run ff --force-first
run st --stable-unmask
run ff_st --force-first --stable-unmask
echo "$(date +%T) ALL_DONE"
