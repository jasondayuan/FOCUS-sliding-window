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
# smoke: the new paths (budget, EB, adaptive window on CUDA graph + micro-batch, window without FOCUS)
run smoke_b --num 32 --eager $FW --window-k 4 --unmask-budget 0.2
run smoke_eb --num 32 --eager $FW --window-k 4 --entropy-bound 0.3
run smoke_opt --num 32 $FW --window-k 4 --window-adapt --window-k-max 8 $OPT
run smoke_wo --num 32 $WO --window-k 4 $OPT
for t in smoke_b smoke_eb smoke_opt smoke_wo; do
  grep -q '"accuracy"' $OUT/$t.json || { echo "SMOKE_FAIL $t"; tail -30 $OUT/$t.out; exit 1; }
done
# 0: FOCUS or not
run E_base --eager $FW --window-k 4
run W_E --eager $WO --window-k 4
run F_E --eager $FD --window-k 4
run O_k4 $FW --window-k 4 $OPT
run O_k4_r2 $FW --window-k 4 $OPT
run W_O $WO --window-k 4 $OPT
run W_O_r2 $WO --window-k 4 $OPT
run F_O $FD --window-k 4 $OPT
# A: eager, k=4
for s in 0.9 0.8 0.7; do run E_s$s --eager $FW --window-k 4 --slack $s; done
for b in 0.1 0.2 0.3 0.5; do run E_b$b --eager $FW --window-k 4 --unmask-budget $b; done
for g in 0.1 0.3 1.0; do run E_eb$g --eager $FW --window-k 4 --entropy-bound $g; done
# B: window size, optimized stack (CUDA graph + fast host/unmask + prefill coalescing + micro-batch 64) and eager
run O_k6 $FW --window-k 6 $OPT
run O_k8 $FW --window-k 8 $OPT
run O_adapt $FW --window-k 4 --window-adapt --window-k-max 8 $OPT
run E_k8 --eager $FW --window-k 8
run E_adapt --eager $FW --window-k 4 --window-adapt --window-k-max 8
# budget on the optimized stack (it needs no host sync)
run O_b0.2 $FW --window-k 4 --unmask-budget 0.2 $OPT
run O_b0.3 $FW --window-k 4 --unmask-budget 0.3 $OPT
echo "$(date +%T) ALL_DONE"
