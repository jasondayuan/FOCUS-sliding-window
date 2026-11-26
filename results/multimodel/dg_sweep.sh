# DiffusionGemma-26B-A4B-it (transformers 5.23, A100 80GB, greedy, GSM8K): modes x batch size (bs:num = 1:8, 4:32, 8:64)
cd ${REPO:-/workspace/FOCUS-sliding-window}
D=${GSM8K:-/workspace/data/gsm8k/main/test-00000-of-00001.parquet}
OUT=${OUT:-results/multimodel/dg}
RUNS=""
for m in eb eb_stable0.5 eb_ff thr0.9 eb_focus eb_focus_window64 eb_focus_window64_stable0.5 eb_focus_window64_ff; do
  RUNS="$RUNS $m:1:8 $m:4:32 $m:8:64"
done
RUNS="$RUNS eb_focus_age2:1:8 eb_focus_age2:8:64 eb_focus_window64_age2:1:8 eb_focus_window64_age2:8:64"
# few threads: the CPU side oversubscribes and stalls on many-core hosts otherwise
OMP_NUM_THREADS=4 python results/multimodel/test_dg_tiny.py || { echo TEST_FAIL; exit 1; }
OMP_NUM_THREADS=4 python exps/multimodel/dg_eval.py --data $D --out $OUT --runs $RUNS
