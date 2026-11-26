# dllm_gpu_multistep go/no-go on SDAR-1.7B (LMDeploy, greedy, GSM8K): unit tests, bs=1 identity, then K=1 vs K=4
# Usage (on the GPU box): bash results/multimodel/multistep_bench.sh [eager|graph]
cd ${REPO:-/workspace/FOCUS-sliding-window}
D=${GSM8K:-/workspace/data/gsm8k/main/test-00000-of-00001.parquet}
M=JetLM/SDAR-1.7B-Chat-b32
G=${1:-eager}
OUT=${OUT:-results/multimodel/multistep_$G}
mkdir -p $OUT
EG=""; [ "$G" = eager ] && EG="--eager"
python results/multimodel/test_multistep.py || { echo TEST_FAIL; exit 1; }
python results/multimodel/test_new_modes.py || { echo TEST_FAIL; exit 1; }
python results/unmask_feat/test_unmask.py || { echo TEST_FAIL; exit 1; }
FL="--fast-host --fast-unmask --masked-logits"
DW="--delayed-cache --window-k 4 $FL"
declare -A MODES=(
  [vanilla]=""
  [dw_k1]="$DW"
  [dw_k4]="$DW --gpu-multistep 4"
  [fw_k1]="--focus --window-k 4 --focus-min-keep 4 --focus-evict-gate 1 $FL"
)
run() {
  tag=$1; shift
  [ -f $OUT/$tag.json ] && { echo "$(date +%T) SKIP $tag"; return; }
  echo "$(date +%T) START $tag"
  timeout 2400 python exps/gsm8k_eval.py --model $M --data $D --out $OUT --tag $tag $EG --slack 1.0 "$@" \
    > $OUT/$tag.out 2>&1
  echo "$(date +%T) END $tag rc=$?"
}
# identity: one sequence at a time, the GPU-built inputs must reproduce the host-built ones token for token
run id_k1 --num 16 --concurrency 1 $DW
run id_k4 --num 16 --concurrency 1 $DW --gpu-multistep 4
python - $OUT <<'EOF' || exit 1
import json, sys
a, b = ([json.loads(l) for l in open(f'{sys.argv[1]}/{t}.samples.jsonl')] for t in ('id_k1', 'id_k4'))
same = sum(x['output'] == y['output'] for x, y in zip(a, b))
print(f'IDENTITY {same}/{len(a)} outputs identical')
sys.exit(0 if same == len(a) == 16 else 1)
EOF
for bs in 128 32 8 1; do
  case $bs in 128) N=0;; 32) N=384;; 8) N=128;; 1) N=24;; esac
  for m in dw_k4 dw_k1 fw_k1 vanilla; do
    run ${m}_bs$bs --concurrency $bs --num $N ${MODES[$m]}
  done
done
python - $OUT <<'EOF'
import json, os, sys
o = sys.argv[1]
print('| mode | ' + ' | '.join(f'bs={b} tok/s' for b in (1, 8, 32, 128)) + ' | acc bs128 | tok/step bs128 |')
print('|---|---|---|---|---|---|---|')
for m in ('vanilla', 'fw_k1', 'dw_k1', 'dw_k4'):
    r = {b: json.load(open(p)) for b in (1, 8, 32, 128) if os.path.exists(p := f'{o}/{m}_bs{b}.json')}
    c = [f"{r[b]['out_tok_per_s']:.0f}" if b in r else '' for b in (1, 8, 32, 128)]
    t = r.get(128, {})
    print(f'| {m} | ' + ' | '.join(c) + f" | {100 * t.get('accuracy', 0):.1f} | {t.get('tokens_per_step', 0):.2f} |")
EOF
echo "$(date +%T) ALL_DONE"
