# I-DLM-8B (official SGLang ISD + patch, greedy, no thinking, GSM8K): ISD configs x concurrency
set -u
W=${W:-/workspace}
R=${REPO:-$W/FOCUS-sliding-window}
OUT=${OUT:-$R/results/multimodel/idlm}
D=${GSM8K:-$W/data/gsm8k/main/test-00000-of-00001.parquet}
mkdir -p $OUT/cfg
cd $W
if [ ! -x $W/venv-idlm/bin/python ]; then
  [ -d I-DLM ] || git clone --depth 1 https://github.com/Introspective-Diffusion/I-DLM.git > $W/idlm_clone.log 2>&1
  pip install uv > /dev/null 2>&1
  uv venv -p 3.12 $W/venv-idlm > $W/idlm_venv.log 2>&1
  VIRTUAL_ENV=$W/venv-idlm uv pip install -e I-DLM/inference/sglang pandas pyarrow requests pyyaml > $W/idlm_pip.log 2>&1 \
    || { echo "$(date +%T) PIP_FAIL"; tail -20 $W/idlm_pip.log; exit 1; }
  $W/venv-idlm/bin/python $R/exps/multimodel/patch_idlm.py I-DLM/inference/sglang/sglang/srt/dllm/algorithm/idlm_blockN.py
fi
PY=$W/venv-idlm/bin/python
$PY -c "from sglang.srt.dllm.algorithm.idlm_blockN import IDLMBlockN; import torch; print('IDLM_OK', torch.__version__)" \
  || { echo IMPORT_FAIL; exit 1; }
grep -q verify_min_prob I-DLM/inference/sglang/sglang/srt/dllm/algorithm/idlm_blockN.py || { echo NOT_PATCHED; exit 1; }
HF_HUB_ENABLE_HF_TRANSFER=0 hf download yifanyu/I-DLM-8B > $W/dl_idlm.log 2>&1 || { echo DL_FAIL; exit 1; }
echo "$(date +%T) SETUP_OK"
[ "${1:-}" = setup ] && exit 0

cfg() {  # name block gen extra...
  name=$1; b=$2; g=$3; shift 3
  { echo "block_size: $b"; echo "gen_block_size: $g"; echo "confidence_threshold: 0.0"; echo "temperature: 0.0"
    echo "top_k: 50"; echo "top_p: 0.95"; for kv in "$@"; do echo "$kv"; done; } > $OUT/cfg/$name.yaml
}
cfg n4 7 4 "use_spec_verify: true"
cfg n2 3 2 "use_spec_verify: true"
cfg n8 15 8 "use_spec_verify: true"
cfg n4_noverify 7 4 "use_spec_verify: false"
cfg n4_top5 7 4 "use_spec_verify: true" "fast_verify: true" "fast_verify_topk: 5"
cfg n4_agree05 7 4 "use_spec_verify: true" "verify_min_prob: 0.5"

for name in n4 n2 n8 n4_noverify n4_top5 n4_agree05; do
  done_all=1
  for bs in 64 32 8 1; do [ -f $OUT/${name}_bs$bs.json ] || done_all=0; done
  [ $done_all = 1 ] && { echo "$(date +%T) SKIP $name"; continue; }
  export IDLM_STATS_PATH=$OUT/${name}.stats.json
  rm -f $IDLM_STATS_PATH
  echo "$(date +%T) SERVER $name"
  $PY -m sglang.launch_server --model-path yifanyu/I-DLM-8B --trust-remote-code --tp-size 1 --dtype bfloat16 \
    --mem-fraction-static 0.85 --max-running-requests 64 --attention-backend flashinfer \
    --dllm-algorithm IDLMBlockN --dllm-algorithm-config $OUT/cfg/$name.yaml --port 30000 > $OUT/${name}.server.log 2>&1 &
  SP=$!
  ok=0
  for i in $(seq 1 120); do
    curl -s -m 5 http://127.0.0.1:30000/v1/models > /dev/null 2>&1 && { ok=1; break; }
    kill -0 $SP 2> /dev/null || break
    sleep 5
  done
  if [ $ok != 1 ]; then echo "$(date +%T) SERVER_FAIL $name"; tail -30 $OUT/${name}.server.log; kill $SP 2>/dev/null; wait $SP 2>/dev/null; continue; fi
  for bs in 64 32 8 1; do
    case $bs in 64) N=512;; 32) N=256;; 8) N=96;; 1) N=24;; esac
    [ -f $OUT/${name}_bs$bs.json ] && continue
    echo "$(date +%T) START ${name}_bs$bs"
    timeout 1800 $PY $R/exps/multimodel/idlm_client.py --data $D --out $OUT --tag ${name}_bs$bs --concurrency $bs --num $N \
      --stats-path $IDLM_STATS_PATH > $OUT/${name}_bs$bs.out 2>&1
    echo "$(date +%T) END ${name}_bs$bs rc=$?"
  done
  kill $SP; wait $SP 2>/dev/null
  sleep 5
done
echo "$(date +%T) ALL_DONE"
