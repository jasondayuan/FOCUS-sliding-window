# usage: nsys_run.sh <name> <gsm8k_eval args...>
name=$1; shift
cd /workspace/prof
PROF_FULL_BS=96 PROF_START=150 PROF_COUNT=30 PROF_TIMING_OUT=/workspace/prof/$name.timing.json \
/opt/nvidia/nsight-systems/2025.5.2/bin/nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi --capture-range-end=stop-shutdown --force-overwrite=true -o /workspace/prof/$name \
  python /workspace/prof/run_prof.py /workspace/FOCUS-sliding-window --model JetLM/SDAR-1.7B-Chat-b32 \
  --data /workspace/data/gsm8k/main/test-00000-of-00001.parquet --out /workspace/prof/out --tag $name --num 640 "$@" \
  > /workspace/prof/$name.log 2>&1
/opt/nvidia/nsight-systems/2025.5.2/bin/nsys export --type sqlite --force-overwrite true -o /workspace/prof/$name.sqlite /workspace/prof/$name.nsys-rep >> /workspace/prof/$name.log 2>&1
python /workspace/prof/analyze_nsys.py /workspace/prof/$name.sqlite /workspace/prof/$name.breakdown.json > /workspace/prof/$name.breakdown.txt 2>&1
echo "DONE $name"
