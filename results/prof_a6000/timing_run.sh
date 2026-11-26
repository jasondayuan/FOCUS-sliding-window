# usage: timing_run.sh <name> <extra env assignments as one string> <gsm8k_eval args...>
name=$1; envs=$2; shift 2
cd /workspace/prof
env PROF_FULL_BS=96 PROF_START=100 PROF_COUNT=150 PROF_EXIT_AFTER=1 PROF_TIMING_OUT=/workspace/prof/$name.timing.json $envs \
  python /workspace/prof/run_prof.py /workspace/FOCUS-sliding-window --model JetLM/SDAR-1.7B-Chat-b32 \
  --data /workspace/data/gsm8k/main/test-00000-of-00001.parquet --out /workspace/prof/out --tag $name --num 640 "$@" \
  > /workspace/prof/$name.log 2>&1
python /workspace/prof/analyze_timing.py /workspace/prof/$name.timing.json 96 100 150 > /workspace/prof/$name.timing.txt 2>&1
echo "DONE $name"
