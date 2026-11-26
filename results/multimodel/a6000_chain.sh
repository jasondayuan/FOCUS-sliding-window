# A6000: I-DLM setup (network / disk, niced) alongside the SDAR sweep, then the I-DLM sweep
R=${REPO:-/workspace/FOCUS-sliding-window}
L=/workspace/mm_logs
mkdir -p $L
nice -n 19 bash $R/results/multimodel/idlm_sweep.sh setup > $L/idlm_setup.log 2>&1 &
SETUP=$!
bash $R/results/multimodel/sdar_sweep.sh > $L/sdar.log 2>&1
wait $SETUP
bash $R/results/multimodel/idlm_sweep.sh > $L/idlm.log 2>&1
echo "$(date +%T) CHAIN_DONE" >> $L/sdar.log
