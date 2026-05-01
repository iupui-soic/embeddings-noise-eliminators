#!/usr/bin/env bash
# Wait for both NIH GPU drivers to finish, merge parquets, dual-GPU MIMIC,
# merge MIMIC parquets, then sensitivity sweep (exp10-16).
set -u
LOG_ROOT=/home/saptpurk/embeddings-noise-eliminators/v4_work
WATCHER_LOG="$LOG_ROOT/chain_watcher.log"
echo "[$(date -Is)] chain watcher v3 armed (NIH -> MIMIC -> sensitivity)" >> "$WATCHER_LOG"

# Kill orphaned ipykernels left over by nbconvert so the next stage starts with clean VRAM.
# Safe to call only when no nbconvert is running.
cleanup_orphan_kernels() {
  if pgrep -u "$USER" -f 'jupyter-nbconvert' >/dev/null 2>&1; then
    echo "[$(date -Is)] cleanup skipped: nbconvert still active" >> "$WATCHER_LOG"
    return
  fi
  local orphans
  orphans=$(pgrep -u "$USER" -f 'ipykernel_launcher' || true)
  if [[ -n "$orphans" ]]; then
    echo "[$(date -Is)] killing orphan ipykernels: $(echo $orphans | tr '\n' ' ')" >> "$WATCHER_LOG"
    echo "$orphans" | xargs -r kill -9 2>/dev/null || true
    sleep 10
    nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv >> "$WATCHER_LOG" 2>&1
  fi
}

while ! grep -q "^DONE_EXIT_" "$LOG_ROOT/nih_gpu0_driver.log" 2>/dev/null; do sleep 60; done
while ! grep -q "^DONE_EXIT_" "$LOG_ROOT/nih_gpu1_driver.log" 2>/dev/null; do sleep 60; done
echo "[$(date -Is)] both NIH drivers done" >> "$WATCHER_LOG"

bash "$LOG_ROOT/merge_gpu_parquets.sh" nih >> "$WATCHER_LOG" 2>&1
echo "[$(date -Is)] NIH parquets merged" >> "$WATCHER_LOG"

cleanup_orphan_kernels

tmux kill-session -t mimic-gpu0 2>/dev/null
tmux kill-session -t mimic-gpu1 2>/dev/null
tmux new-session -d -s mimic-gpu0 "bash $LOG_ROOT/run_mimic_exps_gpu0.sh > $LOG_ROOT/mimic_gpu0_driver.log 2>&1; echo DONE_EXIT_\$? >> $LOG_ROOT/mimic_gpu0_driver.log"
tmux new-session -d -s mimic-gpu1 "bash $LOG_ROOT/run_mimic_exps_gpu1.sh > $LOG_ROOT/mimic_gpu1_driver.log 2>&1; echo DONE_EXIT_\$? >> $LOG_ROOT/mimic_gpu1_driver.log"
echo "[$(date -Is)] MIMIC dual-GPU sessions launched" >> "$WATCHER_LOG"

while ! grep -q "^DONE_EXIT_" "$LOG_ROOT/mimic_gpu0_driver.log" 2>/dev/null; do sleep 120; done
while ! grep -q "^DONE_EXIT_" "$LOG_ROOT/mimic_gpu1_driver.log" 2>/dev/null; do sleep 120; done
echo "[$(date -Is)] both MIMIC drivers done" >> "$WATCHER_LOG"

bash "$LOG_ROOT/merge_gpu_parquets.sh" mimic >> "$WATCHER_LOG" 2>&1
echo "[$(date -Is)] MIMIC parquets merged" >> "$WATCHER_LOG"

cleanup_orphan_kernels

tmux kill-session -t sensitivity 2>/dev/null
tmux new-session -d -s sensitivity "bash $LOG_ROOT/run_sensitivity_exps.sh > $LOG_ROOT/sensitivity_driver.log 2>&1; echo DONE_EXIT_\$? >> $LOG_ROOT/sensitivity_driver.log"
echo "[$(date -Is)] sensitivity sweep launched" >> "$WATCHER_LOG"

while ! grep -q "^DONE_EXIT_" "$LOG_ROOT/sensitivity_driver.log" 2>/dev/null; do sleep 120; done
echo "[$(date -Is)] sensitivity sweep done" >> "$WATCHER_LOG"

echo "DONE_EXIT_0" >> "$WATCHER_LOG"
