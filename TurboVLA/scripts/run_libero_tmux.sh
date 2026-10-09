#!/usr/bin/env bash
set -o pipefail
cd /data/hmp/Robotics/TurboVLA_EOVSAM/TurboVLA || exit 1
mkdir -p ../outputs/libero_ddp_full512
bash scripts/launch_libero_ddp.sh full512 16 2>&1 | tee -a ../outputs/libero_ddp_full512/launch.log
train_exit=${PIPESTATUS[0]}
printf '\nTraining exited with status %s. This tmux window remains available.\n' "$train_exit"
exit "$train_exit"
