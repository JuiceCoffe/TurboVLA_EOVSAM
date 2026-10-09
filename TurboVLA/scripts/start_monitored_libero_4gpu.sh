#!/usr/bin/env bash
set -euo pipefail
PROJECT=/data/hmp/Robotics/TurboVLA_EOVSAM
cd "$PROJECT/TurboVLA"
# This entry point always starts a fresh experiment.
unset WANDB_RUN_ID WANDB_RESUME RUN_TIMESTAMP RESUME_ALL SMOKE SMOKE_EVAL
export RUN_TIMESTAMP="$(date +%Y-%m-%d/%H-%M-%S)"
export NUM_WORKERS=2
export RESUME_ALL=0
exec /data/hmp/conda_envs/turbovla-sam-libero/bin/python -u scripts/monitor_training_memory.py \
    --output-dir "$PROJECT/outputs/libero_4gpu/monitoring/$RUN_TIMESTAMP" \
    --interval 30 --pss-interval 300 --low-memory-gb 40 \
    -- bash scripts/launch_libero_4gpu_hustvl40.sh
