#!/usr/bin/env bash
set -euo pipefail

PROJECT=/data/hmp/Robotics/TurboVLA_EOVSAM
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_HOME="$PROJECT/hf_cache"
export TORCH_HOME="$PROJECT/torch_cache"
export PYTHONPATH="$PROJECT/TurboVLA:$PROJECT/TurboVLA/third_party/vla_adapter:$PROJECT/EOVSAM:$PROJECT/EOVSAM/NVlabs/RADIO${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 USE_TF=0 USE_FLAX=0 TF_CPP_MIN_LOG_LEVEL=2
export LIBERO_CONFIG_PATH=/data/hmp/Robotics/LIBERO_CONFIG
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=4
export TMPDIR="$PROJECT/outputs/tmp"

cd "$PROJECT/TurboVLA"
exec /data/hmp/conda_envs/turbovla-sam-libero/bin/python scripts/evaluate_checkpoint_4gpu.py \
    --checkpoint "$PROJECT/outputs/libero_4gpu/checkpoints/2026-09-29/18-18-15/turbovla_step_10000.pth" \
    --output_dir "$PROJECT/outputs/libero_4gpu/evaluation_recheck/2026-09-29/18-18-15" \
    --min_available_ram_gb 32
