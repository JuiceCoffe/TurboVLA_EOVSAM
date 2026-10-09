#!/usr/bin/env bash
set -euo pipefail
cd /data/hmp/Robotics/TurboVLA_EOVSAM/TurboVLA
export CUDA_VISIBLE_DEVICES=2,3
export USE_TF=0 USE_FLAX=0 HF_HUB_OFFLINE=1 TF_CPP_MIN_LOG_LEVEL=2 OMP_NUM_THREADS=4
export LIBERO_CONFIG_PATH=/data/hmp/Robotics/TurboVLA_libero_config
export TMPDIR=/data/hmp/Robotics/TurboVLA_EOVSAM/outputs/tmp
export WANDB_CACHE_DIR=/data/hmp/Robotics/TurboVLA_EOVSAM/outputs/wandb_cache
export WANDB_CONFIG_DIR=/data/hmp/Robotics/TurboVLA_EOVSAM/outputs/wandb_config
mkdir -p "$TMPDIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"
mode=${1:-full}
batch=${2:-8}
accum=$((64 / batch))
test "$((2 * batch * accum))" -eq 128
steps=80000
trials=50
tasks=-1
name=eovsam-two-view-512-ddp
eval_steps=5000
entry=experiments/libero/train.py
resume_args=()
if [ "${RESUME_ALL:-0}" = 1 ]; then
    resume_args=(--resume_mode all --run_timestamp '2026-09-26/18-17-10')
    export WANDB_RUN_ID=eqklsgsz
    export WANDB_RESUME=must
fi
if [ "$mode" = smoke ]; then
    steps=2
    trials=1
    tasks=1
    name=eovsam-ddp-preflight
fi
if [ "$mode" = capacity ]; then
    steps=2
    eval_steps=0
    name=eovsam-ddp-batch16-capacity
    entry=scripts/profile_libero_ddp.py
fi
mkdir -p "../outputs/libero_ddp_${mode}"
echo $$ > "../outputs/libero_ddp_${mode}/launcher.pid"
exec /data/hmp/conda_envs/turbovla-sam-libero/bin/torchrun --standalone --nproc_per_node=2 "$entry" \
    --dataset_dir /data/hmp/Robotics/TurboVLA/data/libero/libero_10_no_noops/1.0.0 \
    --dataset_dirs /data/hmp/Robotics/TurboVLA/data/libero/libero_10_no_noops/1.0.0,/data/hmp/Robotics/TurboVLA/data/libero/libero_goal_no_noops/1.0.0,/data/hmp/Robotics/TurboVLA/data/libero/libero_object_no_noops/1.0.0,/data/hmp/Robotics/TurboVLA/data/libero/libero_spatial_no_noops/1.0.0 \
    --stats_path experiments/libero/configs/libero_all4_stats.json --stats_key libero_all4_no_noops \
    --dinov3_path /data/hmp/Robotics/TurboVLA_Pretrained/dinov3-vitb16-pretrain-lvd1689m \
    --bert_path /data/hmp/Robotics/TurboVLA_Pretrained/bert-base-uncased \
    --pretrained_init_ckpt /data/hmp/Robotics/TurboVLA_Pretrained/groundingdino_swint_ogc.pth \
    --batch_size "$batch" --grad_accum_steps "$accum" \
    --max_steps "$steps" --lr_schedule_steps 80000 \
    --precision fp32 --dinov3_precision bf16_autocast \
    --eovsam_train_input_size 512 \
    --checkpoint_dir "../outputs/libero_ddp_${mode}/checkpoints" \
    "${resume_args[@]}" \
    --keep_last_checkpoints 3 \
    --eval_steps "$eval_steps" --eval_num_trials_per_task "$trials" --eval_max_tasks "$tasks" \
    --eval_cuda_visible_devices 2 --eval_sync_timeout_minutes 4320 \
    --eval_output_dir "../outputs/libero_ddp_${mode}/evaluation" \
    --wandb_project TurboVLA-EOVSAM-LIBERO --wandb_run_name "$name" \
    --wandb_mode online --wandb_tags "eovsam,frozen,two-views,ddp,$mode" \
    --wandb_dir ../outputs/wandb
