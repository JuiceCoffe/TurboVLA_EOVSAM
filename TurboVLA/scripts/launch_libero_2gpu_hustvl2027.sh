#!/usr/bin/env bash
set -euo pipefail

PROJECT=/data/hmp/Robotics/TurboVLA_EOVSAM
cd "$PROJECT/TurboVLA"
PYTHON=/data/hmp/conda_envs/turbovla-sam-libero/bin/python

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
IFS=, read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
if [ "${#GPU_IDS[@]}" -ne 2 ]; then
    echo "CUDA_VISIBLE_DEVICES must list exactly two GPUs" >&2
    exit 2
fi
export HF_HOME="$PROJECT/hf_cache"
export TORCH_HOME="$PROJECT/torch_cache"
export PYTHONPATH="$PROJECT/TurboVLA:$PROJECT/TurboVLA/third_party/vla_adapter:$PROJECT/EOVSAM:$PROJECT/EOVSAM/NVlabs/RADIO${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 USE_TF=0 USE_FLAX=0 TF_CPP_MIN_LOG_LEVEL=2
export LIBERO_CONFIG_PATH=/data/hmp/Robotics/LIBERO_CONFIG
export TMPDIR="$PROJECT/outputs/tmp"
export WANDB_CACHE_DIR="$PROJECT/outputs/wandb_cache"
export WANDB_CONFIG_DIR="$PROJECT/outputs/wandb_config"
export OMP_NUM_THREADS=4
mkdir -p "$TMPDIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"
EOVSAM_WEIGHT="$PROJECT/weights/EOVSAM_release_3888_1675_6088_2052_9706.pth"
RADIO_WEIGHT="$TORCH_HOME/hub/checkpoints/c-radio_v4-h_half.pth.tar"
SIGLIP_CACHE="$HF_HOME/hub/models--google--siglip2-giant-opt-patch16-384"
if [ "$(stat -Lc %s "$EOVSAM_WEIGHT" 2>/dev/null || echo 0)" -ne 7693636454 ] ||
   [ "$(stat -Lc %s "$RADIO_WEIGHT" 2>/dev/null || echo 0)" -ne 1681320675 ] ||
   [ ! -s "$SIGLIP_CACHE/refs/main" ]; then
    echo "EOVSAM, RADIO, or SigLIP2 weights are missing or incomplete" >&2
    exit 2
fi

# Two ranks x sixteen samples x four accumulation steps = 128 samples/update.
max_steps=80000
eval_steps=5000
eval_trials=8
eval_max_tasks=-1
eval_suites=libero_spatial,libero_object,libero_goal,libero_10
wandb_mode=online
checkpoint_dir="$PROJECT/outputs/libero_2gpu/checkpoints"
eval_output_dir="$PROJECT/outputs/libero_2gpu/evaluation"
extra_args=()
if [ "${SMOKE:-0}" = 1 ]; then
    max_steps=1
    eval_steps=0
    eval_trials=1
    eval_max_tasks=1
    wandb_mode=disabled
    checkpoint_dir="$PROJECT/outputs/libero_2gpu_smoke/checkpoints"
    extra_args=(--no_save_final)
fi
if [ "${SMOKE_EVAL:-0}" = 1 ]; then
    max_steps=2
    eval_steps=1
    eval_trials=1
    eval_max_tasks=4
    eval_suites=libero_spatial
    wandb_mode=disabled
    checkpoint_dir="$PROJECT/outputs/libero_2gpu_smoke_eval/checkpoints"
    eval_output_dir="$PROJECT/outputs/libero_2gpu_smoke_eval/evaluation"
    extra_args=()
fi
resume_args=(--resume_mode none --run_timestamp "${RUN_TIMESTAMP:-$(date +%Y-%m-%d/%H-%M-%S)}")
if [ "${RESUME_ALL:-0}" = 1 ]; then
    : "${RUN_TIMESTAMP:?Set RUN_TIMESTAMP=YYYY-MM-DD/HH-MM-SS to resume}"
    resume_args=(--resume_mode all --run_timestamp "$RUN_TIMESTAMP")
fi

exec "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 \
    experiments/libero/train.py \
    --dataset_dir /data/hmp/Robotics/TurboVLA/data/libero/libero_10_no_noops/1.0.0 \
    --dataset_dirs /data/hmp/Robotics/TurboVLA/data/libero/libero_10_no_noops/1.0.0,/data/hmp/Robotics/TurboVLA/data/libero/libero_goal_no_noops/1.0.0,/data/hmp/Robotics/TurboVLA/data/libero/libero_object_no_noops/1.0.0,/data/hmp/Robotics/TurboVLA/data/libero/libero_spatial_no_noops/1.0.0 \
    --stats_path experiments/libero/configs/libero_all4_stats.json --stats_key libero_all4_no_noops \
    --dinov3_path /data/hmp/Robotics/TurboVLA_Pretrained/dinov3-vitb16-pretrain-lvd1689m \
    --bert_path /data/hmp/Robotics/TurboVLA_Pretrained/bert-base-uncased \
    --pretrained_init_ckpt /data/hmp/Robotics/TurboVLA_Pretrained/groundingdino_swint_ogc.pth \
    --eovsam_repo "$PROJECT/EOVSAM" \
    --eovsam_weights "$EOVSAM_WEIGHT" \
    --eovsam_train_input_size 512 \
    --batch_size 16 --grad_accum_steps 4 --num_workers "${NUM_WORKERS:-2}" \
    --max_steps "$max_steps" --lr_schedule_steps 80000 --lr 1e-4 --head_lr 1e-4 --dinov3_lr 1e-4 \
    --precision fp32 --dinov3_precision bf16_autocast \
    --checkpoint_dir "$checkpoint_dir" \
    --save_steps 1000 --keep_last_checkpoints 1 --keep_eval_checkpoints \
    --eval_steps "$eval_steps" --eval_distributed --eval_suites "$eval_suites" \
    --eval_num_trials_per_task "$eval_trials" --eval_max_tasks "$eval_max_tasks" \
    --eval_cuda_visible_devices "$CUDA_VISIBLE_DEVICES" --eval_sync_timeout_minutes 4320 \
    --eval_min_available_ram_gb 32 \
    --eval_libero_root /data/hmp/Robotics/LIBERO \
    --eval_output_dir "$eval_output_dir" \
    --wandb_project TurboVLA-EOVSAM-LIBERO --wandb_entity 2016067727-ali \
    --wandb_run_name eovsam-two-gpu-eight-rollouts \
    --wandb_mode "$wandb_mode" --wandb_tags "eovsam,frozen,two-views,ddp,2gpu,8-rollouts" \
    --wandb_dir "$PROJECT/outputs/wandb" \
    "${resume_args[@]}" "${extra_args[@]}"
