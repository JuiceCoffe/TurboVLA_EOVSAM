import argparse
from datetime import datetime, timedelta
import glob
import json
import math
import os
import random
import re
import subprocess
import sys

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.libero_rlds import (
    LiberoRLDSDataset,
    vla_collate_fn,
)
from ..models.turbovla import (
    build_turbovla,
)


class DummyArgs:
    pass


def build_wandb_run_name(base_name, timestamp="", now=None):
    timestamp = timestamp or (now or datetime.now()).strftime("%Y-%m-%d/%H-%M-%S")
    return f"{base_name.rstrip('/')}/{timestamp}" if base_name else timestamp


def configure_timestamped_output_dirs(args, is_distributed, rank):
    timestamp = args.run_timestamp.strip().strip("/")
    if is_distributed:
        payload = [timestamp or datetime.now().strftime("%Y-%m-%d/%H-%M-%S") if rank == 0 else None]
        dist.broadcast_object_list(payload, src=0)
        timestamp = payload[0]
    elif not timestamp:
        timestamp = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")

    try:
        datetime.strptime(timestamp, "%Y-%m-%d/%H-%M-%S")
    except ValueError as exc:
        raise ValueError(
            f"--run_timestamp must use YYYY-MM-DD/HH-MM-SS, got {timestamp!r}"
        ) from exc

    args.run_timestamp = timestamp
    path_parts = timestamp.split("/")
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, *path_parts)
    args.eval_output_dir = os.path.join(args.eval_output_dir, *path_parts)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the complete TurboVLA model with online BERT and DINOv3."
    )

    parser.add_argument("--dataset_dir", type=str, default="./data/libero/libero_10_no_noops/1.0.0")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--dinov3_path", type=str, required=True)
    parser.add_argument("--bert_path", type=str, required=True)
    parser.add_argument(
        "--pretrained_init_ckpt",
        type=str,
        required=True,
    )
    parser.add_argument("--checkpoint_dir", type=str, default="outputs/checkpoints")
    parser.add_argument("--checkpoint_prefix", type=str, default="turbovla_step")
    parser.add_argument("--resume_mode", type=str, default="none", choices=["none", "model", "all"])
    parser.add_argument(
        "--run_timestamp",
        type=str,
        default="",
        help="Output subdirectory in YYYY-MM-DD/HH-MM-SS format; generated automatically when omitted.",
    )

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-10)
    parser.add_argument("--head_lr", type=float, default=5e-5)
    parser.add_argument("--dinov3_lr", type=float, default=5e-5)
    parser.add_argument("--head_weight_decay", type=float, default=1e-10)
    parser.add_argument("--dinov3_weight_decay", type=float, default=1e-10)
    parser.add_argument("--precision", type=str, default="fp32", choices=["fp32", "bf16_amp"])
    parser.add_argument("--max_steps", type=int, default=80000)
    parser.add_argument(
        "--lr_schedule_steps",
        type=int,
        default=None,
        help="Cosine LR horizon. Defaults to max_steps; set separately for multi-fidelity searches.",
    )
    parser.add_argument("--warmup_steps", type=int, default=10000)
    parser.add_argument("--min_lr_ratio", type=float, default=1.0)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--keep_last_checkpoints", type=int, default=0,
                        help="Keep N recent checkpoints plus every evaluation checkpoint; 0 keeps all.")
    parser.add_argument("--keep_eval_checkpoints", action="store_true", default=True)
    parser.add_argument("--no_keep_eval_checkpoints", dest="keep_eval_checkpoints", action="store_true",
                        help="Legacy option accepted for compatibility; evaluation checkpoints are always retained.")
    parser.add_argument("--log_freq", type=int, default=20)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--wandb_project", type=str, default=os.environ.get("WANDB_PROJECT", ""))
    parser.add_argument("--wandb_entity", type=str, default=os.environ.get("WANDB_ENTITY", ""))
    parser.add_argument("--wandb_run_name", type=str, default=os.environ.get("WANDB_NAME", ""))
    parser.add_argument("--wandb_group", type=str, default=os.environ.get("WANDB_RUN_GROUP", ""))
    parser.add_argument("--wandb_tags", type=str, default="")
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default=os.environ.get("WANDB_MODE", "online"),
        choices=["online", "offline", "disabled"],
    )
    parser.add_argument("--wandb_dir", type=str, default=os.environ.get("WANDB_DIR", "outputs/wandb"))
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=0,
        help="Run LIBERO rollout evaluation every N optimizer steps and at the final step; 0 disables it.",
    )
    parser.add_argument(
        "--eval_suites",
        type=str,
        default="libero_spatial,libero_object,libero_goal,libero_10",
        help="Comma-separated LIBERO suites evaluated independently.",
    )
    parser.add_argument("--eval_num_trials_per_task", type=int, default=50)
    parser.add_argument("--eval_max_tasks", type=int, default=-1)
    parser.add_argument(
        "--eval_distributed", action="store_true",
        help="Shard all LIBERO tasks across the training ranks and their GPUs.",
    )
    parser.add_argument("--eval_seed", type=int, default=7)
    parser.add_argument("--eval_precision", type=str, default="bf16", choices=["bf16", "fp32"])
    parser.add_argument("--eval_output_dir", type=str, default="outputs/evaluation")
    parser.add_argument("--eval_libero_root", type=str, default="/data/hmp/Robotics/LIBERO")
    parser.add_argument("--eval_mujoco_gl", type=str, default="egl")
    parser.add_argument("--eval_pyopengl_platform", type=str, default="egl")
    parser.add_argument(
        "--eval_sync_timeout_minutes",
        type=int,
        default=720,
        help="Timeout for the CPU/Gloo process group used during rollout evaluation.",
    )
    parser.add_argument(
        "--eval_cuda_visible_devices",
        type=str,
        default="",
        help="Optional CUDA_VISIBLE_DEVICES override for evaluation subprocesses.",
    )
    parser.add_argument(
        "--eval_min_available_ram_gb", type=float, default=32.0,
        help="Wait before loading another evaluation model if host available RAM is lower.",
    )

    parser.set_defaults(save_final=True)
    parser.add_argument("--save_final", dest="save_final", action="store_true")
    parser.add_argument("--no_save_final", dest="save_final", action="store_false")

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--shuffle_buffer", type=int, default=512)
    parser.add_argument("--step_mix_buffer_size", type=int, default=64)
    parser.add_argument("--expected_image_size", type=int, default=256)

    parser.add_argument("--use_eovsam", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eovsam_repo", default="/data/hmp/Robotics/TurboVLA_EOVSAM/EOVSAM")
    parser.add_argument("--eovsam_weights", default="/data/hmp/SAMweights/EOVSAM_release_3888_1675_6088_2052_9706.pth")
    parser.add_argument("--eovsam_images_per_batch", type=int, default=1)
    parser.add_argument("--eovsam_train_input_size", type=int, default=None)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--nheads", type=int, default=8)
    parser.add_argument("--dim_feedforward", type=int, default=2048)
    parser.add_argument("--max_text_len", type=int, default=256)
    parser.add_argument(
        "--text_padding_length",
        type=int,
        default=21,
        help="Fixed token length. LIBERO uses 21 to match the released training features.",
    )
    parser.add_argument(
        "--text_layout_path",
        type=str,
        default="experiments/libero/configs/online_text_layout.json",
        help="Online tokenizer layout metadata for exact released-checkpoint compatibility.",
    )
    parser.add_argument("--vla_feature_enhancer_layers", type=int, default=6)
    parser.add_argument("--enhancer_inner_dim", type=int, default=1024)
    parser.add_argument("--action_dim", type=int, default=7)
    parser.add_argument("--chunk_size", type=int, default=12)
    parser.add_argument("--state_dim", type=int, default=8)
    parser.add_argument("--num_state_tokens", type=int, default=2)
    parser.add_argument("--text_dropout", type=float, default=0.0)
    parser.add_argument("--fusion_dropout", type=float, default=0.0)
    parser.add_argument("--fusion_droppath", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    parser.set_defaults(allow_hf_download=False)
    parser.add_argument("--allow_hf_download", dest="allow_hf_download", action="store_true")
    parser.add_argument("--no_allow_hf_download", dest="allow_hf_download", action="store_false")

    parser.set_defaults(shuffle_steps_within_episode=True)
    parser.add_argument("--shuffle_steps_within_episode", dest="shuffle_steps_within_episode", action="store_true")
    parser.add_argument("--no_shuffle_steps_within_episode", dest="shuffle_steps_within_episode", action="store_false")

    parser.set_defaults(freeze_backbones=False)
    parser.add_argument("--freeze_backbones", dest="freeze_backbones", action="store_true")
    parser.add_argument("--no_freeze_backbones", dest="freeze_backbones", action="store_false")

    parser.set_defaults(freeze_text_encoder=True)
    parser.add_argument("--freeze_text_encoder", dest="freeze_text_encoder", action="store_true")
    parser.add_argument("--train_text_encoder", dest="freeze_text_encoder", action="store_false")

    parser.set_defaults(require_feature_enhancer_preload=True)
    parser.add_argument(
        "--require_feature_enhancer_preload",
        dest="require_feature_enhancer_preload",
        action="store_true",
    )
    parser.add_argument(
        "--no_require_feature_enhancer_preload",
        dest="require_feature_enhancer_preload",
        action="store_false",
    )

    parser.set_defaults(load_text_projection_from_init=True)
    parser.add_argument(
        "--load_text_projection_from_init",
        dest="load_text_projection_from_init",
        action="store_true",
    )
    parser.add_argument(
        "--no_load_text_projection_from_init",
        dest="load_text_projection_from_init",
        action="store_false",
    )

    parser.set_defaults(require_text_proj_preload=True)
    parser.add_argument("--require_text_proj_preload", dest="require_text_proj_preload", action="store_true")
    parser.add_argument("--no_require_text_proj_preload", dest="require_text_proj_preload", action="store_false")

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        use_cuda = torch.cuda.is_available()
        backend = "nccl" if use_cuda else "gloo"
        dist.init_process_group(backend=backend)
        if use_cuda:
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
        is_distributed = True
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_distributed = False
    return is_distributed, rank, world_size, local_rank, device


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def build_model_architecture(args):
    text_layout = {}
    if args.text_layout_path:
        with open(args.text_layout_path, "r", encoding="utf-8") as handle:
            text_layout = json.load(handle)
        configured_length = int(text_layout["output_padding_length"])
        if configured_length != args.text_padding_length:
            raise ValueError(
                f"text layout output length {configured_length} does not match "
                f"--text_padding_length={args.text_padding_length}"
            )
    model_args = DummyArgs()
    model_args.use_eovsam = args.use_eovsam
    model_args.eovsam_repo = args.eovsam_repo
    model_args.eovsam_weights = args.eovsam_weights
    model_args.eovsam_images_per_batch = args.eovsam_images_per_batch
    model_args.eovsam_train_input_size = args.eovsam_train_input_size
    model_args.dinov3_path = args.dinov3_path
    model_args.bert_path = args.bert_path
    model_args.hidden_dim = args.hidden_dim
    model_args.nheads = args.nheads
    model_args.dim_feedforward = args.dim_feedforward
    model_args.max_text_len = args.max_text_len
    model_args.text_padding_length = args.text_padding_length
    model_args.text_padding_length_by_instruction = text_layout.get("padding_length_by_instruction", {})
    model_args.vla_feature_enhancer_layers = args.vla_feature_enhancer_layers
    model_args.enhancer_inner_dim = args.enhancer_inner_dim
    model_args.text_dropout = args.text_dropout
    model_args.fusion_dropout = args.fusion_dropout
    model_args.fusion_droppath = args.fusion_droppath
    model_args.action_dim = args.action_dim
    model_args.chunk_size = args.chunk_size
    model_args.state_dim = args.state_dim
    model_args.num_state_tokens = args.num_state_tokens
    model_args.local_files_only = not args.allow_hf_download
    model_args.freeze_vision_encoder = args.freeze_backbones
    model_args.freeze_text_encoder = args.freeze_text_encoder
    model_args.dinov3_precision = getattr(args, "dinov3_precision", "bf16_autocast")
    model_args.num_views = 2
    model_args.image_size = args.expected_image_size
    model_args.position_embedding = "view"
    model_args.encode_views_separately = True
    model_args.padding_strategy = "key_padding_mask"
    return build_turbovla(model_args)


def freeze_backbones(model):
    for name, param in model.named_parameters():
        if name.startswith("vision_encoder.backbone"):
            param.requires_grad = False


def get_latest_checkpoint(ckpt_dir, prefix):
    ckpts = glob.glob(os.path.join(ckpt_dir, f"{prefix}_*.pth"))
    if not ckpts:
        return None

    def extract_step(path):
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            return int(name.split("_")[-1])
        except ValueError:
            return -1

    return max(ckpts, key=extract_step)


def masked_l1_loss(pred, target, mask):
    l1 = torch.abs(pred - target)
    mask = mask.unsqueeze(-1).float()
    l1 = l1 * mask
    denom = (mask.sum() * pred.shape[-1]).clamp_min(1.0)
    return l1.sum() / denom


def build_scheduler(optimizer, max_steps, warmup_steps, min_lr_ratio=0.1):
    def lr_lambda(step):
        if step < warmup_steps:
            warmup_scale = float(step + 1) / float(max(1, warmup_steps))
            return 0.1 + 0.9 * warmup_scale

        progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(progress * math.pi))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _use_weight_decay(name, param):
    if param.ndim <= 1:
        return False
    lowered = name.lower()
    if lowered.endswith(".bias"):
        return False
    if "norm" in lowered or "layernorm" in lowered:
        return False
    return True


def build_param_group_optimizer(model, args):
    head_lr = args.lr if args.head_lr is None else args.head_lr
    head_wd = args.weight_decay if args.head_weight_decay is None else args.head_weight_decay
    grouped = {
        ("dinov3_decay", args.dinov3_lr, args.dinov3_weight_decay): [],
        ("dinov3_no_decay", args.dinov3_lr, 0.0): [],
        ("head_decay", head_lr, head_wd): [],
        ("head_no_decay", head_lr, 0.0): [],
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_dino = name.startswith("vision_encoder.backbone")
        decay = _use_weight_decay(name, param)
        if is_dino and decay:
            key = ("dinov3_decay", args.dinov3_lr, args.dinov3_weight_decay)
        elif is_dino:
            key = ("dinov3_no_decay", args.dinov3_lr, 0.0)
        elif decay:
            key = ("head_decay", head_lr, head_wd)
        else:
            key = ("head_no_decay", head_lr, 0.0)
        grouped[key].append(param)

    param_groups = []
    summary = []
    for (group_name, lr, weight_decay), params in grouped.items():
        if not params:
            continue
        count = sum(p.numel() for p in params)
        param_groups.append({"params": params, "lr": lr, "weight_decay": weight_decay, "name": group_name})
        summary.append({"name": group_name, "lr": lr, "weight_decay": weight_decay, "params": count})
    return AdamW(param_groups), summary


def reduce_mean(value, device, is_distributed, world_size):
    tensor = torch.tensor(value, device=device, dtype=torch.float32)
    if is_distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= world_size
    return tensor.item()


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def _libero_eval_suites(args):
    suites = [suite.strip() for suite in args.eval_suites.split(",") if suite.strip()]
    valid_suites = {"libero_spatial", "libero_object", "libero_goal", "libero_10"}
    invalid_suites = [suite for suite in suites if suite not in valid_suites]
    if not suites:
        raise ValueError("--eval_suites must contain at least one suite")
    if invalid_suites:
        raise ValueError(f"unsupported --eval_suites values: {invalid_suites}")
    return suites


def _libero_task_count(args):
    # Each of the four official LIBERO suites contains exactly ten tasks.
    return 10 if args.eval_max_tasks <= 0 else min(args.eval_max_tasks, 10)


def _libero_rank_gpu(args, rank, world_size):
    visible = args.eval_cuda_visible_devices or os.environ.get("CUDA_VISIBLE_DEVICES", "")
    gpu_ids = [item.strip() for item in visible.split(",") if item.strip()]
    if not gpu_ids:
        gpu_ids = [str(index) for index in range(world_size)]
    if len(gpu_ids) != world_size or len(set(gpu_ids)) != world_size:
        raise ValueError(
            f"Distributed evaluation needs one unique visible GPU per rank; got {gpu_ids}"
        )
    return gpu_ids[rank]


def _libero_eval_command(args, checkpoint_path):
    command = [
        sys.executable,
        "experiments/libero/evaluate.py",
        "--ckpt_path", checkpoint_path,
        "--libero_root", args.eval_libero_root,
        "--dinov3_path", args.dinov3_path,
        "--bert_path", args.bert_path,
        "--stats_path", args.stats_path,
        "--stats_key", args.stats_key,
        "--num_trials_per_task", str(args.eval_num_trials_per_task),
        "--max_tasks", str(args.eval_max_tasks),
        "--num_open_loop_steps", str(args.chunk_size),
        "--chunk_size", str(args.chunk_size),
        "--seed", str(args.eval_seed),
        "--precision", args.eval_precision,
        "--normalize_binary_gripper", args.normalize_binary_gripper,
        "--mujoco_gl", args.eval_mujoco_gl,
        "--pyopengl_platform", args.eval_pyopengl_platform,
        "--hidden_dim", str(args.hidden_dim),
        "--nheads", str(args.nheads),
        "--dim_feedforward", str(args.dim_feedforward),
        "--max_text_len", str(args.max_text_len),
        "--text_padding_length", str(args.text_padding_length),
        "--vla_feature_enhancer_layers", str(args.vla_feature_enhancer_layers),
        "--enhancer_inner_dim", str(args.enhancer_inner_dim),
        "--action_dim", str(args.action_dim),
        "--state_dim", str(args.state_dim),
        "--num_state_tokens", str(args.num_state_tokens),
        "--text_dropout", str(args.text_dropout),
        "--fusion_dropout", str(args.fusion_dropout),
        "--fusion_droppath", str(args.fusion_droppath),
    ]
    if args.allow_hf_download:
        command.append("--allow_hf_download")
    return command


def _libero_rank_plan(args, step_dir, rank, world_size):
    plan = []
    for suite_index, suite in enumerate(_libero_eval_suites(args)):
        task_ids = [
            task_id for task_id in range(_libero_task_count(args))
            if (suite_index * 10 + task_id) % world_size == rank
        ]
        if task_ids:
            plan.append({
                "suite": suite,
                "task_ids": task_ids,
                "result_json_path": os.path.join(step_dir, f"{suite}.rank{rank}.json"),
                "log_path": os.path.join(step_dir, f"{suite}.rank{rank}.log"),
            })
    return plan


def run_libero_evaluation(args, checkpoint_path, global_step, rank=0, world_size=1):
    suites = _libero_eval_suites(args)
    distributed_eval = args.eval_distributed and world_size > 1

    step_dir = os.path.join(args.eval_output_dir, f"step_{global_step}")
    os.makedirs(step_dir, exist_ok=True)
    env = os.environ.copy()
    env["MUJOCO_GL"] = args.eval_mujoco_gl
    env["PYOPENGL_PLATFORM"] = args.eval_pyopengl_platform
    if distributed_eval:
        env["CUDA_VISIBLE_DEVICES"] = _libero_rank_gpu(args, rank, world_size)
    elif args.eval_cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.eval_cuda_visible_devices

    if distributed_eval:
        plan = _libero_rank_plan(args, step_dir, rank, world_size)
        if not plan:
            return None
        plan_path = os.path.join(step_dir, f"eval_plan.rank{rank}.json")
        with open(plan_path, "w", encoding="utf-8") as handle:
            json.dump(plan, handle, indent=2)
        command = _libero_eval_command(args, checkpoint_path) + [
            "--suite_plan_path", plan_path,
            "--model_load_lock_path", os.path.join(step_dir, "model_load.lock"),
            "--model_load_min_available_gb", str(getattr(args, "eval_min_available_ram_gb", 32.0)),
        ]
        print(f"rank {rank} evaluating {len(plan)} suites at step {global_step}: {' '.join(command)}", flush=True)
        subprocess.run(command, check=True, env=env)
        return None

    metrics = {"train/global_step": global_step}
    total_episodes = 0
    total_successes = 0
    suite_rates = []
    for suite_index, suite in enumerate(suites):
        task_ids = None
        if distributed_eval:
            task_ids = [
                task_id for task_id in range(_libero_task_count(args))
                if (suite_index * 10 + task_id) % world_size == rank
            ]
            if not task_ids:
                continue
        suffix = f".rank{rank}" if distributed_eval else ""
        result_path = os.path.join(step_dir, f"{suite}{suffix}.json")
        log_path = os.path.join(step_dir, f"{suite}{suffix}.log")
        command = _libero_eval_command(args, checkpoint_path) + [
            "--task_suite_name", suite,
            "--result_json_path", result_path,
            "--log_path", log_path,
        ]
        if task_ids is not None:
            command.extend(["--task_ids", ",".join(map(str, task_ids))])
        print(f"rank {rank} evaluating {suite} at step {global_step}: {' '.join(command)}", flush=True)
        subprocess.run(command, check=True, env=env)

        if distributed_eval:
            continue

        with open(result_path, "r", encoding="utf-8") as handle:
            result = json.load(handle)
        suite_episodes = int(result["total_episodes"])
        suite_successes = int(result["total_successes"])
        suite_rate = float(result["overall_success_rate"])
        total_episodes += suite_episodes
        total_successes += suite_successes
        suite_rates.append(suite_rate)
        metrics[f"libero_success_rate/{suite}"] = suite_rate
        metrics[f"eval/{suite}/successes"] = suite_successes
        metrics[f"eval/{suite}/episodes"] = suite_episodes
        for task in result.get("tasks", []):
            task_id = int(task["task_id"])
            metrics[f"eval/{suite}/task_{task_id:02d}_success_rate"] = float(task["success_rate"])

    if distributed_eval:
        return None
    metrics["libero_success_rate/overall"] = total_successes / max(total_episodes, 1)
    metrics["eval/total_successes"] = total_successes
    metrics["eval/total_episodes"] = total_episodes
    return metrics


def aggregate_distributed_libero_evaluation(args, global_step, world_size):
    """Merge rank-local results and verify that every requested task ran once."""
    step_dir = os.path.join(args.eval_output_dir, f"step_{global_step}")
    metrics = {"train/global_step": global_step}
    total_successes = total_episodes = 0
    for suite_index, suite in enumerate(_libero_eval_suites(args)):
        task_results = {}
        template = None
        for rank in range(world_size):
            expected_ids = [
                task_id for task_id in range(_libero_task_count(args))
                if (suite_index * 10 + task_id) % world_size == rank
            ]
            if not expected_ids:
                continue
            path = os.path.join(step_dir, f"{suite}.rank{rank}.json")
            with open(path, "r", encoding="utf-8") as handle:
                shard = json.load(handle)
            if shard.get("requested_task_ids") != expected_ids:
                raise RuntimeError(f"{path} has unexpected task IDs")
            template = template or shard
            for task in shard["tasks"]:
                task_id = int(task["task_id"])
                if task_id in task_results:
                    raise RuntimeError(f"{suite} task {task_id} was evaluated twice")
                if int(task["episodes"]) != args.eval_num_trials_per_task:
                    raise RuntimeError(f"{suite} task {task_id} has incomplete rollouts")
                task_results[task_id] = task
        expected_all = set(range(_libero_task_count(args)))
        if set(task_results) != expected_all:
            raise RuntimeError(f"{suite} task coverage is {sorted(task_results)}, expected {sorted(expected_all)}")
        suite_tasks = [task_results[task_id] for task_id in sorted(task_results)]
        suite_successes = sum(int(task["successes"]) for task in suite_tasks)
        suite_episodes = sum(int(task["episodes"]) for task in suite_tasks)
        suite_rate = suite_successes / suite_episodes
        combined = dict(template)
        combined.update({
            "requested_task_ids": sorted(task_results),
            "tasks": suite_tasks,
            "total_successes": suite_successes,
            "total_episodes": suite_episodes,
            "overall_success_rate": suite_rate,
        })
        with open(os.path.join(step_dir, f"{suite}.json"), "w", encoding="utf-8") as handle:
            json.dump(combined, handle, indent=2, ensure_ascii=False)
        metrics[f"libero_success_rate/{suite}"] = suite_rate
        metrics[f"eval/{suite}/successes"] = suite_successes
        metrics[f"eval/{suite}/episodes"] = suite_episodes
        for task in suite_tasks:
            metrics[f"eval/{suite}/task_{int(task['task_id']):02d}_success_rate"] = float(task["success_rate"])
        total_successes += suite_successes
        total_episodes += suite_episodes
    metrics["libero_success_rate/overall"] = total_successes / total_episodes
    metrics["eval/total_successes"] = total_successes
    metrics["eval/total_episodes"] = total_episodes
    return metrics


def prune_training_checkpoints(args):
    if args.keep_last_checkpoints <= 0:
        return
    pattern = re.compile(re.escape(args.checkpoint_prefix) + r"_(\d+)\.pth$")
    entries = []
    for name in os.listdir(args.checkpoint_dir):
        match = pattern.fullmatch(name)
        if match:
            entries.append((int(match.group(1)), name))
    entries.sort(reverse=True)
    for step, name in entries[args.keep_last_checkpoints:]:
        # Evaluation weights must survive pruning, including legacy runs that
        # were launched with keep_eval_checkpoints=False.
        if args.eval_steps > 0 and (step % args.eval_steps == 0 or step == args.max_steps):
            continue
        path = os.path.join(args.checkpoint_dir, name)
        os.remove(path)
        print(f"pruned old training checkpoint: {path}", flush=True)


def _extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        for key in ["model_state_dict", "model", "state_dict"]:
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                return ckpt_obj[key]
    if isinstance(ckpt_obj, dict):
        return ckpt_obj
    raise ValueError("unsupported checkpoint format")


def load_interaction_from_init_checkpoint(model, ckpt_path):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    source_state = _extract_state_dict(ckpt)
    source_state = {(k[7:] if k.startswith("module.") else k): v for k, v in source_state.items()}

    target_state = model.state_dict()
    mapped = {}
    skipped_missing = 0
    skipped_shape = 0
    missing_keys = []
    shape_mismatch_keys = []

    prefix_pairs = [
        ("transformer.encoder.fusion_layers.", "vision_language_interaction.fusion_layers."),
        ("transformer.encoder.text_layers.", "vision_language_interaction.text_layers."),
    ]

    for src_key, tensor in source_state.items():
        tgt_key = None
        for src_prefix, tgt_prefix in prefix_pairs:
            if src_key.startswith(src_prefix):
                tgt_key = tgt_prefix + src_key[len(src_prefix):]
                break
        if tgt_key is None:
            continue
        if tgt_key not in target_state:
            skipped_missing += 1
            missing_keys.append((src_key, tgt_key))
            continue
        if target_state[tgt_key].shape != tensor.shape:
            skipped_shape += 1
            shape_mismatch_keys.append((src_key, tgt_key, tuple(tensor.shape), tuple(target_state[tgt_key].shape)))
            continue
        mapped[tgt_key] = tensor

    if not mapped:
        raise RuntimeError("no feature-enhancer parameters were mapped from checkpoint")

    target_state.update(mapped)
    missing, unexpected = model.load_state_dict(target_state, strict=False)
    return {
        "mapped": len(mapped),
        "skipped_missing": skipped_missing,
        "skipped_shape": skipped_shape,
        "load_missing_after_update": len(missing),
        "load_unexpected_after_update": len(unexpected),
        "missing_keys": missing_keys,
        "shape_mismatch_keys": shape_mismatch_keys,
    }


def load_text_projection_from_init_checkpoint(model, ckpt_path):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    source_state = _extract_state_dict(ckpt)
    source_state = {(k[7:] if k.startswith("module.") else k): v for k, v in source_state.items()}

    target_state = model.state_dict()
    mapped = {}
    skipped_missing = 0
    skipped_shape = 0
    missing_keys = []
    shape_mismatch_keys = []

    for src_key, tensor in source_state.items():
        tgt_key = None
        if src_key.startswith("feat_map."):
            tgt_key = "text_encoder.text_projection." + src_key[len("feat_map."):]
        else:
            continue

        if tgt_key not in target_state:
            skipped_missing += 1
            missing_keys.append((src_key, tgt_key))
            continue
        if target_state[tgt_key].shape != tensor.shape:
            skipped_shape += 1
            shape_mismatch_keys.append((src_key, tgt_key, tuple(tensor.shape), tuple(target_state[tgt_key].shape)))
            continue
        mapped[tgt_key] = tensor

    if not mapped:
        raise RuntimeError("no text projection parameters were mapped from checkpoint")

    target_state.update(mapped)
    missing, unexpected = model.load_state_dict(target_state, strict=False)
    return {
        "mapped": len(mapped),
        "skipped_missing": skipped_missing,
        "skipped_shape": skipped_shape,
        "load_missing_after_update": len(missing),
        "load_unexpected_after_update": len(unexpected),
        "missing_keys": missing_keys,
        "shape_mismatch_keys": shape_mismatch_keys,
    }


def move_samples_to_device(samples, device):
    if isinstance(samples, dict):
        return {k: v.to(device, non_blocking=True) for k, v in samples.items()}
    return samples.to(device, non_blocking=True)


def train_model():
    args = parse_args()
    if args.eval_steps < 0:
        raise ValueError("--eval_steps must be >= 0")
    if args.eval_num_trials_per_task <= 0:
        raise ValueError("--eval_num_trials_per_task must be > 0")
    if args.eval_sync_timeout_minutes <= 0:
        raise ValueError("--eval_sync_timeout_minutes must be > 0")
    if args.head_lr is None:
        args.head_lr = args.lr
    if args.head_weight_decay is None:
        args.head_weight_decay = args.weight_decay
    if args.precision == "bf16_amp":
        if not torch.cuda.is_available():
            raise ValueError("bf16_amp precision requires CUDA")
        if not torch.cuda.is_bf16_supported():
            raise ValueError("bf16_amp requested but current CUDA device does not report bf16 support")
    torch.set_float32_matmul_precision("high")

    if args.lr_schedule_steps is None:
        args.lr_schedule_steps = args.max_steps
    if args.lr_schedule_steps <= args.warmup_steps:
        raise ValueError(
            f"lr_schedule_steps={args.lr_schedule_steps} must be greater than warmup_steps={args.warmup_steps}"
        )
    set_seed(args.seed)

    is_distributed, rank, world_size, local_rank, device = setup_distributed()
    configure_timestamped_output_dirs(args, is_distributed, rank)
    eval_sync_group = None
    if is_distributed and args.eval_steps > 0:
        # A rollout can exceed NCCL's default 10-minute watchdog. Keep NCCL
        # free of pending work and wait on a CPU/Gloo group sized for evals.
        eval_sync_group = dist.new_group(
            backend="gloo",
            timeout=timedelta(minutes=args.eval_sync_timeout_minutes),
        )
    wandb_run = None

    try:
        model = build_model_architecture(args)

        need_any_init_preload = (
            args.require_feature_enhancer_preload
            or args.require_text_proj_preload
            or args.load_text_projection_from_init
        )
        if args.pretrained_init_ckpt is None and need_any_init_preload:
            raise ValueError(
                "`--pretrained_init_ckpt` is required for requested preload options "
                "(feature-enhancer and/or text_proj)"
            )

        if args.pretrained_init_ckpt is not None:
            if args.require_feature_enhancer_preload:
                fe_report = load_interaction_from_init_checkpoint(model, args.pretrained_init_ckpt)
                if rank == 0:
                    print("vision-language interaction preload from init checkpoint:")
                    print(f"  ckpt: {args.pretrained_init_ckpt}")
                    print(f"  mapped={fe_report['mapped']}")
                    print(f"  skipped_missing={fe_report['skipped_missing']}")
                    print(f"  skipped_shape={fe_report['skipped_shape']}")
                    print("missing_keys:", fe_report["missing_keys"])
                    print("shape_mismatch_keys:", fe_report["shape_mismatch_keys"])

            if args.load_text_projection_from_init:
                text_proj_report = load_text_projection_from_init_checkpoint(model, args.pretrained_init_ckpt)
                if rank == 0:
                    print("text projection preload from init checkpoint:")
                    print(f"  ckpt: {args.pretrained_init_ckpt}")
                    print(f"  mapped={text_proj_report['mapped']}")
                    print(f"  skipped_missing={text_proj_report['skipped_missing']}")
                    print(f"  skipped_shape={text_proj_report['skipped_shape']}")
                    print("missing_keys:", text_proj_report["missing_keys"])
                    print("shape_mismatch_keys:", text_proj_report["shape_mismatch_keys"])
            elif args.require_text_proj_preload:
                raise ValueError(
                    "text projection preload is required, but `--no_load_text_projection_from_init` was set"
                )

        if args.freeze_backbones:
            freeze_backbones(model)

        if rank == 0:
            trainable = [n for n, p in model.named_parameters() if p.requires_grad]
            frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
            print(f"device={device}, distributed={is_distributed}, world_size={world_size}")
            print(f"dinov3_path={args.dinov3_path}")
            print(f"bert_path={args.bert_path}")
            print(
                f"online_bert=True, freeze_text_encoder={args.freeze_text_encoder}, "
                f"text_padding_length={args.text_padding_length}, text_layout_path={args.text_layout_path}"
            )
            print(f"pretrained_init_ckpt={args.pretrained_init_ckpt}")
            print(f"run_timestamp={args.run_timestamp}")
            print(f"checkpoint_dir={args.checkpoint_dir}")
            print(f"eval_output_dir={args.eval_output_dir}")
            print(f"load_text_projection_from_init={args.load_text_projection_from_init}")
            print(f"max_steps={args.max_steps}, lr_schedule_steps={args.lr_schedule_steps}")
            print(
                f"precision={args.precision}, head_lr={args.head_lr}, dinov3_lr={args.dinov3_lr}, "
                f"head_wd={args.head_weight_decay}, dinov3_wd={args.dinov3_weight_decay}"
            )
            print(f"warmup_steps={args.warmup_steps}, min_lr_ratio={args.min_lr_ratio}")
            print(
                f"shuffle: episode_buffer={args.shuffle_buffer}, "
                f"within_episode={args.shuffle_steps_within_episode}"
            )
            print(f"trainable params: {len(trainable)}")
            print(f"frozen params: {len(frozen)}")
            print("first 30 trainable:", trainable[:30])

        if rank == 0:
            os.makedirs(args.checkpoint_dir, exist_ok=True)
        if is_distributed:
            dist.barrier()

        global_step = 0
        ckpt = None
        latest_ckpt_path = get_latest_checkpoint(args.checkpoint_dir, args.checkpoint_prefix)
        if latest_ckpt_path is not None and args.resume_mode != "none":
            if rank == 0:
                print(f"resume from {latest_ckpt_path} with mode={args.resume_mode}")
            ckpt = torch.load(latest_ckpt_path, map_location="cpu")
            missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
            if rank == 0:
                print("resume missing keys:", len(missing))
                print("resume unexpected keys:", len(unexpected))
            if args.resume_mode == "all":
                global_step = int(ckpt.get("global_step", 0))
            else:
                global_step = 0
                if rank == 0:
                    print("optimizer/scheduler reset due to resume_mode=model")
        elif rank == 0:
            print("no checkpoint resumed, train from current initialization")

        model.to(device)
        if is_distributed:
            if device.type == "cuda":
                model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
            else:
                model = DDP(model, find_unused_parameters=True)

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer, optimizer_summary = build_param_group_optimizer(unwrap_model(model), args)
        if rank == 0:
            print("optimizer param groups:", optimizer_summary)
        scheduler = build_scheduler(
            optimizer,
            max_steps=args.lr_schedule_steps,
            warmup_steps=args.warmup_steps,
            min_lr_ratio=args.min_lr_ratio,
        )

        if ckpt is not None and args.resume_mode == "all":
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            if hasattr(optimizer, "_ema_params"):
                ema_state = ckpt.get("ema_model_state_dict")
                if ema_state is None:
                    raise RuntimeError("Cannot resume EMA optimizer without ema_model_state_dict")
                with torch.no_grad():
                    for param, name in optimizer._ema_param_names.items():
                        optimizer._ema_params[param].copy_(ema_state[name])
                if rank == 0:
                    print(f"restored EMA tensors: {len(optimizer._ema_param_names)}", flush=True)

        model_to_load = unwrap_model(model)

        dataset = LiberoRLDSDataset(
            dataset_dir=args.dataset_dir,
            LOCAL_DINOV3_PATH=args.dinov3_path,
            rank=rank,
            world_size=world_size,
            chunk_size=args.chunk_size,
            split=args.dataset_split,
            shuffle_buffer=args.shuffle_buffer,
            shuffle_steps_within_episode=args.shuffle_steps_within_episode,
            step_mix_buffer_size=args.step_mix_buffer_size,
            seed=args.seed,
            local_files_only=not args.allow_hf_download,
            expected_image_size=args.expected_image_size,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=None,
            collate_fn=vla_collate_fn,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(args.num_workers > 0),
        )
        data_iter = iter(dataloader)

        model.train()
        if args.freeze_backbones:
            model_to_load.dinov3.eval()

        if rank == 0:
            per_gpu_effective_bs = args.batch_size * args.grad_accum_steps
            global_effective_bs = per_gpu_effective_bs * world_size
            print(f"batch_size(per_gpu)={args.batch_size}, grad_accum_steps={args.grad_accum_steps}")
            print(f"effective_batch_size(per_gpu)={per_gpu_effective_bs}")
            print(f"effective_batch_size(global)={global_effective_bs}")

            if args.wandb_project and args.wandb_mode != "disabled":
                import wandb

                os.makedirs(args.wandb_dir, exist_ok=True)
                tags = [tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()]
                run_name = build_wandb_run_name(args.wandb_run_name, timestamp=args.run_timestamp)
                wandb_run = wandb.init(
                    project=args.wandb_project,
                    entity=args.wandb_entity or None,
                    name=run_name,
                    group=args.wandb_group or None,
                    tags=tags or None,
                    mode=args.wandb_mode,
                    dir=args.wandb_dir,
                    config={
                        **vars(args),
                        "world_size": world_size,
                        "effective_batch_size_per_gpu": per_gpu_effective_bs,
                        "effective_batch_size_global": global_effective_bs,
                    },
                )
                wandb_run.define_metric("train/global_step")
                wandb_run.define_metric("train/*", step_metric="train/global_step")
                wandb_run.define_metric("eval/*", step_metric="train/global_step")
                wandb_run.define_metric("libero_success_rate/*", step_metric="train/global_step")

            pbar = tqdm(total=args.max_steps, initial=global_step, desc="training")
        else:
            pbar = None

        loss_window = []
        while global_step < args.max_steps:
            optimizer.zero_grad(set_to_none=True)
            loss_accum = 0.0

            for _ in range(args.grad_accum_steps):
                try:
                    samples, instructions, states, gt_actions, action_chunk_masks = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    samples, instructions, states, gt_actions, action_chunk_masks = next(data_iter)

                samples = move_samples_to_device(samples, device)
                states = states.to(device, non_blocking=True)
                gt_actions = gt_actions.to(device, non_blocking=True)
                action_chunk_masks = action_chunk_masks.to(device, non_blocking=True)

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=args.precision == "bf16_amp",
                ):
                    pred_actions = model(instructions, samples, states)
                    if pred_actions.shape != gt_actions.shape:
                        raise ValueError(
                            f"pred_actions.shape={pred_actions.shape}, gt_actions.shape={gt_actions.shape} mismatch"
                        )

                    loss = masked_l1_loss(pred_actions, gt_actions, action_chunk_masks)
                (loss / args.grad_accum_steps).backward()
                loss_accum += loss.detach().item()

            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            local_loss = loss_accum / args.grad_accum_steps
            global_loss = reduce_mean(local_loss, device, is_distributed, world_size)

            loss_window.append(global_loss)
            if len(loss_window) > args.log_freq:
                loss_window.pop(0)

            global_step += 1

            if rank == 0:
                avg_window_loss = sum(loss_window) / len(loss_window)
                pbar.update(1)
                if wandb_run is not None:
                    metrics = {
                        "train/global_step": global_step,
                        "train/loss": global_loss,
                        "train/loss_avg": avg_window_loss,
                        "train/grad_norm": float(grad_norm.detach().item()),
                    }
                    for group in optimizer.param_groups:
                        metrics[f"train/lr/{group['name']}"] = group["lr"]
                    if device.type == "cuda":
                        metrics["system/gpu_memory_allocated_gb"] = torch.cuda.memory_allocated(device) / 2**30
                        metrics["system/gpu_memory_reserved_gb"] = torch.cuda.memory_reserved(device) / 2**30
                    wandb_run.log(metrics)
                if global_step % args.log_freq == 0:
                    pbar.set_postfix(
                        {
                            "loss": f"{global_loss:.5f}",
                            "avg": f"{avg_window_loss:.5f}",
                            "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                        }
                    )

            should_eval = args.eval_steps > 0 and (
                global_step % args.eval_steps == 0 or global_step == args.max_steps
            )
            should_save = should_eval or global_step % args.save_steps == 0 or (
                args.save_final and global_step == args.max_steps
            )
            save_path = os.path.join(args.checkpoint_dir, f"{args.checkpoint_prefix}_{global_step}.pth")
            if rank == 0 and should_save:
                torch.save(
                    {
                        "global_step": global_step,
                        "model_state_dict": unwrap_model(model).state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "loss": global_loss,
                        "args": vars(args),
                        "model_config": unwrap_model(model).config.to_dict(),
                    },
                    save_path,
                )
                print(f"saved: {save_path}")

            if should_eval:
                if is_distributed:
                    dist.barrier()
                encoder = getattr(unwrap_model(model), "eovsam_encoder", None)
                if args.eval_distributed and is_distributed:
                    # Every rank gives its GPU to one rollout worker. All ranks
                    # later reload EOVSAM together because RADIO's model loader
                    # performs a collective on the default NCCL process group.
                    if encoder is not None:
                        encoder.release()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    local_error = None
                    try:
                        run_libero_evaluation(args, save_path, global_step, rank, world_size)
                    except Exception as exc:
                        local_error = f"LIBERO evaluation failed on rank {rank} at step {global_step}: {exc}"
                    errors = [None] * world_size
                    dist.all_gather_object(errors, local_error, group=eval_sync_group)
                    if any(errors):
                        raise RuntimeError("; ".join(error for error in errors if error))
                    eval_error = None
                    if rank == 0:
                        try:
                            eval_metrics = aggregate_distributed_libero_evaluation(args, global_step, world_size)
                            print(f"evaluation metrics at step {global_step}: {eval_metrics}", flush=True)
                            if wandb_run is not None:
                                wandb_run.log(eval_metrics)
                            if should_save:
                                prune_training_checkpoints(args)
                        except Exception as exc:
                            eval_error = f"LIBERO evaluation aggregation failed at step {global_step}: {exc}"
                    error_payload = [eval_error]
                    dist.broadcast_object_list(error_payload, src=0, group=eval_sync_group)
                    if error_payload[0] is not None:
                        raise RuntimeError(error_payload[0])
                    if encoder is not None:
                        encoder._load()
                        with torch.no_grad():
                            encoder.model.get_SAM_text_classifier("openvocab_ade20k_panoptic_val")
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                    dist.barrier(group=eval_sync_group)
                else:
                    eval_error = None
                    if rank == 0:
                        try:
                            if encoder is not None:
                                encoder.release()
                            if device.type == "cuda":
                                torch.cuda.empty_cache()
                            eval_metrics = run_libero_evaluation(args, save_path, global_step)
                            print(f"evaluation metrics at step {global_step}: {eval_metrics}", flush=True)
                            if wandb_run is not None:
                                wandb_run.log(eval_metrics)
                            if encoder is not None:
                                encoder._load()
                                with torch.no_grad():
                                    encoder.model.get_SAM_text_classifier("openvocab_ade20k_panoptic_val")
                                if device.type == "cuda":
                                    torch.cuda.synchronize(device)
                            if should_save:
                                prune_training_checkpoints(args)
                        except Exception as exc:
                            eval_error = f"LIBERO evaluation failed at step {global_step}: {exc}"
                    if is_distributed:
                        error_payload = [eval_error]
                        dist.broadcast_object_list(error_payload, src=0, group=eval_sync_group)
                        eval_error = error_payload[0]
                    if eval_error is not None:
                        raise RuntimeError(eval_error)
            if rank == 0 and should_save and not should_eval:
                prune_training_checkpoints(args)

        if pbar is not None:
            pbar.close()
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        cleanup_distributed()


if __name__ == "__main__":
    train_model()
