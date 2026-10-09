"""Frozen, externally checkpointed EOVSAM with ADE20K classification support."""
from __future__ import annotations

import sys
import gc
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .configuration import EOVSAMConfig


class FrozenEOVSAMEncoder(nn.Module):
    def __init__(self, config: EOVSAMConfig) -> None:
        super().__init__()
        self.config = config
        self.register_buffer("device_anchor", torch.empty(0), persistent=False)
        # External frozen weights are loaded lazily and never duplicated in a
        # TurboVLA checkpoint or exposed to its optimizer/DDP parameter list.
        object.__setattr__(self, "_model", None)
        self.cfg = None
        self.metadata = None
        self.resize = None
        self.policy_training = True

    @property
    def model(self):
        return self._model

    def release(self):
        """Release external weights before a rollout subprocess shares this GPU.

        The next training forward reloads the unchanged external checkpoint.
        Registered policy parameters and DDP buckets remain untouched.
        """
        object.__setattr__(self, "_model", None)
        gc.collect()

    def train(self, mode: bool = True):
        self.policy_training = mode
        super().train(False)
        if self._model is not None:
            self._model.eval()
        return self

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        if self._model is not None:
            # Keep EOVSAM's native FP32 parameters/BF16 backbone autocast even
            # when the policy is converted wholesale to BF16 for evaluation.
            self._model.to(device=self.device_anchor.device)
            self._model.clear_inference_cache()
        return self

    @torch.no_grad()
    def _load(self):
        if self._model is not None:
            return
        repo = Path(self.config.repo_path).resolve()
        if not repo.is_dir() or not Path(self.config.weights_path).is_file():
            raise FileNotFoundError("EOVSAM repository or external checkpoint not found")
        sys.path.insert(0, str(repo))
        from detectron2.config import get_cfg
        from detectron2.projects.deeplab import add_deeplab_config
        from detectron2.data import MetadataCatalog, transforms as T
        from detectron2.checkpoint import DetectionCheckpointer
        import maft  # registers ADE20K and the original training metadata
        from sam3.config import add_sam3_config
        from sam3.EOVSAM import EOVSAM
        from sam3.mask_adapter_head import MASKAdapterHead  # registers optional head

        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_sam3_config(cfg)
        config_path = Path(self.config.config_path)
        cfg.merge_from_file(str(config_path if config_path.is_absolute() else repo / config_path))
        cfg.MODEL.SAM3.BPE_PATH = str(repo / "sam3/assets/bpe_simple_vocab_16e6.txt.gz")
        cfg.MODEL.WEIGHTS = self.config.weights_path
        cfg.MODEL.DEVICE = str(self.device_anchor.device)
        cfg.DATASETS.TEST = ("openvocab_ade20k_panoptic_val",)
        cfg.eval_only = True
        cfg.freeze()
        model = EOVSAM(cfg)
        DetectionCheckpointer(model).load(self.config.weights_path)
        model.eval().requires_grad_(False)
        model.to(device=self.device_anchor.device)
        object.__setattr__(self, "_model", model)
        self.cfg = cfg
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        self.resize = T.ResizeShortestEdge(cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST)

    def prepare_inputs(self, rgb: torch.Tensor, *, use_training_resolution: bool = False):
        """Follow EOVSAM's DefaultPredictor test resize; preserve raw H/W."""
        self._load()
        if rgb.ndim != 4 or rgb.shape[1] != 3 or rgb.dtype != torch.uint8:
            raise ValueError("EOVSAM expects uint8 RGB [N,3,H,W], with values in [0,255]")
        inputs = []
        for i, image in enumerate(rgb):
            array = image.detach().cpu().permute(1, 2, 0).numpy()
            h, w = array.shape[:2]
            if use_training_resolution and self.config.train_input_size is not None:
                from detectron2.data.transforms import ResizeTransform
                size = self.config.train_input_size
                resized = ResizeTransform(h, w, size, size).apply_image(array)
            else:
                resized = self.resize.get_transform(array).apply_image(array)
            inputs.append({
                "image": torch.as_tensor(np.ascontiguousarray(resized.transpose(2, 0, 1)), dtype=torch.float32),
                "height": h, "width": w, "file_name": f"libero_{i}.png",
                "meta": {"dataname": "openvocab_ade20k_panoptic_val"},
            })
        return inputs

    @torch.no_grad()
    def extract(self, rgb: torch.Tensor, *, segment: bool = False):
        self._load()
        self._model.eval()
        features, results = [], []
        for start in range(0, rgb.shape[0], self.config.images_per_batch):
            inputs = self.prepare_inputs(
                rgb[start:start + self.config.images_per_batch],
                use_training_resolution=self.policy_training and not segment,
            )
            # EOVSAM has its own backbone autocast. Prevent policy AMP from
            # changing the rest of its pretrained inference computation.
            with torch.autocast(self.device_anchor.device.type, enabled=False):
                if segment:
                    predictions, tokens = self._model(inputs, return_object_features=True)
                    results.extend(predictions)
                else:
                    tokens = self._model(inputs, features_only=True)
            if tokens.shape[-1] != self.config.feature_dim:
                raise RuntimeError(f"Expected {self.config.feature_dim} EOVSAM features, got {tokens.shape[-1]}")
            features.append(tokens.detach())
        return torch.cat(features), results

    @torch.no_grad()
    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        if rgb.ndim != 5 or rgb.shape[2] != 3:
            raise ValueError("EOVSAM requires raw RGB [B,V,3,H,W]")
        b, v = rgb.shape[:2]
        features, _ = self.extract(rgb.flatten(0, 1))
        # Each camera's query indices remain independent and ordered.
        return features.reshape(b, v * features.shape[1], features.shape[2])
