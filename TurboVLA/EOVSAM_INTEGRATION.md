# TurboVLA + frozen EOVSAM

Project: `/data/hmp/Robotics/TurboVLA_EOVSAM`
Environment: `conda activate turbovla-sam-libero`

## Data and model path

The LIBERO loader preserves uint8 RGB for both agentview and wrist alongside
the original DINOv3 tensors. DINOv3 preprocessing is unchanged. EOVSAM uses
`configs/eval_semantic.yaml` and Detectron2 `ResizeShortestEdge` with the native
`MIN_SIZE_TEST=1152`, `MAX_SIZE_TEST=1152` settings. Original image dimensions
are retained for segmentation restoration. RGB is normalized by EOVSAM itself.

For each view, take `hs[-1]` (200 queries, 256 channels) and pool the RADIO
SigLIP2-G features (1536 channels) with that same decoder layer's attention.
The trained attention-head weights and pooling temperature are reused. The
four-dimensional final-layer attention input selects the last head-weight row.
No query sorting, filtering or class-dependent selection is applied. The
feature is taken before query projection, pixel-feature addition, or classifier
normalization. Two views concatenate in agentview-then-wrist order, giving
`[B,400,1792]`. The action head uses two independent linear projectors (one
per camera, 1792 -> 256), projects to 256 channels and adds a learned
camera embedding; final object memory is `[B,400,256]`.

Every action decoder layer follows:

1. Original self-attention, normalization, dropout and residual.
2. Original cross-attention to vision-language + state memory.
3. New pre-normalized object cross-attention with dropout and residual.
4. Original feed-forward block.

The original layer parameter names, layer cloning, activation, masks, causal
handling, and optional final norm are retained. No additional norm is applied
to the original blocks. An absent/zero object-attention branch is numerically
identical to the original decoder.

## Freeze and checkpoints

EOVSAM is permanently in eval mode, all of its parameters have
`requires_grad=False`, and extraction runs under `torch.no_grad()`. Its native
FP32/BF16-backbone precision is isolated from policy autocast. During action
training the feature-only path skips mask heads and classification postprocessing.
ADE20K (150 classes with the upstream synonyms/templates) classifier logic is
retained for complete segmentation checks.

EOVSAM is an external, lazy-loaded frozen model: its large weights are not
included in the optimizer, DDP parameter list or TurboVLA checkpoint. The model
configuration stores the repo/checkpoint paths. The default checkpoint remains
`/data/hmp/SAMweights/EOVSAM_release_3888_1675_6088_2052_9706.pth`.

The release configuration disables VPT. The loader now respects this flag
instead of injecting random prompts absent from the checkpoint. The resulting
missing tuner prompt tensors are empty and have no effect on inference. Other
resource paths were made relative to their source files so launches from the
TurboVLA directory work.

## Run

Training enables EOVSAM by default. Existing training arguments still apply;
`--no-use_eovsam` restores the original model. Additional options are
`--eovsam_repo`, `--eovsam_weights`, and `--eovsam_images_per_batch` (default 1).
Both views are always processed; the last setting only controls image microbatching.

From the TurboVLA directory:

```bash
USE_TF=0 USE_FLAX=0 HF_HUB_OFFLINE=1 python scripts/test_decoder.py
USE_TF=0 python scripts/test_object_projection.py
USE_TF=0 USE_FLAX=0 HF_HUB_OFFLINE=1 python scripts/verify_eovsam_pipeline.py
```

Visualization and verification output is stored in
`../outputs/eovsam_verification`. The script reads actual LIBERO RLDS samples,
uses the new RGB pipeline, runs full ADE20K segmentation, compares full and
feature-only representations, then runs a complete TurboVLA forward/backward.
It hashes the EOVSAM checkpoint before and after.

## Verified on 2026-09-26

- Three LIBERO suites, two views each: native input 1152x1152, restored output 256x256.
- Features per image: `[200,1792]`; policy object tokens: `[1,400,1792]`.
- Full segmentation vs feature-only path: maximum feature error 0.
- Policy action output: `[1,12,7]`, finite forward/backward.
- EOVSAM stayed in eval mode, with no trainable parameters or gradients.
- All three new cross-attention layers received nonzero finite gradients.
- The two camera projectors have disjoint parameters, both receive gradients,
  and changing the agentview projector leaves wrist memory unchanged.
- Original decoder output and input gradients were exactly reproduced with the extra branch disabled/zeroed.
- Weight SHA-256 unchanged: `dd624dc41937d5ff9f13bb81184162d5eb84c165f3f6d4bfea8225d3b07785da`.
- Single-sample verification peak allocated CUDA memory: about 9.14 GiB (RTX 3090).

ADE20K has no robot class; robot/gripper regions can receive labels such as
plaything/person. Cups, bottles, basket, plates and table regions have useful
visible boundaries in the saved examples. This is an integration/qualitative
check, not a trained policy accuracy or segmentation benchmark.

The joint environment retains Torch 2.5.1+cu121, Transformers 4.57.0,
NumPy 1.26.4, and Pillow 12.3.0. Detectron2 0.6 was built against this Torch/CUDA
environment from the local source snapshot in `../dependencies/detectron2`.
OpenCLIP 2.24.0 and protobuf 6.32.0 avoid the older OpenCLIP/TensorFlow dependency
conflict. Setuptools 80.9.0 retains the upstream `pkg_resources` import.
`pip check` passes; an installed-version snapshot is saved beside the verification artifacts.
The upstream Decord 0.6 wheel mislabeled its internal tag as CPython 3.6 despite
being distributed as a Python 3 ctypes wheel. The local copy in
`../dependencies/wheels` corrects that metadata and RECORD only; its code and
shared libraries are unchanged. `../dependencies/fix_decord_wheel.py` records
the reproducible correction.

The saved segmentation samples are from existing 256x256 LIBERO RLDS RGB,
upsampled by the native EOVSAM test transform to 1152x1152. They are not native
1152 simulator frames. Original TurboVLA policy inputs are LIBERO 256x256 (two
cameras) and RoboTwin 224x224 (three cameras). High-resolution LIBERO training
data requires re-rendering the original demonstrations/full simulator states;
the RLDS 8-dimensional proprioception alone cannot reconstruct the whole scene.
Native 1152x1152 agentview and wrist rendering was successfully probed on the
server with `scripts/native_render_probe.py`; samples and shape checks are in
`../outputs/native_resolution_probe`. This probe does not regenerate or replace
the existing training dataset. Independent projection checks are saved in
`../outputs/eovsam_verification/independent_projections.json`.
