# ncct-segmentation

## Overview

Fork of [OpenMMLab MMSegmentation v1.2.2](https://github.com/open-mmlab/mmsegmentation) with a custom binary stroke segmentation module for NCCT (Non-Contrast Computed Tomography) brain images.

**Custom additions** (fork's delta from upstream):
- `mmseg/datasets/stroke_ncct.py` — `StrokeNCCTDataset` (2 class: background, stroke)
- `configs/ncct/unet_stroke_ncct.py` — U-Net config for NCCT
- `colab/NCCT_Segmentation_V4.ipynb` — experiment notebook

Everything else is stock mmsegmentation.

## Entrypoints

```
tools/train.py <config> [--work-dir] [--resume] [--amp] [--cfg-options]
tools/test.py  <config> <checkpoint> [--out] [--show-dir] [--tta] [--cfg-options]
```

Distributed:
```
tools/dist_train.sh <config> <gpus>       # torch.distributed.launch
tools/dist_test.sh  <config> <ckpt> <gpus>
tools/slurm_train.sh <partition> <job> <config>   # srun
```

Single-image inference:
```
demo/image_demo.py <img> <config> <checkpoint> [--device] [--out-file]
```

Analysis tools under `tools/analysis_tools/`: benchmark, browse_dataset, get_flops, confusion_matrix, analyze_logs.

## Config system

Python-based configs via `mmengine.config.Config.fromfile()`. The config uses dicts merged from base configs:

```python
_base_ = [
    '../_base_/models/fcn_unet_s5-d16.py',   # model arch
    '../_base_/default_runtime.py',           # logging, vis, env
    '../_base_/schedules/schedule_20k.py'     # optimizer, lr, train cfg
]
```

Base configs live in `configs/_base_/{datasets,models,schedules}/`. Custom configs at `configs/ncct/`.

Key overrides pattern: import base, then redefine dict keys to override.

## Dataset

`StrokeNCCTDataset` extends `BaseSegDataset` (registered via `@DATASETS.register_module()`).

- 2 classes: `background` (index 0, black), `stroke` (index 1, white)
- Expects `.png` images and masks under `{data_root}/{split}/images/` and `{data_root}/{split}/masks/`
- Split dirs: `train/`, `val/`, `test/` (80/10/10 convention)
- Registered in `mmseg/datasets/__init__.py` as `StrokeNCCTDataset`
- Config uses `class_weight=[0.1, 1.0]` for imbalance

## Training schedule

- Default: 20k iterations (`schedule_20k.py`)
- Optimizer: SGD lr=0.01, momentum=0.9, PolyLR scheduler
- Validation every 2000 iters, checkpoint every 2000 iters
- Logging every 50 iters
- Iter-based (not epoch-based). `by_epoch=False` throughout.
- Work dir defaults to `./work_dirs/{config_name}/`

## Codebase conventions

- **Copyright header**: `# Copyright (c) OpenMMLab. All rights reserved.` in every `.py` file
- **Pre-commit** (run via `pre-commit run --all-files`): flake8, isort, yapf, trailing-whitespace, check-yaml, end-of-file-fixer, docformatter (--wrap-descriptions 79), pyupgrade (--py36-plus), codespell, check-algo-readme, check-copyright
- **isort**: line_length=79, known_first_party=mmseg, known_third_party sorted
- **yapf**: based_on_style=pep8
- **Registry system** (mmseg/registry/): 21 registries (MODELS, DATASETS, HOOKS, etc.), all children of MMEngine root registries. Modules decorate with `@MODELS.register_module()` / `@DATASETS.register_module()`. Build via `REGISTRY.build(cfg)`.
- **Model aliases** (mmseg/models/builder.py): BACKBONES=NECKS=HEADS=LOSSES=SEGMENTORS=MODELS (all use the same MODELS registry)
- **No type suppression**: `as any`, `@ts-ignore` patterns don't apply (Python). Don't add `# type: ignore` without reason.

## Testing

```
pip install -r requirements/tests.txt
pytest tests/ --ignore tests/test_models/test_backbones/test_timm_backbone.py --ignore tests/test_apis/test_rs_inferencer.py
```

Coverage: `coverage run --branch --source mmseg -m pytest tests/ ...`

Key test area: `tests/test_config.py` validates all configs load correctly.

## Dependencies

Core (runtime.txt): matplotlib, numpy, packaging, prettytable, scipy
MM: mmcv>=2.0.0rc4,<2.2.0 + mmengine>=0.5.0,<1.0.0 (installed separately, not in requirements)
Optional (optional.txt): timm, transformers, diffusers, clip, cityscapesscripts, albumentations

Install MMCV via `mim install mmcv>=2.0.0`. Do NOT pip install it.

## Colab notebook

`colab/NCCT_Segmentation_V4.ipynb` uses **segmentation-models-pytorch** (smp) for model definitions (not mmseg's model registry). It implements five architectures: UNet (resnet34), Attention UNet, ResUNet (resnet50), UNet_EffNet (efficientnet-b3), Linknet, FPN (resnext50), PSPNet, PAN. Uses BCE-Dice loss with focal loss variant. Dataset class is standalone PyTorch `Dataset`, not mmseg's `BaseSegDataset`. This is a separate experiment from the mmseg config pipeline.

## MIM packaging

`setup.py` `add_mim_extension()` creates symlinks/copies of `tools/`, `configs/`, `model-index.yml`, `dataset-index.yml` into `mmseg/.mim/` for OpenMMLab MIM compatibility.

## Git

Fork base is mmsegmentation upstream. Two custom commits on top of upstream history:
```
2de7aba1 add model         ← NCCT additions
6ee84d8f first commit       ← fork point (copy of mmsegmentation at some commit)
```
Remote: `origin/main` tracks the fork. Upstream branches available under `origin/1.x`, `origin/dev-1.x`, etc.
