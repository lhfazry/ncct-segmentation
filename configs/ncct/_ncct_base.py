# NCCT Stroke Segmentation — Shared Base Configuration
#
# This base config provides:
# 1. Multi-loss strategy (CE weighted + Dice) for severe class imbalance
# 2. Proper inverse-frequency class weights (98% bg / 2% stroke)
# 3. Enhanced augmentation pipeline for medical images
# 4. AMP optimizer (mixed precision)
# 5. Dataset / dataloader defaults for NCCT
#
# Usage: add '_ncct_base.py' to _base_ list BEFORE the model base.

_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py'
]

# ========== Class Imbalance Strategy ==========
# Dataset: 98% background, 2% stroke (49:1 ratio)
# [0.02, 1.0] gives equal gradient from both classes
# (was 2.45x bg bias with [0.05, 1.0])
# DiceLoss naive_dice=True gives stronger gradients when model is very wrong

loss_decode = [
    dict(
        type='CrossEntropyLoss',
        use_sigmoid=False,
        loss_weight=1.0,
        class_weight=[0.02, 1.0]),
    dict(
        type='DiceLoss',
        use_sigmoid=False,
        naive_dice=True,
        loss_weight=2.0)
]

loss_decode_aux = [
    dict(
        type='CrossEntropyLoss',
        use_sigmoid=False,
        loss_weight=0.4,
        class_weight=[0.02, 1.0]),
    dict(
        type='DiceLoss',
        use_sigmoid=False,
        naive_dice=True,
        loss_weight=0.8)
]

# AMP override for mixed precision training
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005))

# ---------- Dataset ----------
dataset_type = 'StrokeNCCTDataset'
data_root = 'data/ncct/'

# ---------- Data Pipelines ----------
# Enhanced training pipeline with imbalance-aware augmentations.
# Key additions vs the original pipeline:
# - RandomRotate: simulates head tilt in NCCT scans
# - RandomBrightnessContrast + GaussNoise: CT scanner noise variation
# - RandomGamma: simulates different window/level settings
# These are "safe" for medical images (don't change anatomical structure).

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    # Spatial
    dict(type='RandomResize', scale=(256, 256), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='RandomRotate', prob=0.5, degree=(-30, 30), pad_val=0, seg_pad_val=255),
    # Photometric (safe for CT)
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(256, 256), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(256, 256), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]

# ---------- Data Loaders ----------
# batch_size is PER GPU when using DDP.
# With 2 × T4 (16 GB each), we can comfortably fit batch 24 for U-Net
# architectures at 256×256 with AMP. Effective total batch = 48.
# For larger architectures (DeepLabV3+, PSPNet R-50), reduce to 16 if OOM.
# Pass --cfg-options train_dataloader.batch_size=16 to override.
train_dataloader = dict(
    batch_size=24,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='train/images', seg_map_path='train/masks'),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='val/images', seg_map_path='val/masks'),
        pipeline=val_pipeline))

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='test/images', seg_map_path='test/masks'),
        pipeline=test_pipeline))

# ---------- Evaluators ----------
val_evaluator = dict(type='IoUMetric', iou_metrics=['mDice', 'mIoU'])
test_evaluator = dict(type='IoUMetric', iou_metrics=['mDice', 'mIoU'])
