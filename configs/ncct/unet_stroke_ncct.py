_base_ = [
    '../_base_/models/fcn_unet_s5-d16.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py'
]

# Use BN instead of SyncBN — single-GPU training so they're equivalent,
# and BN avoids needing to compile MMCV CUDA ops (saves 30+ min).
norm_cfg = dict(type='BN', requires_grad=True)

# Model config for NCCT stroke segmentation
model = dict(
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        # NCCT images are grayscale stacked to 3-channel. Normalize to [0,1].
        mean=[0, 0, 0],
        std=[255, 255, 255],
        bgr_to_rgb=False,
        # U-Net has 5 downsampling stages (2^5=32), so pad spatial dims
        # to be divisible by 32 to avoid dimension mismatch in skip connections.
        size_divisor=32),
    decode_head=dict(
        num_classes=2,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            # Heavier weight on stroke (class 1) to combat class imbalance
            class_weight=[0.1, 1.0])),
    auxiliary_head=dict(
        num_classes=2,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=0.4,
            class_weight=[0.1, 1.0])))

# Dataset settings
dataset_type = 'StrokeNCCTDataset'
data_root = 'data/ncct/'

# Training pipeline with augmentations for medical images
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomResize', scale=(256, 256), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]

# Validation/test pipeline (fixed resize, no augmentation)
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

# Data loaders
train_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='train/images', seg_map_path='train/masks'),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='val/images', seg_map_path='val/masks'),
        pipeline=val_pipeline))

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='test/images', seg_map_path='test/masks'),
        pipeline=test_pipeline))

# Evaluators
val_evaluator = dict(type='IoUMetric', iou_metrics=['mDice', 'mIoU'])
test_evaluator = dict(type='IoUMetric', iou_metrics=['mDice', 'mIoU'])
