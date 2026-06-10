_base_ = [
    '../_base_/models/pspnet_unet_s5-d16.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py'
]

norm_cfg = dict(type='BN', requires_grad=True)

model = dict(
    backbone=dict(norm_cfg=norm_cfg),
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[0, 0, 0],
        std=[255, 255, 255],
        bgr_to_rgb=False,
        size_divisor=32),
    decode_head=dict(
        norm_cfg=norm_cfg,
        num_classes=2,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=[0.1, 1.0])),
    auxiliary_head=dict(
        norm_cfg=norm_cfg,
        num_classes=2,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=0.4,
            class_weight=[0.1, 1.0])),
    test_cfg=dict(mode='slide', crop_size=(256, 256), stride=(170, 170)))

optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005))

dataset_type = 'StrokeNCCTDataset'
data_root = 'data/ncct/'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomResize', scale=(256, 256), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
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

val_evaluator = dict(type='IoUMetric', iou_metrics=['mDice', 'mIoU'])
test_evaluator = dict(type='IoUMetric', iou_metrics=['mDice', 'mIoU'])
