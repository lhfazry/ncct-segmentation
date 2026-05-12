_base_ = [
    '../_base_/models/fcn_unet_s5-d16.py', # Contoh menggunakan base U-Net
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py'  # Jadwal training untuk 20.000 iterasi
]

# 1. Sesuaikan Arsitektur Model
model = dict(
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        # Normalisasi ke format RGB umum agar pre-trained weight bekerja optimal
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    decode_head=dict(
        num_classes=2, # Diubah menjadi 2 class (Background & Stroke)
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, 
            class_weight=[0.1, 1.0] # Memberikan bobot lebih besar pada piksel stroke (1.0) vs background (0.1) untuk mengatasi imbalance
        )
    )
)

# 2. Sesuaikan Dataset
dataset_type = 'StrokeNCCTDataset'
data_root = '../dataset/' # Path direktori data Anda (80/10/10)

train_pipeline = [
    dict(type='LoadImageFromFile'), # Otomatis meload sebagai 3-channel RGB
    dict(type='LoadAnnotations'),
    dict(type='RandomResize', scale=(256, 256), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='train/images', seg_map_path='train/masks'),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='val/images', seg_map_path='val/masks'),
        pipeline=train_pipeline)) # Pipeline val biasanya tanpa augmentasi RandomFlip
        
test_dataloader = dict(
    dataset=dict(
        data_prefix=dict(img_path='test/images', seg_map_path='test/masks')))