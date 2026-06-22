_base_ = [
    '_ncct_base.py',
    '../_base_/models/fcn_unet_s5-d16.py',
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
        init_cfg=dict(
            type='Normal', std=0.01,
            override=dict(name='conv_seg')),
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=True,
                class_weight=[0.02, 1.0],
                loss_weight=1.0),
            dict(
                type='DiceLoss',
                use_sigmoid=True,
                naive_dice=True,
                loss_weight=1.0),
        ]),
    auxiliary_head=dict(
        norm_cfg=norm_cfg,
        init_cfg=dict(
            type='Normal', std=0.01,
            override=dict(name='conv_seg')),
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=True,
                class_weight=[0.02, 1.0],
                loss_weight=0.4),
            dict(
                type='DiceLoss',
                use_sigmoid=True,
                naive_dice=True,
                loss_weight=0.4),
        ]))

# U-Net + decoder head + auxiliary head exceeds 14.5 GiB T4 at batch_size=12.
train_dataloader = dict(batch_size=6)
