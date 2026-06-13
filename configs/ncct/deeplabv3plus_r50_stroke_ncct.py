_base_ = [
    '_ncct_base.py',
    '../_base_/models/deeplabv3plus_r50-d8.py',
]

norm_cfg = dict(type='BN', requires_grad=True)

model = dict(
    backbone=dict(norm_cfg=norm_cfg),
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[0, 0, 0],
        std=[255, 255, 255],
        bgr_to_rgb=False),
    decode_head=dict(
        norm_cfg=norm_cfg,
        loss_decode=[
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
        ]),
    auxiliary_head=dict(
        norm_cfg=norm_cfg,
        loss_decode=[
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
        ]))
