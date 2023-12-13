# dataset setting
dataset_type = "MFNet"
data_root = "MFNet/"
img_suffix = '.png'
reduce_zero_label = False
img_norm_cfg = dict(mean=[96.9255, 96.9255, 96.9255], std=[47.8976, 47.8976, 47.8976], to_rgb=True)
crop_size = (512, 512)  
train_pipeline = [
    dict(type="LoadImageFromFile"), 
    dict(type="LoadAnnotations"), 
    dict(type="Resize", img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    dict(type="RandomCrop", crop_size=crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_semantic_seg"]),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(2048, 512),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="thermal_imgs",
        ann_dir="labels",
        split="train.txt",
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="thermal_imgs",
        ann_dir="labels",
        split="val_test.txt",
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="thermal_imgs",
        ann_dir="labels",
        split="test.txt",
        pipeline=test_pipeline,
    ),
)
