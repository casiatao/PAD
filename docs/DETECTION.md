## Object detection

Firstly, go into the detectopm directory `cd ./detection/projects/ViTDet/train_net.py`

##### pre-training with adapter (PAD)

```bash
python train_net.py --config-file configs/FLIR/mask_rcnn_vitdet_b_adapt_12ep_step.py \
	--num-gpus=<num of gpus> train.output_dir=<output path> \
    model.backbone.net.down_size=64 model.backbone.net.adapt_scalar="frozen" \
    model.backbone.net.patch_wise_scalar=True train.seed=42 \
    train.init_checkpoint=<model pre-trained on MSIP with PAD>
```

##### other paradigms

```bash
python train_net.py --config-file configs/FLIR/mask_rcnn_vitdet_b_12ep_step.py \
	--num-gpus=<num of gpus> train.output_dir=<output path> \
    train.seed=42 \
    train.init_checkpoint=<model pre-trained on MSIP with corresponding paradigm>
```

