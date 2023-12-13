## Semantic segmentation

Firstly, go into the segmentation directory `cd ./segmentation/`.

#### SODA

##### pre-training with adapter (PAD)

```bash
tools/dist_train.sh configs/mae/upernet_mae_base_adapt_12_512_slide_14k_soda.py <num of gpus> --seed 0 \
	--options model.pretrained=<model pre-trained on MSIP with PAD> \
    model.backbone.down_size=64 model.backbone.adapt_scalar="frozen" model.backbone.patch_wise_scalar=True \
	log_config.hooks.1.log_dir=<tensorboard log path> --work-dir=<other log path> \
```

##### other paradigms

```bash
tools/dist_train.sh configs/mae/upernet_mae_base_12_512_slide_14k_soda.py <num of gpus> --seed 0 \
	--options model.pretrained=<model pre-trained on MSIP with corresponding paradigm> \
	log_config.hooks.1.log_dir=<tensorboard log path> --work-dir=<other log path> \
```

#### MFNet

##### pre-training with adpter (PAD)

```bash
tools/dist_train.sh configs/mae/upernet_mae_base_adapt_12_512_slide_9k_mfnet.py <num of gpus> --seed 0 \
	--options model.pretrained=<model pre-trained on MSIP with PAD> \
    model.backbone.down_size=64 model.backbone.adapt_scalar="frozen" model.backbone.patch_wise_scalar=True \
	log_config.hooks.1.log_dir=<tensorboard log path> --work-dir=<other log path> \
```

##### other paradigms

```bash
tools/dist_train.sh configs/mae/upernet_mae_base_12_512_slide_9k_mfnet.py <num of gpus> --seed 0 \
	--options model.pretrained=<model pre-trained on MSIP with corresponding paradigm> \
	log_config.hooks.1.log_dir=<tensorboard log path> --work-dir=<other log path> \
```

### 