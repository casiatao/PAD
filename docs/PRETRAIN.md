## Pre-training

### RoI generation

Use the script `/pretraining/dataset/roi.py` to generate candidate RoIs for each sub-dataset: 

```bash
python roi.py --base_path=<path of MSIP> --dataset=<name of sub-dataset>
```

### Pre-training with MAE

Firstly, use the script `pre-training/mae/change_mae_mm.py` to convert the pre-trained model from MMSelfSup into a model structure suitable for the MAE pre-training code.

Then go into the mae directory `cd ./pretraining/mae/`.

##### pre-training from scratch

```bash
python -m torch.distributed.launch --nproc_per_node=<num of gpus> main_pretrain.py \
    --batch_size=<batch size per gpu> --accum_iter=<accum iter> --model mae_vit_base_patch16 --pin_mem \
    --output_dir=<output path> --log_dir=<log path> \
    --norm_pix_loss --mask_ratio=0.75 --warmup_epochs=40 --epochs=400 --blr=1e-4 \
    --use_roi --random --ratio_p=0.85 \
```

##### pre-training from IN1K

```bash
python -m torch.distributed.launch --nproc_per_node=<num of gpus> main_pretrain.py \
    --batch_size=<batch size per gpu> --accum_iter=<accum iter> --model mae_vit_base_patch16 --pin_mem \
    --pretrain=<model pre-trained on IN1K with MAE> --output_dir=<output path> --log_dir=<log path> \
    --norm_pix_loss --mask_ratio=0.75 --warmup_epochs=30 --epochs=100 --blr=1e-4 \
    --use_roi --random --ratio_p=0.85
```

##### pre-training from IN1K + layerwise-decay learning rate

```bash
python -m torch.distributed.launch --nproc_per_node=<num of gpus> main_pretrain.py \
    --batch_size=<batch size per gpu> --accum_iter=<accum iter> --model mae_vit_base_patch16 --pin_mem \
    --pretrain=<model pre-trained on IN1K with MAE> --output_dir=<output path> --log_dir=<log path> \
    --norm_pix_loss --mask_ratio=0.75 --warmup_epochs=30 --epochs=100 --blr=1e-4 \
    --use_roi --random --ratio_p=0.85 --layer_decay=0.7
```

##### pre-training with adapter (PAD)

```bash
python -m torch.distributed.launch --nproc_per_node=<num of gpus> main_pretrain.py \
    --batch_size=<batch size per gpu> --accum_iter=<accum iter> --model mae_vit_base_patch16 --pin_mem \
    --pretrain=<model pre-trained on IN1K with MAE> --output_dir=<output path> --log_dir=<log path> \
    --norm_pix_loss --mask_ratio=0.75 --warmup_epochs=30 --epochs=100 --blr=1e-4 \
    --use_roi --random --ratio_p=0.85 \
    --adapt --patch_wise_scalar --fusion_method=concat --later_study 
```

The effective batch size is `batch_size` \* `nproc_per_node`\* `accum_iter` = 4096. 

### Pre-training with MILAN

Firstly, use the script `pre-training/milan/change_milan_mm.py` to convert the pre-trained model from MMSelfSup into a model structure suitable for the MILAN pre-training code.

Then go into the milan directory `cd ./pretraining/milan/`.

##### pre-training from IN1K + layerwise-decay learning rate

```bash
python -m torch.distributed.launch --nproc_per_node=<num of gpus> main_pretrain_adapt.py \
    --batch_size=<batch size per gpu> --accum_iter=<accum iter> --model milan_vit_base_patch16 --pin_mem \
    --pretrain=<model pre-trained on IN1K with MILAN> --output_dir=<output path> --log_dir=<log path> \
    --norm_pix_loss --mask_ratio=0.75 --warmup_epochs=30 --epochs=100 --blr=1e-4 --use_clip --attn_mask \
    --use_roi --random --ratio_p=0.85 --layer_decay=0.7
```

##### pre-training with adapter (PAD)

```bash
python -m torch.distributed.launch --nproc_per_node=<num of gpus> main_pretrain_adapt.py \
    --batch_size=<batch size per gpu> --accum_iter=<accum iter> --model milan_vit_base_patch16 --pin_mem \
    --pretrain=<model pre-trained on IN1K with MILAN> --output_dir=<output path> --log_dir=<log path> \
    --norm_pix_loss --mask_ratio=0.75 --warmup_epochs=30 --epochs=100 --blr=1e-4 --use_clip --attn_mask \
    --use_roi --random --ratio_p=0.85 \
    --adapt --patch_wise_scalar --fusion_method=concat --later_study 
```

The effective batch size is `batch_size` \* `nproc_per_node`\* `accum_iter` = 4096.