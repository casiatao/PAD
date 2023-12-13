from functools import partial
from fvcore.common.param_scheduler import MultiStepParamScheduler, CosineParamScheduler

from detectron2 import model_zoo
from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler, WarmupCosineLR, WarmupMultiStepLR
from detectron2.modeling.backbone.vit import get_vit_lr_decay_rate
from detectron2.evaluation import COCOEvaluator

from ..common.coco_loader_lsj import dataloader


dataloader.train.dataset.names = "flir_train" 
dataloader.test.dataset.names = "flir_val"
dataloader.train.total_batch_size = 2  # batch_size
dataloader.train.num_workers = 8
dataloader.test.num_workers = 8
dataloader.train.mapper.use_instance_mask = False
dataloader.train.mapper.recompute_boxes = False

dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="${..test.dataset.names}",
    output_dir="test/"
)

model = model_zoo.get_config("common/models/mask_rcnn_vitdet.py").model
model.roi_heads.mask_in_features = None
model.roi_heads.num_classes = 3 
model.pixel_mean = [96.9255, 96.9255, 96.9255]
model.pixel_std = [47.8976, 47.8976, 47.8976]


# Initialization and trainer settings
train = model_zoo.get_config("common/train.py").train
train.amp.enabled = True
train.ddp.fp16_compression = True
train.log_period = 100
train.checkpointer = dict(period=10000, max_to_keep=1)
train.max_iter = 53172  #  12ep = 5300 iters * 2 images/iter / 8862 images/ep
train.eval_period = 1000
train.init_checkpoint = (
    None
)  
train.seed = 42

# Optimizer
optimizer = model_zoo.get_config("common/optim.py").AdamW
optimizer.params.lr_factor_func = partial(get_vit_lr_decay_rate, num_layers=12, lr_decay_rate=0.7)
optimizer.params.overrides = {"pos_embed": {"weight_decay": 0.0}}



lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
        milestones=[35448, 48741],
        num_updates=train.max_iter,
    ),
    warmup_length=250 / train.max_iter,
    warmup_factor=0.001,
)


