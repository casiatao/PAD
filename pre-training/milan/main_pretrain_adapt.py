import argparse
import datetime
import json
import numpy as np
import os
import sys
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DistributedSampler

import timm

assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from torchvision.transforms.functional import InterpolationMode

import model_milan_adapt

from util.msip_dataset import RandomRoiCrop, RoICompose, MSIP
from torchvision.transforms import RandomResizedCrop, RandomHorizontalFlip, ToTensor, Normalize, Resize, Compose
from util.log import Logger
import util.lr_decay as lrd
from engine_pretrain_adapt import train_one_epoch
from clip_demo import build_clip_model
from slip_demo import build_slip_model
from dino_demo import build_dino_model



def get_args_parser():
    parser = argparse.ArgumentParser('MILAN pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='milan_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=30, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='./MSIP/', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--pretrain', default='',
                        help='init with pretrain checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--layer_decay', type=float, default=1.0)
    parser.add_argument('--spec_dataset', default=None, type=str)
    
    parser.add_argument('--adapt', action='store_true', default=False)
    parser.add_argument('--down_size', default=64, type=int)
    parser.add_argument('--adapt_scalar', default="0.5", type=str)
    parser.add_argument('--init_value', default="0.5", type=str)
    parser.add_argument("--patch_wise_scalar", action='store_true', default=False)
    parser.add_argument("--fusion_method", default='concat', type=str)
    parser.add_argument("--later_study", action='store_true', default=False)

    
    parser.add_argument('--use_clip', action='store_true', default=False) # use CLIP as reconstruction target
    parser.add_argument('--average_targets', type=int, default=1) # average top-K blocks' output from CLIP as target
    parser.add_argument('--attn_mask', action='store_true', default=False) # attention based importance sampling masking
    parser.add_argument('--cluster_loss', action='store_true', default=False) # use cluster loss together with normalized mse loss
    parser.add_argument('--use_clip_slip', action='store_true', default=False) # use SLIP for the role of CLIP
    parser.add_argument('--use_clip_dino', action='store_true', default=False) # use DINO for the role of CLIP
    parser.add_argument('--use_ablation_model', action='store_true', default=False)
    
    parser.add_argument('--use_roi', action='store_true', default=False)
    parser.add_argument('--random', action='store_true', default=False)
    parser.add_argument('--ratio_p', default=0.5, type=float)
    parser.add_argument('--data_ratio', default=1.0, type=float)
    parser.add_argument('--scale_epoch', default=0.6, type=float)

    return parser


def main(args):
    t = time.strftime("-%Y%m%d-%H%M%S", time.localtime()) 
    filename = 'log' + t + '.txt'
    log = Logger(os.path.join(args.output_dir, filename))
    sys.stdout = log
    
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # simple augmentation
    transform_train = RoICompose([
            RandomRoiCrop(size=(224, 224), random=args.random, use_roi=args.use_roi, p=args.ratio_p, scale=(0.2, 1.0), interpolation=InterpolationMode.BICUBIC),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize(mean=[0.3801, 0.3801, 0.3801], std=[0.1871, 0.1871, 0.1871])
            ])
    if args.spec_dataset is not None:
        args.spec_dataset = args.spec_dataset.split(',')
    dataset_train = MSIP(args.data_path, transforms=transform_train, use_roi=args.use_roi, spec_dataset=args.spec_dataset, data_ratio=args.data_ratio)

    
    print(dataset_train)

    if args.cluster_loss:
        print('Load clustering results')
        cluster_result = torch.load('./clusters_0', map_location='cpu')
    else:
        cluster_result = None

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    
    # define the model
    model = model_milan_adapt.__dict__[args.model](norm_pix_loss=args.norm_pix_loss, 
                                                 use_clip=args.use_clip, 
                                                 attn_mask=args.attn_mask,
                                                 adapt=args.adapt, 
                                                 down_size=args.down_size, 
                                                 adapt_scalar=args.adapt_scalar, 
                                                 init_value=args.init_value,
                                                 patch_wise_scalar=args.patch_wise_scalar, 
                                                 fusion_method=args.fusion_method) 
    
    if args.use_clip:
        if args.use_clip_slip is False and args.use_clip_dino is False:
            # clip_model = build_clip_model(torch.load('./clip_vit_base_16.pth.tar'), 'base')
            clip_model = build_clip_model(torch.load('./pretrain/clip_vit_base_mm.pth', map_location='cpu')['model'])
        elif args.use_clip_slip:
            def process_state_dict(state_dict):
                for k in list(state_dict.keys()):
                    state_dict[k.replace('module.', '')] = state_dict.pop(k)
                return state_dict
            clip_model = build_slip_model(process_state_dict(torch.load('./pretrain/slip_vit_base_16.pth.tar', map_location='cpu')['state_dict']), 'base')
        elif args.use_clip_dino:
            clip_model = build_dino_model()
    else:
        clip_model = None

    model.to(device)

    if args.use_clip:
        clip_model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

        if args.use_clip:
            clip_model = torch.nn.parallel.DistributedDataParallel(clip_model, device_ids=[args.gpu], find_unused_parameters=True)
    
    # following timm: set wd as 0 for bias and norm layers
    if args.layer_decay != 1.0:
        param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,  
            no_weight_decay_list=model_without_ddp.no_weight_decay(), layer_decay=args.layer_decay)

    else:
        param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    # param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    
    # freeze other parameters
    if args.adapt:
        for name, p in model_without_ddp.named_parameters():
            if "decoder_embed" in name:
                p.requires_grad = False
            elif "adaptmlp" in name or "encoder_pred" in name or "scalar_pred" in name or "decoder" in name or "mask_token" in name:
                p.requires_grad = True
            else:
                p.requires_grad = False
    
    for name, p in model.module.named_parameters():
        print(name, p.requires_grad)
    
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)  
        
        if args.later_study:
            if epoch <= int(round(args.scale_epoch * args.epochs)):
                print("freeze parameters of scalar pred layers")
                for name, p in model_without_ddp.named_parameters():
                    if 'scalar_pred' in name:
                        p.requires_grad = False
            else:
                print("unfreeze parameters of scalar pred layers")
                for name, p in model_without_ddp.named_parameters():
                    if 'scalar_pred' in name:
                        p.requires_grad = True
        

        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler, args.epochs - epoch,
            log_writer=log_writer,
            args=args,
            clip_model=clip_model,
            cluster_result=cluster_result,
        )
        if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
