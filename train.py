"""
Train a diffusion model on images.
"""

import argparse
import os
from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop
from dataloader_scripts.load_pet_2_5D import LoadValData, load_data
from torch.utils.data import DataLoader

import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn as nn
def main():
    args = create_argparser().parse_args()
    args.in_channels = args.load_adj * 2 + 1 + args.out_channels
    
    dist_util.setup_dist()
    logger.configure(dir=args.logdir)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")

    train_dir = os.path.join(args.data_root, "train")
    data = load_data(args.batch_size, root_dir=train_dir, axis=args.train_axis, load_adj = args.load_adj)
    val_dir = os.path.join(args.data_root, "val")
    val_data = LoadValData(root_dir=val_dir, axis=args.train_axis, load_adj = args.load_adj)

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        val_data=val_data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        logdir = args.logdir,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=200000,
        batch_size=24,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=100,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        train_axis="z", # axis of model train on: x, y, or z
        load_adj=8, # number of adjacent slices loaded as 2.5D condition input
        out_channels=1,
        logdir="checkpoint_z", # save checkpoint and log in this folder
        data_root="NAC_data",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
