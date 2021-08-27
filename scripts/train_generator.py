"""
Train a diffusion model on images.
"""

import argparse

from diff_dalle import dist_util, logger
from diff_dalle.datasets import load_data
from diff_dalle.resample import create_named_schedule_sampler
from diff_dalle.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    create_classifier_and_diffusion,
    classifier_and_diffusion_defaults,
    args_to_dict,
    add_dict_to_argparser,
)
from diff_dalle.train_util import TrainLoop
import torch


def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    logger.configure()
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)
    
    logger.log("creating data loader...")
    data = load_data(
        index_dir=args.index_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        small_size=args.small_size,
        text_length=args.text_length,
        gaussian_blur=args.gaussian_blur,
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
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
        warmup_steps=args.warmup_steps,
    ).run_loop()


def create_argparser():
    defaults = dict(
        index_dir=None,
        schedule_sampler="uniform",
        lr=3e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        warmup_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=100,
        save_interval=2500,
        resume_checkpoint="",
        fp16_scale_growth=1e-3,
        text_length=64,
        gaussian_blur=False,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_and_diffusion_defaults())    
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
