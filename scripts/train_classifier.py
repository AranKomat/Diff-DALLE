"""
Train a noised image classifier on ImageNet.
"""

import argparse
import os
from time import time 
import numpy as np

import blobfile as bf
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from diff_dalle import dist_util, logger
from diff_dalle.dist_util import cached
from diff_dalle.nn import clip_loss 
from diff_dalle.fp16_util import MixedPrecisionTrainer
from diff_dalle.datasets import load_data
from diff_dalle.resample import create_named_schedule_sampler
from diff_dalle.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    classifier_and_diffusion_defaults,
    create_classifier_and_diffusion,
)
from diff_dalle.train_util import parse_resume_step_from_filename, log_loss_dict


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_classifier_and_diffusion(
        **args_to_dict(args, classifier_and_diffusion_defaults().keys())
    )
    print(args)
    model.to(dist_util.dev())
    if args.noised:
        schedule_sampler = create_named_schedule_sampler(
            args.schedule_sampler, diffusion
        )

    resume_step = 0
    if args.clip_resume_checkpoint:
        resume_step = parse_resume_step_from_filename(args.clip_resume_checkpoint)
        if dist.get_rank() == 0:
            logger.log(
                f"loading model from checkpoint: {args.clip_resume_checkpoint}... at {resume_step} step"
            )
        dist.barrier()
        model.load_state_dict(
            dist_util.load_state_dict(
                args.clip_resume_checkpoint, map_location=dist_util.dev()
            )
        )

    # Needed for creating correct EMAs and fp16 parameters.
    dist_util.sync_params(model.parameters())

    mp_trainer = MixedPrecisionTrainer(
        model=model, use_fp16=args.use_fp16, initial_lg_loss_scale=16.0
    )

    model_fn = lambda model: DDP(
        model,
        device_ids=[dist_util.dev()],
        output_device=dist_util.dev(),
        broadcast_buffers=False,
        bucket_cap_mb=128,
        find_unused_parameters=False,
    )
    
    ddp_model = {"text": model_fn(model.text_encoder), "image": model_fn(model.image_encoder), "scale": model_fn(model.logit_scale)}

    logger.log("creating data loader...")
    data = load_data(
        index_dir=args.index_dir,
        batch_size=args.microbatch,
        image_size=args.image_size,
        random_crop=True,
        text_length=args.text_length,
        num_workers=8,
    )
    if args.index_dir_val:
        val_data = load_data(
            index_dir=args.index_dir_val,
            batch_size=args.microbatch,
            image_size=args.image_size,
            text_length=args.text_length,
            phase='valid',
        )
    else:
        val_data = None

    logger.log(f"creating optimizer...")
    opt = AdamW(mp_trainer.master_params, lr=args.lr, weight_decay=args.weight_decay)
    if args.clip_resume_checkpoint:
        opt_checkpoint = bf.join(
            bf.dirname(args.clip_resume_checkpoint), f"opt{resume_step:06}.pt"
        )
        dist.barrier()
        logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
        opt.load_state_dict(
            dist_util.load_state_dict(opt_checkpoint, map_location=dist_util.dev())
        )

    logger.log("training classifier model...")
    
    def print_time(name, t):
        rank = dist.get_rank()
        print(name, rank, args.batch_size / (time() - t))
        
    def forward_backward_log(data_loader, prefix="train"):
        cache_x = []
        cache_y = []
        closuresx = []
        closuresy = []
        aggregate = lambda x: th.cat(x)
        microbatch = args.microbatch if args.microbatch > 0 else args.batch_size
        num_microbatches = args.batch_size // microbatch

        for i in range(num_microbatches):
            t = time()
            micro, micro_cond = next(data_loader)
            micro = micro.to(dist_util.dev())
            micro_text = micro_cond.to(dist_util.dev())
            # Noisy images
            if args.noised:
                micro_t, _ = schedule_sampler.sample(micro.size(0), dist_util.dev())
                micro = diffusion.q_sample(micro, micro_t)
            else:
                micro_t = th.zeros(micro.size(0), dtype=th.long, device=dist_util.dev())

            last_step = i == num_microbatches - 1

            if num_microbatches == 1:
                out_img = ddp_model["image"](micro, micro_t)
                out_txt = ddp_model["text"](micro_text)
                logit_scale = ddp_model["scale"](out_img.dtype)
                loss = clip_loss(out_img, out_txt, logit_scale)
                mp_trainer.backward(loss)                            
                
            elif prefix == "train":
                rx, cx = cached(ddp_model["image"], last_step)(micro, micro_t)
                ry, cy = cached(ddp_model["text"], last_step)(micro_text)

                cache_x.append(rx)
                cache_y.append(ry)
                closuresx.append(cx)
                closuresy.append(cy)
                
                if last_step:
                    logit_scale = ddp_model["scale"](rx.dtype)
                    loss = clip_loss(*map(aggregate, (cache_x, cache_y)), logit_scale)
                    logger.logkv_mean(f"{prefix}_loss", loss.detach().item())
                    mp_trainer.backward(loss)            

                    for f, r in zip(closuresx, cache_x):
                        f(r)
                    for f, r in zip(closuresy, cache_y):
                        f(r)
            else:
                rx, ry = ddp_model["image"](micro, micro_t), ddp_model["text"](micro_text) 
                cache_x.append(rx)
                cache_y.append(ry)
                if last_step:
                    logit_scale = ddp_model["scale"](rx.dtype)
                    cache_x, cache_y = map(aggregate, (cache_x, cache_y))
                    loss = clip_loss(cache_x, cache_y, logit_scale)  
                    logger.logkv_mean(f"{prefix}_loss", loss.detach().item())
        
        
    for step in range(args.iterations - resume_step):
        logger.logkv("step", step + resume_step)
        logger.logkv(
            "samples",
            (step + resume_step + 1) * args.batch_size * dist.get_world_size(),
        )
        if args.anneal_lr:
            set_annealed_lr(opt, args.lr, step=step + resume_step, total_steps=args.iterations, warmup_steps=args.warmup_steps)
        if val_data is not None and not step % args.eval_interval:
            with th.no_grad():
                with model.no_sync():
                    model.eval()
                    forward_backward_log(val_data, prefix="val")
                    model.train()
        t = time()
        forward_backward_log(data)
        mp_trainer.optimize(opt)
        mp_trainer.zero_grad()                
        print_time("step", t)
        if not step % args.log_interval:
            logger.dumpkvs()
        if (step
            and dist.get_rank() == 0
            and not (step + resume_step) % args.save_interval
        ):
            logger.log("saving model...")
            save_model(mp_trainer, opt, step + resume_step)

    if dist.get_rank() == 0:
        logger.log("saving model...")
        save_model(mp_trainer, opt, step + resume_step)
    dist.barrier()
    
    
def set_annealed_lr(opt, peak_lr, step, total_steps, warmup_steps):
    end_lr = 0.1 * peak_lr
    warmup_pct = min(max(step, 0), warmup_steps) / warmup_steps
    anneal_pct = min(max(step - warmup_steps, 0), total_steps) / total_steps
    lr = warmup_pct * peak_lr - (peak_lr - end_lr) * (1 - np.cos(np.pi * anneal_pct)) / 2

    for param_group in opt.param_groups:
        param_group["lr"] = lr

def save_model(mp_trainer, opt, step):
    if dist.get_rank() == 0:
        th.save(
            mp_trainer.master_params_to_state_dict(mp_trainer.master_params),
            os.path.join(logger.get_dir(), f"model{step:06d}.pt"),
        )
        th.save(opt.state_dict(), os.path.join(logger.get_dir(), f"opt{step:06d}.pt"))

        
def create_argparser():
    defaults = dict(
        index_dir=None,
        index_dir_val=None,
        noised=True,
        iterations=400000,
        lr=5e-4,
        weight_decay=0.2,
        anneal_lr=True,
        warmup_steps=3000,
        batch_size=64,
        text_length=64,
        microbatch=-1,
        schedule_sampler="uniform",
        clip_resume_checkpoint="",
        log_interval=1000,
        eval_interval=1000,
        save_interval=40000,
    )
    defaults.update(classifier_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()