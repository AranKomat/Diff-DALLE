import argparse
import os
from PIL import Image
import glob

import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F

from transformers import GPT2TokenizerFast

from diff_dalle import dist_util, logger
from diff_dalle.datasets import load_data
from diff_dalle.nn import clip_loss
from diff_dalle.script_util import (
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    _, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    schedule_sampler = create_named_schedule_sampler(
            args.schedule_sampler, diffusion
        )

    logger.log("loading classifier...")
    classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
    classifier.to(dist_util.dev())
    classifier.eval()
    
    resume_step = parse_resume_step_from_filename(args.clip_resume_checkpoint)
    if dist.get_rank() == 0:
        logger.log(
            f"loading model from checkpoint: {args.clip_resume_checkpoint}... at {resume_step} step"
        )
    dist.barrier()
    classifier.load_state_dict(
        dist_util.load_state_dict(
            args.clip_resume_checkpoint, map_location=dist_util.dev()
        )
    )

    dist_util.sync_params(classifier.parameters())
    
    if args.use_fp16:
        classifier.convert_to_fp16()
        
    classifier = DDP(
        classifier,
        device_ids=[dist_util.dev()],
        output_device=dist_util.dev(),
        broadcast_buffers=False,
        bucket_cap_mb=128,
        find_unused_parameters=False,
    )
    
    logger.log("preparing dataloder...")
    data = load_data(
        index_dir=args.index_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        small_size=args.small_size,
        text_length=args.text_length,
        text_aug_factor=args.text_aug_factor,
        phase='valid',
    )

    logger.log("sampling...")
    all_embs = []
    while len(all_embs) * args.batch_size < args.num_embs:
        batch, cond = next(data)
        batch = batch.to(dist_util.dev())
        if args.noised:
            t, _ = schedule_sampler.sample(batch.shape[0], dist_util.dev())
            batch = diffusion.q_sample(batch, t)
        else:
            t = th.zeros(batch.shape[0], dtype=th.long, device=dist_util.dev())
        model_kwargs = {k: v.to(dist_util.dev()) for k, v in cond.items()}
        with th.no_grad():
            img_emb, txt_emb, _ = classifier(batch, t, **model_kwargs)
            emb = th.stack([img_emb, txt_emb])

        gathered_embs = [th.zeros_like(emb) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_embs, emb)  # gather not supported with NCCL
        all_embs.extend([emb.cpu() for emb in gathered_embs])
        logger.log(f"created {len(all_embs) * args.batch_size} samples")

    all_embs = th.cat(all_embs, dim=1)
    all_embs = all_embs[:, : args.num_samples]
    if dist.get_rank() == 0:
        logger.log(f"saving to {out_path}")
        th.save(all_embs, args.save_emb_path)
                
    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        index_dir="",
        num_embs=2**20,
        batch_size=64,
        use_ddim=False,
        text_aug_factor=1,
        clip_resume_checkpoint="",
        text_length=48,
        save_emb_path="",
        noised=True,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()