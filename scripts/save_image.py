"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist

from diff_dalle.datasets import load_data 
from diff_dalle import dist_util, logger
from diff_dalle.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from transformers import GPT2TokenizerFast, RobertaTokenizerFast

from PIL import Image
import glob

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    print(args)
    
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        small_size=args.small_size,
        text_length=args.text_length,
        custom_dataset=args.custom_dataset,
    )
    batch, cond = next(data)
    
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    #tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    te = tokenizer.batch_decode(cond["y"], skip_special_tokens=True)
    print(te)
    all_images = []
    batch = batch.to(dist_util.dev())
    sample = ((batch + 1) * 127.5).clamp(0, 255).to(th.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()

    gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
    all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    arr = [Image.fromarray(arr[idx]) for idx in range(64)]
    grid = image_grid(arr, rows=8, cols=8).resize((512, 512))
    grid.save("grid2.jpg")
    
    
    logger.log("sampling...")
    all_images = []
    all_labels = []
    model_kwargs = {}

    model_kwargs = {k: v.to(dist_util.dev()) for k, v in cond.items()}
    sample_fn = (
        diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
    )
    sample = sample_fn(
        model,
        (args.batch_size, 3, args.image_size, args.image_size),
        clip_denoised=args.clip_denoised,
        model_kwargs=model_kwargs,
    )
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()

    gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
    all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    arr = [Image.fromarray(arr[idx]) for idx in range(64)]

    grid = image_grid(arr, rows=8, cols=8).resize((512, 512))
    grid.save("grid.jpg")
    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=64,
        batch_size=64,
        use_ddim=False,
        model_path="",
        text_length=48,
        custom_dataset=False,
        #enc_lr=3e-4,
        enc_lr=None,
        data_dir=None,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()