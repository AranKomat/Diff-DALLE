"""
Use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

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

    logger.log("loading classifier...")
    classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
    if args.classifier_scale > 0:
        classifier.load_state_dict(
            dist_util.load_state_dict(args.classifier_path, map_location="cpu")
        )
    classifier.to(dist_util.dev())
    if args.use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()
    
    logger.log("preparing dataloder...")
    data = load_data(
        data_dir=args.data_dir,
        index_dir=args.index_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        small_size=args.small_size,
        text_length=args.text_length,
        text_aug_factor=args.text_aug_factor,
        phase='valid',
        text_loader=True,
    )
    
    def cond_fn(x, t, **model_kwargs):
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            loss = clip_loss(*classifier(x_in, t, **model_kwargs))
            return th.autograd.grad(loss, x_in)[0] * args.classifier_scale

    def model_fn(x, t, **kwargs):
        return model(x, t, **kwargs)

    if args.classifier_scale == 0:
        cond_fn = None
    logger.log("sampling...")
    all_images = []
    all_labels = []
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        batch, cond = next(data)
        
        model_kwargs = {k: v.to(dist_util.dev()) for k, v in cond.items()}
                
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model_fn,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=dist_util.dev(),
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()
        texts = tokenizer.batch_decode(cond["y"], skip_special_tokens=True)

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        gathered_labels = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered_labels, texts)
        all_labels.extend(sum(gathered_labels, []))
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    all_labels = all_labels[:args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr)
        
        with open(os.path.splitext(out_path)[0] + ".txt", "w") as txt_file:
            for line in all_labels:
                txt_file.write(line + "\n") 
        
        images_per_axis = args.grid_size // args.image_size
        arr = arr[: images_per_axis ** 2]
        arr = [Image.fromarray(img) for img in arr]
        grid = image_grid(arr, rows=images_per_axis, cols=images_per_axis).resize((args.grid_size, args.grid_size))
        grid.save(os.path.join(logger.get_dir(), "grid.jpg"))
        
        with open(os.path.join(logger.get_dir(), "grid.txt"), "w") as txt_file:
            for idx, line in enumerate(all_labels):
                txt_file.write(line + "\n")
                if idx == images_per_axis ** 2 - 1:
                    break
                
    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        index_dir=None,
        data_dir="",
        clip_denoised=True,
        num_samples=64,
        batch_size=64,
        use_ddim=False,
        text_aug_factor=1,
        model_path="",
        classifier_path="",
        classifier_scale=0,
        text_length=48,
        grid_size=512,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()