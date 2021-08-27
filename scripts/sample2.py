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

from diff_dalle.simple_tokenizer import SimpleTokenizer

from diff_dalle import dist_util
from diff_dalle.datasets import load_data, tokenize
from diff_dalle.nn import approx_clip_loss
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
    device = th.device(args.device)

    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(args.model_path)
    model.to(device)
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    if args.classifier_scale > 0:
        print("loading classifier...")
        classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
            classifier.load_state_dict(args.classifier_path)
        classifier.to(device)
        if args.use_fp16:
            classifier.convert_to_fp16()
        classifier.eval()
    
    model_kwargs = {}
    if args.small_size > 0:
        print("upscaling ... we set the batch size to 1")
        args.batch_size = 1
        arr = np.array(Image.open(args.load_img_path).convert('RGB'))
        arr = arr.astype(np.float32) / 127.5 - 1        
        arr = np.transpose(arr, [2, 0, 1])
        model_kwargs["low_res"] = th.tensor(arr, device=device).unsqueeze(0)
    else:
        tokenizer = SimpleTokenizer()
        text = tokenize(tokenizer, args.prompt, args.text_length)
        model_kwargs["y"] = th.tensor(text, device=device).repeat(args.batch_size, 1)
    
    saved_img_embs, saved_txt_embs = th.load(args.emb_path).to(device).unbind(dim=0)
    assert args.batch_size == args.num_duplicates * args.input_batch_size
    
    def classifier_fn(x_t, t, model_kwargs):
        out_img = classifier.image_encoder(x_t, t)
        out_txt = classifier.text_encoder(model_kwargs["y"])
        logit_scale = classifier.logit_scale(out_img.dtype)
        return out_img, out_txt, logit_scale
    
    def cond_fn(x, t, **model_kwargs):
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            loss = approx_clip_loss(*classifier_fn(classifier x_in, t, **model_kwargs), saved_img_embs, saved_txt_embs, args.contra_batch_size)
            return th.autograd.grad(loss, x_in)[0] * args.classifier_scale

    if args.classifier_scale == 0:
        cond_fn = None
    print("sampling...")

    sample_fn = (
        diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
    )
    sample = sample_fn(
        model,
        (args.batch_size, 3, args.image_size, args.image_size),
        clip_denoised=args.clip_denoised,
        model_kwargs=model_kwargs,
        cond_fn=cond_fn,
        device=device,
    )
    images = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
    images = images.permute(0, 2, 3, 1)
    images = images.contiguous()
    images = [Image.fromarray(img.cpu().numpy()) for img in images]
    
    # sort images in terms of cosine distance of clip embeddings
    if args.sort_opt:
        if args.sort_opt == 'clip':
            from CLIP import clip
            clip = clip.load(args.clip_model, jit=False)[0].eval().requires_grad_(False).to(device)
            text_embs = clip.encode_text(clip.tokenize(text).to(device))
            # resize_image(Image.open(fetch(path)).convert('RGB'), (sideX, sideY))
            image_embs = clip.encode_image(normalize(images))
        elif args.sort_opt == 'classifier':
            ...sample
        prod = image_embs @ text_embds
        sample = images[prod.argsort(dim=-1, descending=True)][:args.topk]
        
    for idx, img in enumerate(sample):
        os.makedirs(args.save_img_dir, exist_ok = False) 
        img.save(os.path.join(args.save_img_dir, f"{idx}.jpg"))
    
    print(f"sampling complete for image_size = {args.image_size}")
    

def create_argparser():
    defaults = dict(
        clip_denoised=True,
        batch_size=32,
        sort_opt=None,
        topk=10000,
        contra_batch_size=2**15,
        use_ddim=False,
        model_path="",
        classifier_path="",
        classifier_scale=0,
        text_length=64,
        save_img_dir="./saved_images",
        load_img_path="",
        device="cuda:0",
        emb_path="",
        sort=False,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()