"""
Various utilities for neural networks.
"""

import math
import numpy as np

import torch as th
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

# PyTorch 1.7 has SiLU, but we support PyTorch 1.5.
class SiLU(nn.Module):
    def forward(self, x):
        return x * th.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)

    
class LayerNorm32(nn.LayerNorm):
    def forward(self, x):
        return super().forward(x.float().transpose(1, 2)).type(x.dtype).transpose(1, 2)

    
def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.

    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(th.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with th.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with th.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = th.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads

    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = th.zeros(max_len, d_model)
        position = th.arange(0, max_len, dtype=th.float).unsqueeze(1)
        div_term = th.exp(th.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = th.sin(position * div_term)
        pe[:, 1::2] = th.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].type(x.dtype)
        return self.dropout(x)
    
    
class FFN(nn.Module):
    def __init__(self, d_model, dropout=0):
        super().__init__()
        layers = []
        d_ffn = 4 * d_model 
        layers += [LayerNorm32(d_model), conv_nd(1, d_model, d_ffn, 1), nn.Dropout(p=dropout)]
        layers += [nn.GELU(), conv_nd(1, d_ffn, d_model, 1), nn.Dropout(p=dropout)]
        self.ffn = nn.Sequential(*layers)
            
    def forward(self, x):
        return self.ffn(x) + x
    
    
def clip_loss(image_features, text_features, logit_scale):
    rank = dist.get_rank()
    
    bs = len(image_features)
    
    text_features_all = [th.zeros_like(text_features) for _ in range(dist.get_world_size())]
    image_features_all = [th.zeros_like(image_features) for _ in range(dist.get_world_size())]
    
    dist.all_gather(text_features_all, text_features)
    dist.all_gather(image_features_all, image_features)  
    
    # image loss
    ground_truth = th.arange(rank*bs, (rank+1)*bs, device=image_features.device)
    loss_image = F.cross_entropy(logit_scale * image_features @ th.cat(text_features_all).t(), ground_truth)

    # text loss
    ground_truth = th.arange(rank*bs, (rank+1)*bs, device=text_features.device)
    loss_text = F.cross_entropy(logit_scale * text_features @ th.cat(image_features_all).t(), ground_truth)

    return (loss_image + loss_text) / 2


def approx_clip_loss(img_embs, txt_embs, logit_scale, saved_img_embs, saved_txt_embs, contra_batch_size):
    ids = np.random.choice(len(saved_img_embs), contra_batch_size, replace=False)
    saved_img_embs, saved_txt_embs = saved_img_embs[ids], saved_txt_embs[ids]
    
    positives = th.einsum('bd,bd->b', img_embs, txt_embs).unsqueeze(-1) 
    negatives_img_txt = th.einsum('bd,sd->bs', img_embs, saved_txt_embs) 
    negatives_txt_img = th.einsum('bd,sd->bs', txt_embs, saved_img_embs) 
    ground_truth = img_embs.new_zeros(len(x))
    
    def loss(logits):
        logits = torch.cat([positives, logits], dim=-1)
        return F.cross_entropy(logit_scale * logits, ground_truth)
        
    return loss(negatives_img_text) + loss(negatives_txt_img) / 2