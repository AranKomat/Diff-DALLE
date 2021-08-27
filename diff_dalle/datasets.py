import math
import random
import os
import json
from time import time
from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader, Dataset
from diff_dalle.simple_tokenizer import SimpleTokenizer
import torch.nn.functional as F
import torch as th
import webdataset as wds
from glob import glob
import io
import imageio

def tokenize(self, tokenizer, text, context_length):
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    tokens = tokenizer.encode(text)[:context_length - 2]
    all_tokens = [sot_token] + tokens + [eot_token]
    return all_tokens + [0] * (context_length - len(all_tokens))

def _load_data(
    index_dir=None,
    batch_size=1,
    image_size=64,
    deterministic=False,
    random_crop=False,
    random_flip=True,
    text_length=None,
    small_size=0,
    phase='train',
    gaussian_blur=False,
    num_workers=8,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param index_dir: a dataset index directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.

    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    
    shards = glob(f"{index_dir}/**/*.tar", recursive=True)
    
    dataset = (
        wds.WebDataset(shards, shardshuffle=True)
        .shuffle(100)
    )
    
    decoder = ImageTextDecoder(
        image_size,
        random_crop=random_crop,
        random_flip=random_flip,
        text_length=text_length,
        phase=phase,
        small_size=small_size,
        gaussian_blur=gaussian_blur,
    )
    
    dataset = wds.Processor(dataset, wds.map, decoder.decode)
    dataset = dataset.batched(batch_size, partial=False)
    
    loader = wds.WebLoader(
        dataset, batch_size=None, shuffle=False, num_workers=num_workers,
    )
        
    while True:
        yield from loader


class ImageTextDecoder:
    def __init__(
        self,
        resolution,
        random_crop=False,
        random_flip=True,
        text_length=None,
        phase='train',
        small_size=0,
        gaussian_blur=False,
    ):
        super().__init__()
        self.resolution = resolution
        self.phase = phase
        self.gaussian_blur = gaussian_blur
        self.small_size = small_size
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.text_length = text_length
        self.tokenizer = SimpleTokenizer()    
    
    def decode(self, sample):
        
        def check_ext(key, exts):
            for ext in exts:
                if key == ext or key.endswith(f".{ext}"):
                    return True
            return False
        
        out_dict = {}
        for key, value in sample.items():
            if check_ext(key, ["jpg", "jpeg", "png"]):
                with io.BytesIO(value) as f:
                    pil_image = Image.fromarray(imageio.imread(f, as_gray=False, pilmode="RGB"))

                if self.random_crop:
                    arr = random_crop_arr(pil_image, self.resolution)
                else:
                    arr = center_crop_arr(pil_image, self.resolution)

                if self.random_flip and random.random() < 0.5:
                    arr = arr[:, ::-1]

                arr = arr.astype(np.float32) / 127.5 - 1  
                #print(arr)
                image = np.transpose(arr, [2, 0, 1])

                if self.small_size > 0:
                    if self.phase == 'train' and self.gaussian_blur and random.uniform(0, 1) < 0.5:
                        sigma = random.uniform(0.4, 0.6)
                        noised_image = image.copy()
                        out_dict["low_res"] = gaussian_filter(noised_image, [0, sigma, sigma], truncate=1.0)
                    else:
                        out_dict["low_res"] = image.copy()

            elif check_ext(key, ["txt"]):
                with io.BytesIO(value) as f:
                    text = f.read().decode('UTF-8')
                text = self.tokenize(self.tokenizer, text, self.text_length)
                out_dict["y"] = np.array(text, dtype=np.int32)

        if "y" in out_dict:
            return image, out_dict["y"]            
        else:
            return image, out_dict["low_res"] 
    
def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def load_data(**kwargs):
    data = _load_data(**kwargs)
    small_size = kwargs["small_size"] if "small_size" in kwargs.keys() else 0
    if small_size > 0:
        for large_batch, low_res in data:
            model_kwargs = dict(low_res=F.interpolate(low_res, small_size, mode="area"))
            yield large_batch, model_kwargs
    else
        return data