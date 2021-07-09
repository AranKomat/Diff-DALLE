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
from transformers import GPT2TokenizerFast
import torch.nn.functional as F
import torch as th

def _load_data(
    index_dir=None,
    data_dir=None,
    batch_size=1,
    image_size=64,
    deterministic=False,
    random_crop=False,
    random_flip=True,
    text_length=None,
    small_size=0,
    text_loader=False,
    text_aug_factor=1,
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

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.

    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir and not index_dir:
        raise ValueError("unspecified data directory")
    
    data_loader = TextDataset if text_loader else ImageTextDataset
    dataset = data_loader(
        image_size,
        index_dir=index_dir,
        data_dir=data_dir,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
        text_length=text_length,
        phase=phase,
        small_size=small_size,
        gaussian_blur=gaussian_blur,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True
        )
    while True:
        yield from loader


def _list_text_files_recursively(data_dir):
    results = []
    t = time()
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["txt"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_text_files_recursively(full_path))
    return results


class ImageTextDataset(Dataset):
    def __init__(
        self,
        resolution,
        index_dir=None,
        data_dir=None,
        shard=0,
        num_shards=1,
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
        if index_dir is not None:
            with open(index_dir, 'r') as f:
                indices = json.load(f)
            self.local_texts = indices[shard:][::num_shards]                  
        else:
            txts = _list_text_files_recursively(data_dir)
            self.local_texts = txts[shard:][::num_shards]  
        random.shuffle(self.local_texts)
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.text_length = text_length

    def __len__(self):
        return len(self.local_texts)

    def __getitem__(self, idx):
        path = self.local_texts[idx]
        out_dict = {}
        
        # load text
        with open(path) as f:
            text = f.read()
        text = self.tokenizer(text)["input_ids"] 
        text = text[:self.text_length-1] 
        text = text + [self.tokenizer.vocab_size - 1] * (self.text_length - len(text))
        out_dict["y"] = np.array(text, dtype=np.int32)
        
        # load image
        path = os.path.splitext(path)[0]
        path = path.replace('/texts/', '/images/')        
        for ext in [".jpg", ".jpeg", ".png", ".gif"]:
            cur_path = path + ext
            if os.path.exists(cur_path):
                path = cur_path
                break
        
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1        
        image = np.transpose(arr, [2, 0, 1])
        
        if self.small_size > 0:
            if self.phase == 'train' and self.gaussian_blur and random.uniform(0, 1) < 0.5:
                sigma = random.uniform(0.4, 0.6)
                noised_image = image.copy()
                out_dict["low_res"] = gaussian_filter(noised_image, [0, sigma, sigma], truncate=1.0)
            else:
                out_dict["low_res"] = image.copy()
                
        return image, out_dict

    
class TextDataset(ImageTextDataset):
    def __getitem__(self, idx):
        path = self.local_texts[idx]
        with open(path) as f:
            text = f.read()
        text = self.tokenizer(text)["input_ids"] 
        text = text[:self.text_length-1] 
        text = text + [self.tokenizer.vocab_size - 1] * (self.text_length - len(text))
        return np.array(text, dtype=np.int32)      

    
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
    text_aug_factor = kwargs["text_aug_factor"] if "text_aug_factor" in kwargs.keys() else 1
    if text_aug_factor > 1:
        batch_size = kwargs["batch_size"]
        aug_text_data = _load_data(**{**kwargs, **{"batch_size": batch_size * (text_aug_factor - 1), "text_loader": True, "num_workers": 64}})
    for large_batch, model_kwargs in data:
        if small_size > 0:
            model_kwargs["low_res"] = F.interpolate(model_kwargs["low_res"], small_size, mode="area")
        if text_aug_factor > 1:
            aug_text = next(aug_text_data)
            model_kwargs["y"] = th.cat([model_kwargs["y"], aug_text])
        yield large_batch, model_kwargs