# Diff-DALLE (WIP)

Diff-DALLE is a DDPM for text-to-image generation: 
-  This model consists of: 
	- Transformer encoder for taking in the input text
	- DDPM U-Net for generating an image conditioned on the input text
		- Token-level input embeddings are fed from the encoder to the U-Net via encoder-decoder attention.
		- Sequence-level input embedding is fed as the class embedding of DDPM.  
	- CLIP classifier for guiding the sampling.
- We plan to train with large amount of computes and dataset to release a pretrained model.
- We also plan to investigate how well unconditional image generation is improved by generating alt-texts/captions with a fine-tuned GPT-2/3 and then generating images with Diff-DALLE. 

We have designed Diff-DALLE by starting from [Guided Diffusion](https://github.com/openai/guided-diffusion/) by Dhariwal et. al. Hence, its design, hyperparameter choice and code are heavily inspired by this.  

#### Major steps to follow
1.  Installation
2. Data preparation
3. Training generator (encoder + U-Net)
4. Training classifier
5. Sampling

## Acknowledgment 
Thanks to everyone who have helped out one way or another (listed alphabetically):
* [EleutherAI](https://www.eleuther.ai/) for providing their computational resources for initial experiments
* [Shivanshu Purohit](https://github.com/ShivanshuPurohit) for securing and setting up the computational resources used for this project 
* [Spell](https://spell.ml/) for offering us a generous grant of $40,000 from [Spell Open Research Grant](https://spell.ml/blog/spell-open-research-grant-YIMtSxEAACQAKm81), which was used for training our model 

## Installation

Clone this repository and navigate to it in your terminal. Then run:

```bash
pip install -e .
```

This should install the `diff_dalle` python package that the scripts depend on.

##### Caveat
* If your accelerator is Ampere (e.g. A100), you should run this model on [NGC](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch) to avoid substantial slowdown on mixed-precision convolutions due to a bug of PyTorch (<=1.9).

## Preparing data (WIP)

For each dataset, the dataset directory `/dataset` (or any directory name of your choice) should consist of two subdirectories `dataset/images` and `dataset/texts`, where each image-text pair should have the same filename and (relative) path in each subdirectory, (e.g. `dataset/images/01/0000001.jpg` and `dataset/texts/01/0000001.jpg`). 

For creating your own dataset, simply dump all of your images into a directory with ".jpg", ".jpeg", or ".png" extensions. Subdirectories will automatically be enumerated as well, so the images can be organized into a recursive structure (although the directory names will be ignored, and the underscore prefixes are used as names). Likewise for text files, but only ".txt" extension is accepted, and make sure that they have the same relative path as the paired image. 

The images will automatically be scaled and center-cropped by the data-loading pipeline. Simply pass `--data_dir path/to/dataset` to the script, and it will take care of the rest. For larger datasets, it is recommended to set `-- index_dir path/to/index` instead of `data_dir`, which directs to the json index file that stores all the paths of text files, since search for all the text files before each run becomes more time-consuming. An index file can be produced by running the following: 
```bash
python make_index.py --data_path path/to/dataset --img_min_sizes 64,512 \
--index_path_save index.json
``` 
This takes several minutes per 1M files. 

For training a medium-sized model, you can download Conceptual Captions 12M dataset from
* [https://github.com/robvanvolt/DALLE-datasets/blob/main/general/cc12m.py](https://github.com/robvanvolt/DALLE-datasets/blob/main/general/cc12m.py)

Beware that this costs 1TB of storage and more than a day of processing time. PR of adding a smaller dataset for demonstration is encouraged.

## Training

### General remarks
* In the case of using perceptual loss, you need to first train a classifier, after which you can start training a generator. This feature is optional.
* To generate with image resolution larger than 128, cascading training should be used. 
	* For example, for 256 x 256  images, you need to train 64 x 64 generator + classifier as well as 256 x 256 upsampling generator (optionally + classifier).   
	* While a model with a higher-resolution hierarchy costs more FLOPS per image, you can mitigate the increased cost without substantial performance degradation by doing the following:
		* Use a smaller model  (e.g. `num_channels` = 256 for 64 x 64, while = 128 for 256 x 256)
		* Reduce `batch_size` (e.g. `batch_size = 2048` for 64 x 64, while = 256 for 256 x 256)
	* This way, you can aim for spending about the same amount of computes on each hierarchy. 
	* It is often more efficient to make the number of sampling steps larger for low-res hierarchy than high-res hierarchy.

To train your model, you should first decide some hyperparameters. We will split up our hyperparameters into three groups: model architecture, diffusion process (for generator), and training flags. Here is an example:

### Logs & checkpoints

The logs and checkpoints will be written to a logging directory determined by the `OPENAI_LOGDIR` environment variable (e.g. `export OPENAI_LOGDIR="/path/to/logdir`). If it is not set, then a temporary directory will be created in `/tmp`.

The training scripts below save checkpoints to `.pt` files in the logging directory. These checkpoints will have names like `ema_0.9999_200000.pt` and `model200000.pt`. You will likely want to sample from the EMA models, since those produce much better samples.

### Generator
```bash
MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 2 --enc_attn_dim 512 \
--dropout 0.1 --use_fp16 True"
DIFFUSION_FLAGS="--noise_schedule cosine"
TRAIN_FLAGS="--lr 3e-4 --batch_size 512 --microbatch 64"
```
Once you have setup your hyper-parameters, you can run an experiment like so:

```bash
python scripts/train_genrator.py --data_dir path/to/images $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
```
##### Perceptual loss (not recommended)
To add perceptual loss, you can specify `--clip_coeff` and `--clip_resume_checkpoint` for `TRAIN_FLAGS` and add the `CLASSIFIER_FLAGS` for the corresponding classifier. While this idea makes sense, we have not trained with this option stably yet. 

##### Cascaded training
* For generator, you also need to modify `small_size` to the size of the input image. 
* Setting `--gaussian_blur True` should be beneficial.

### Classifier
```bash
CLASSIFIER_FLAGS="--image_size 64 --classifier_depth 2 --classifier_width 128 \
--classifier_enc_attn_dim 512 --use_fp16 True"
DIFFUSION_FLAGS="--noise_schedule cosine"
TRAIN_FLAGS="--iterations 300000 --batch_size 128 --lr 3e-4 --weight_decay 0.1 \
--text_aug_factor 16"
```
As in generator training, you can run an experiment for classifier as:

```bash
python scripts/train_classifier.py --data_dir path/to/train_data \
--val_data_dir path/to/val_data $CLASSIFIER_FLAGS $TRAIN_FLAGS
```
##### Cascaded training
* For this, it suffices to modify `image_size`, model size, `batch_size`, etc. and run with the exact same command.

##### Recommendations
* Due to the use of contrastive loss, using a larger batch size is more effective. Since FLOPS per text is much cheaper than FLOPS per image, we tried increasing `text_aug_factor` (= text batch size / image batch size) above 1 by sampling random texts as negatives. Note that gradient accumulation does not equate to using a larger batch size under contrastive loss.
* Unlike the generator, overfitting is more likely. Hence, it is recommended to take the checkpoint with the best validation loss.

### Distributed training

You may also want to train in a distributed manner. In this case, run the same command with `mpiexec`:

```bash
mpiexec -n $NUM_GPUS python scripts/train_generator.py --data_dir path/to/images \
$MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
```
Classifier training can be run likewise.

## Sampling

Once you have a path to your trained model, you can generate a large batch of samples like so:
```bash
MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 2 --use_fp16 True"
DIFFUSION_FLAGS="--noise_schedule cosine --timestep_respacing 250"
CLASSIFIER_FLAGS="--classifier_depth 2 --classifier_width 128 --classifier_enc_attn_dim 512 \
--classifier_scale 1.0"
```
```bash
python scripts/sample.py --model_path /path/to/model_ema.pt \
--classifier_path /path/to/classifier.pt --data_dir path/to/texts $MODEL_FLAGS \
$CLASSIFIER_FLAGS $DIFFUSION_FLAGS
```
You can remove the relevant parts if classifier-augmented sampling is not needed.

Again, this will save results to a logging directory. Samples are saved as a large `npz` file. A small subset of the samples is also saved as a grid image in jpg and a txt file. 

Just like for training, you can run `sample.py` through MPI to use multiple GPUs and machines.

You can change the number of sampling steps using the `--timestep_respacing` argument. For example, `--timestep_respacing 250` uses 250 steps to sample. Passing `--timestep_respacing ddim25` is similar, but uses the uniform stride from the [DDIM paper](https://arxiv.org/abs/2010.02502).

To sample using [DDIM](https://arxiv.org/abs/2010.02502), pass `--use_ddim True`.

## Major hyperparameters

* `image_size`: the resolution of the output image.
* `num_channels`: the number of channels of the outermost layer of U-Net. 
* `enc_attn_dim`: the number of channels of the Transformer encoder.
* `num_res_blocks`: the number of layers for each resolution of U-Net.
* `dropout`: recommended to set this to 0.1 for 64 x 64 and 0 otherwise.
* `noise_schedule`: `cosine` is recommended for `image_size` = 64, and `linear` is recommended otherwise.
*  `lr`: the base learning rate. The rate is constant for generator and cosine annealed for classifier with linear warmup. 
* `batch_size`: batch size per core.
* `microbatch`: if set, gradient accumulation is performed with each microbatch size = `microbatch`. Setting this is not recommended for classifier training.
* `resume_checkpoint`:  path to model parameter checkpoint to resume training (e.g. `--resume_checkpoints path/to/log_dir/model010000_ema.pt`)
* `text_length`: the length of input text (default 48). The texts longer or shorter than `text_length` are curtailed or padded to this length.  

## Models and Hyperparameters (WIP)

For model checkpoints (if available) and run flags we have attempted, please refer to [models_hparams.md]() (not ready yet).

## FAQ
* How can we scale up the model with this repo?
	* Model parallelism: 
		* While our repo does not allow model parallelism yet, with cascading training and classifier training, we have multiple models that we can train separately in parallel. This allows a ~4B model without model parallelism on A100s.
	* Data parallelism:
		* Since critical batch size of DDPM seems to be rather large, we can aggressively utilize the data parallelism (e.g. maybe up to `batch_size = 2048` for 64 x 64 generator). 
	* Dataset size:
		* Given that even base model requires several hundreds of millions of images (counting multiplicity) for nearly compute-optimal training, and that typical dataset size used for DDPM variants is no more less than 2M, using a dataset of the order of 100M images should improve the performance substantially.
	
## TODO
- [ ] Finish core components of Diff-DALLE
	- [x] Test generator training without classifier
	- [x] Test classifier training 
	- [x] Test sampling without classifier
	- [ ] Test sampling with classifier (no bug; performance not checked)
- [ ] Add the code for preparing a small dataset for demo
- [ ] Perform large-scale training
- [ ] Add evaluation metrics (e.g. Precision & Recall, FID, etc) and perform ablations
- [ ] Improve the documentation
- [ ] Release a pretrained model, colab notebook and web demo 
