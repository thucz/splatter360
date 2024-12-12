# splatter360

<!-- Splatter-360: Generalizable 360$^{\circ}$ Gaussian Splatting for Wide-baseline Panoramic Images -->
Official implementation of **Splatter-360: Generalizable 360 Gaussian Splatting for Wide-baseline Panoramic Images**

<!-- Authors: [Yuedong Chen](https://donydchen.github.io/), [Haofei Xu](https://haofeixu.github.io/), [Chuanxia Zheng](https://chuanxiaz.com/), [Bohan Zhuang](https://bohanzhuang.github.io/), [Marc Pollefeys](https://people.inf.ethz.ch/marc.pollefeys/), [Andreas Geiger](https://www.cvlibs.net/), [Tat-Jen Cham](https://personal.ntu.edu.sg/astjcham/) and [Jianfei Cai](https://jianfei-cai.github.io/). -->

### [Project Page](https://3d-aigc.github.io/Splatter-360/) | [arXiv]() | [Pretrained Models](https://drive.google.com/file/d/1v3JVll12F9ReQ71bWLnz_ca9Xd2wEnhD/view?usp=drive_link) 

<!-- https://github.com/donydchen/mvsplat/assets/5866866/c5dc5de1-819e-462f-85a2-815e239d8ff2 -->

## Installation

To get started, create a conda virtual environment using Python 3.10+ and install the requirements:

```bash
conda create -n splat360 python=3.10
conda activate splat360
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Acquiring Datasets

Replica: Download `replica_dataset.zip` (rgb and depth files) and `replica_dataset_pt.zip` (scene indices) from [BaiduNetDisk](https://pan.baidu.com/s/1_GWfDn3XfNffZNvXoJUZ9Q?pwd=bair) or [OneDrive](https://1drv.ms/f/c/3e01a23b343bc186/EpTo7XbTMNhAsJNAu7pxsS8BPtHjq8v0prpc6aXN6Hid4g?e=6W62sq) and unzip them in the same directory. Revise `dataset.roots` and `dataset.rgb_roots` respectively in `config/experiment/replica.yaml` according to your storage directory.

HM3D: Too large, we are uploading HM3D these days.



<!-- ### RealEstate10K and ACID

Our MVSplat uses the same training datasets as pixelSplat. Below we quote pixelSplat's [detailed instructions](https://github.com/dcharatan/pixelsplat?tab=readme-ov-file#acquiring-datasets) on getting datasets.

> pixelSplat was trained using versions of the RealEstate10k and ACID datasets that were split into ~100 MB chunks for use on server cluster file systems. Small subsets of the Real Estate 10k and ACID datasets in this format can be found [here](https://drive.google.com/drive/folders/1joiezNCyQK2BvWMnfwHJpm2V77c7iYGe?usp=sharing). To use them, simply unzip them into a newly created `datasets` folder in the project root directory.

> If you would like to convert downloaded versions of the Real Estate 10k and ACID datasets to our format, you can use the [scripts here](https://github.com/dcharatan/real_estate_10k_tools). Reach out to us (pixelSplat) if you want the full versions of our processed datasets, which are about 500 GB and 160 GB for Real Estate 10k and ACID respectively.

### DTU (For Testing Only)

* Download the preprocessed DTU data [dtu_training.rar](https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view).
* Convert DTU to chunks by running `python src/scripts/convert_dtu.py --input_dir PATH_TO_DTU --output_dir datasets/dtu`
* [Optional] Generate the evaluation index by running `python src/scripts/generate_dtu_evaluation_index.py --n_contexts=N`, where N is the number of context views. (For N=2 and N=3, we have already provided our tested version under `/assets`.)
 -->
## Running the Code

### Evaluation

To render novel views and compute evaluation metrics from a pretrained model,

* get the [pretrained models](https://drive.google.com/file/d/1v3JVll12F9ReQ71bWLnz_ca9Xd2wEnhD/view?usp=drive_link), and save them to `/checkpoints`

* run the following:

```
# eval on HM3D
output_dir="./outputs/splat360_log_depth_near0.1-100k/"
checkpoint_path="./checkpoints/hm3d.ckpt"
CUDA_VISIBLE_DEVICES=0 python -m src.main \
    +experiment=hm3d \
    model.encoder.shim_patch_size=8\
    model.encoder.downscale_factor=8\
    model.encoder.depth_sampling_type="log_depth" \
    output_dir=$output_dir \
    dataset.near=0.1 \
    mode="test" \
    dataset/view_sampler=evaluation \
    checkpointing.load=$checkpoint_path \
    dataset.view_sampler.index_path="assets/evaluation_index_hm3d.json"\
    test.eval_depth=true

```

* the rendered novel views will be stored under `outputs/test`

To render videos from a pretrained model, run the following

```bash
# HM3D render video
output_dir="./outputs/splat360_log_depth_near0.1-100k/"
checkpoint_path="./checkpoints/hm3d.ckpt"
CUDA_VISIBLE_DEVICES=0 python -m src.main \
    +experiment=hm3d \
    model.encoder.shim_patch_size=8 \
    model.encoder.downscale_factor=8 \
    model.encoder.depth_sampling_type="log_depth" \
    output_dir=$output_dir \
    dataset.near=0.1 \
    mode="test" \
    dataset/view_sampler=evaluation\ 
    checkpointing.load=$checkpoint_path\
    dataset.view_sampler.index_path="assets/evaluation_index_hm3d_video.json" \
    test.save_video=true \
    test.save_image=false \
    test.compute_scores=false \
    test.eval_depth=true
```
### Training

```


# download the backbone pretrained weight from unimath and save to 'checkpoints/'
wget 'https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmdepth-scale1-resumeflowthings-scannet-5d9d7964.pth' -P checkpoints

# download the pretrained weight of depth-anything and save to 'pretrained/'
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth -P checkpoints


# Our models are trained with 8 V100 (32GB) GPU.
max_steps=100000
output_dir="./outputs/splat360_log_depth_near0.1-100k/"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m src.main \
     +experiment=hm3d data_loader.train.batch_size=1 \
     model.encoder.shim_patch_size=8 \
     model.encoder.downscale_factor=8 \
     trainer.max_steps=$max_steps \
     model.encoder.depth_sampling_type="log_depth" \
     output_dir=$output_dir \
     dataset.near=0.1
```
<!-- 
### Training

Run the following:

```bash
# download the backbone pretrained weight from unimath and save to 'checkpoints/'
wget 'https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmdepth-scale1-resumeflowthings-scannet-5d9d7964.pth' -P checkpoints
# train mvsplat
python -m src.main +experiment=re10k data_loader.train.batch_size=14
```

Our models are trained with a single A100 (80GB) GPU. They can also be trained on multiple GPUs with smaller RAM by setting a smaller `data_loader.train.batch_size` per GPU. -->
<!-- 
### Ablations

We also provide a collection of our [ablation models](https://drive.google.com/drive/folders/14_E_5R6ojOWnLSrSVLVEMHnTiKsfddjU) (under folder 'ablations'). To evaluate them, *e.g.*, the 'base' model, run the following command

```bash
# Table 3: base
python -m src.main +experiment=re10k \
checkpointing.load=checkpoints/ablations/re10k_worefine.ckpt \
mode=test \
dataset/view_sampler=evaluation \
test.compute_scores=true \
wandb.name=abl/re10k_base \
model.encoder.wo_depth_refine=true 
``` -->

### Cross-Dataset Generalization

We use the default model trained on HM3D to conduct cross-dataset evalutions. To evaluate them, *e.g.*, on Replica, run the following command

```bash
output_dir="./outputs/splat360_log_depth_near0.1-100k/"
# eval on Replica
checkpoint_path="./checkpoints/hm3d.ckpt"
CUDA_VISIBLE_DEVICES=0 python -m src.main \
    +experiment=replica \
    model.encoder.shim_patch_size=8 \
    model.encoder.downscale_factor=8 \
    model.encoder.depth_sampling_type="log_depth" \
    output_dir=$output_dir \
    dataset.near=0.1 \
    mode="test" \
    dataset/view_sampler=evaluation \
    checkpointing.load=$checkpoint_path \
    dataset.view_sampler.index_path="assets/evaluation_index_replica.json"\
    test.eval_depth=true
```

<!-- **More running commands can be found at [more_commands.sh](more_commands.sh).** -->
<!-- 
## BibTeX

```bibtex
@article{chen2024mvsplat,
    title   = {MVSplat: Efficient 3D Gaussian Splatting from Sparse Multi-View Images},
    author  = {Chen, Yuedong and Xu, Haofei and Zheng, Chuanxia and Zhuang, Bohan and Pollefeys, Marc and Geiger, Andreas and Cham, Tat-Jen and Cai, Jianfei},
    journal = {arXiv preprint arXiv:2403.14627},
    year    = {2024},
}
``` -->

## Acknowledgements

The project is largely based on [pixelSplat](https://github.com/dcharatan/pixelsplat), [MVSplat](https://github.com/donydchen/mvsplat), and [PanoGRF](https://github.com/thucz/PanoGRF) and has incorporated numerous code snippets from [UniMatch](https://github.com/autonomousvision/unimatch). Many thanks to these projects for their excellent contributions!
