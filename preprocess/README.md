
# Enviroment Setup
```
conda create -n dataset360 python=3.7
conda activate dataset360
```

Note: you must have EGL in your machine to run this code as is required by AI-habitat.
check by [egl-example](https://github.com/erwincoumans/egl_example).
```
# install habitat=0.2.2
pip install git+https://github.com/facebookresearch/habitat-lab.git@v0.2.2

# install habitat-sim=0.2.2
conda install habitat-sim=0.2.2 headless -c conda-forge -c aihabitat -y

or install habitat-sim=0.2.2 from local file
# download `habitat-sim-0.2.2-py3.7_headless_bullet_linux_011191f65f37587f5a5452a93d840b5684593a00.tar.bz2` from `https://anaconda.org/aihabitat/habitat-sim/files`
conda install --use-local habitat-sim-0.2.2-py3.7_headless_bullet_linux_011191f65f37587f5a5452a93d840b5684593a00.tar.bz2

```
Other problems refer to [PanoGRF](https://github.com/thucz/PanoGRF/blob/main/docs/install.md)


# Prepare Dataset

Acquire the access from official website [HM3D](https://matterport.com/partners/meta)


And after you get the access, you can
download HM3Dv0.1

```
1. Download these GLB+habitat files:
hm3d-train-glb.tar
hm3d-train-habitat.tar

hm3d-val-glb.tar
hm3d-val-habitat.tar

2. Download the pointnav files:
wget https://dl.fbaipublicfiles.com/habitat/data/datasets/pointnav/hm3d/v1/pointnav_hm3d_v1.zip

3. Unzip them
```
Our folder structure is like this:
```
# HM3D: OBJ+Habitat
ROOT/dataset/hm3d/train
ROOT/dataset/hm3d/val

# HM3D: Pointnav
ROOT/pointnav/hm3d/train
ROOT/pointnav/hm3d/val

# Replica: OBJ+Habitat
ROOT/replica/train
ROOT/replica/val

# Replica has no Pointnav
The episodes are preprocessed into `dataset_one_ep_per_scene.json.gz` in data_readers/scene_episodes/replica_test/ by [SynSin](https://github.com/facebookresearch/synsin).
```


# Revise directories of data
revise `base_dir` to your paths of `dataset_generation.configs.options.py` respectively for HM3D and Replica.

# Generate dataset for HM3D:
```
# revise args, e.g., root_dir to save the dataset.
bash generate_hm3d_train.sh
```

```
# change environment
conda activate splat360

# remember to set basedir and dataset_name
python convert_cubemaps_mp.py

# remember to set basedir, dataset_name and output_basedir
python convert.py

```

# Note: The code generate the random trajectories, which is not the same as our generated dataset.
