# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# Taken from https://github.com/facebookresearch/splitnet

import gzip
import os
import habitat
import habitat.datasets.pointnav.pointnav_dataset as mp3d_dataset
import numpy as np
import tqdm
from habitat.config.default import get_config
from habitat.datasets import make_dataset
from dataset_generation.configs import vector_env

def _load_datasets(config_keys, dataset, data_path, scenes_path, num_workers):
  # For each scene, create a new dataset which is added with the config
  # to the vector environment.

  print(len(dataset.episodes))
  datasets = []
  configs = []
  # import pdb;pdb.set_trace()
  num_episodes_per_worker = len(dataset.episodes) / float(num_workers)

  for i in range(0, min(len(dataset.episodes), num_workers)):
    config = make_config(*config_keys)
    config.defrost()

    dataset_new = mp3d_dataset.PointNavDatasetV1()
    with gzip.open(data_path, "rt") as f:
      dataset_new.from_json(f.read())
      dataset_new.episodes = dataset_new.episodes[
          int(i * num_episodes_per_worker): int(
              (i + 1) * num_episodes_per_worker
          )
      ]

      # import pdb;pdb.set_trace()
      for episode_id in range(0, len(dataset_new.episodes)):
        if 'replica' in dataset_new.episodes[episode_id].scene_id:
          dataset_new.episodes[episode_id].scene_id = \
            dataset_new.episodes[episode_id].scene_id.replace(
              '/checkpoint/ow045820/data/replica/',
              scenes_path)

          # dataset.episodes[i].scene_id = dataset.episodes[i].scene_id.replace(
          # '/checkpoint/ow045820/data/replica/',
          # scenes_path)
        elif 'mp3d' in dataset.episodes[i].scene_id:
          dataset_new.episodes[episode_id].scene_id = \
            dataset_new.episodes[episode_id].scene_id.replace(
              '/checkpoint/erikwijmans/data/mp3d/',
              scenes_path)

    config.SIMULATOR.SCENE = str(dataset_new.episodes[0].scene_id)
    config.freeze()
    datasets += [dataset_new]
    configs += [config]
  return configs, datasets


def make_config(
    config, gpu_id, split, data_path, sensors, resolution, scenes_dir
):
  config = get_config(config)
  config.defrost()
  config.TASK.NAME = "Nav-v0"
  config.TASK.MEASUREMENTS = []
  config.DATASET.SPLIT = split
  # config.DATASET.POINTNAVV1.DATA_PATH = data_path
  config.DATASET.DATA_PATH = data_path
  config.DATASET.SCENES_DIR = scenes_dir
  config.HEIGHT = resolution
  config.WIDTH = resolution
  for sensor in sensors:
    config.SIMULATOR[sensor]["HEIGHT"] = resolution
    config.SIMULATOR[sensor]["WIDTH"] = resolution
    config.SIMULATOR[sensor]["POSITION"] = np.array([0, 0, 0])

  config.TASK.HEIGHT = resolution
  config.TASK.WIDTH = resolution
  config.SIMULATOR.TURN_ANGLE = 15
  config.SIMULATOR.FORWARD_STEP_SIZE = 0.1  # in metres
  config.SIMULATOR.AGENT_0.SENSORS = sensors
  config.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH = False

  # config.SIMULATOR.DEPTH_SENSOR.HFOV = 90
  config.SIMULATOR.DEPTH_SENSOR.HFOV = 90
  config.SIMULATOR.RGB_SENSOR.HFOV = 90

  config.ENVIRONMENT.MAX_EPISODE_STEPS = 2 ** 32
  config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = gpu_id
  return config


class RandomImageGenerator(object):
  def __init__(self, split, gpu_id, opts, vectorize=False, seed=0, num_parallel_envs=1):

    self.vectorize = vectorize

    print("gpu_id", gpu_id)
    resolution = opts.W
    if opts.use_semantics:
      sensors = ["RGB_SENSOR", "DEPTH_SENSOR", "SEMANTIC_SENSOR"]
    else:
      sensors = ["RGB_SENSOR", "DEPTH_SENSOR"]
    if split == "train":
      data_path = opts.train_data_path
    elif split == "val":
      data_path = opts.val_data_path
    elif split == "test":
      data_path = opts.test_data_path
    else:
      raise Exception("Invalid split")
    
    unique_dataset_name = opts.dataset

    self.num_parallel_envs = num_parallel_envs #1
    # self.use_rand = use_rand

    self.images_before_reset = opts.images_before_reset
    config = make_config(
      opts.config,
      gpu_id,
      split,
      data_path,
      sensors,
      resolution,
      opts.scenes_dir,
    )
    self.config=config

    data_dir = os.path.join(
      "data_readers/scene_episodes/", unique_dataset_name + "_" + split
    )
    self.dataset_name = config.DATASET.TYPE
    print(data_dir)
    if not os.path.exists(data_dir):
      os.makedirs(data_dir, exist_ok=True)
    data_path = os.path.join(data_dir, "dataset_one_ep_per_scene.json.gz")
    # Creates a dataset where each episode is a random spawn point in each scene.
    print("One ep per scene", flush=True)
    if not (os.path.exists(data_path)):
      print("Creating dataset...", flush=True)
      dataset = make_dataset(config.DATASET.TYPE, config=config.DATASET)

      # Get one episode per scene in dataset
      scene_episodes = {}
      for episode in tqdm.tqdm(dataset.episodes):
        if episode.scene_id not in scene_episodes:
          scene_episodes[episode.scene_id] = episode

      scene_episodes = list(scene_episodes.values())
      dataset.episodes = scene_episodes
      if not os.path.exists(data_path):
        # Multiproc do check again before write. 
        # import pdb;pdb.set_trace()
        json = dataset.to_json().encode("utf-8")
        with gzip.GzipFile(data_path, "w") as fout:
          fout.write(json)
      
      print("Finished dataset...", flush=True)

    # Load in data and update the location to the proper location (else
    # get a weird, uninformative, error -- Affine2Dtransform())
    dataset = mp3d_dataset.PointNavDatasetV1()
    with gzip.open(data_path, "rt") as f:
      dataset.from_json(f.read())

      for i in range(0, len(dataset.episodes)):
        # import pdb;pdb.set_trace()
        if 'replica' in dataset.episodes[i].scene_id:
          dataset.episodes[i].scene_id = dataset.episodes[i].scene_id.replace(
          '/checkpoint/ow045820/data/replica/',
          opts.scenes_dir + '/replica/')   
        elif 'mp3d' in dataset.episodes[i].scene_id:
          # import pdb;pdb.set_trace()
          dataset.episodes[i].scene_id = dataset.episodes[i].scene_id.replace(
            '/checkpoint/erikwijmans/data/mp3d/',
            opts.scenes_dir + '/mp3d/')
        elif 'hm3d' in dataset.episodes[i].scene_id:
          pass
        else:
          raise NotImplementedError
          


    config.TASK.SENSORS = ["POINTGOAL_SENSOR"]

    config.freeze()

    self.rng = np.random.RandomState(seed)
    # self.reference_idx = reference_idx

    # Now look at vector environments
    if self.vectorize:
      datadir = opts.scenes_dir + "/" + opts.dataset + "/"
      print("len(dataset.episodes):",len(dataset.episodes) )
      configs, datasets = _load_datasets(
        (
          opts.config,
          gpu_id,
          split,
          data_path,
          sensors,
          resolution,
          opts.scenes_dir,
        ),
        dataset,
        data_path,
        datadir,
        num_workers=self.num_parallel_envs,
      )
      # print("configs, datasets:", len(configs), len(datasets))
      # print("configs:", configs)
      # print("datasets:" , datasets)

      num_envs = len(configs)
      env_fn_args = tuple(zip(configs, datasets, range(num_envs)))
      envs = vector_env.VectorEnv(
        env_fn_args=env_fn_args,
        multiprocessing_start_method="forkserver",
      )

      self.env = envs
      self.dataset = datasets
      print("len(self.datasets):", len(datasets))
      # self.num_train_envs = int(0.9 * (self.num_parallel_envs))
      # self.num_val_envs = self.num_parallel_envs - self.num_train_envs
    else:
      raise NotImplementedError
      # self.env = habitat.Env(config=config, dataset=dataset)
      # self.env_sim = self.env.sim
      # self.rng.shuffle(self.env.episodes)
      # self.env_sim = self.env.sim

    # self.num_samples = 0

    # Set up intrinsic parameters
    # self.hfov = config.SIMULATOR.DEPTH_SENSOR.HFOV * np.pi / 180.0
    # self.W = resolution
    # self.K = np.array(
    #   [
    #     [1.0 / np.tan(self.hfov / 2.0), 0.0, 0.0, 0.0],
    #     [0, 1.0 / np.tan(self.hfov / 2.0), 0.0, 0.0],
    #     [0.0, 0.0, 1.0, 0.0],
    #     [0.0, 0.0, 0.0, 1.0],
    #   ],
    #   dtype=np.float32,
    # )

    # self.invK = np.linalg.inv(self.K)

    self.config = config
    self.opts = opts