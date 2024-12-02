import json
from dataclasses import dataclass
from functools import cached_property
from io import BytesIO
from pathlib import Path
from typing import Literal
import torch
import torchvision.transforms as tf
from einops import rearrange, repeat
from jaxtyping import Float, UInt8, Int
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset
from ..geometry.projection import get_fov
from .dataset import DatasetCfgCommon
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim
from .types import Stage
from .view_sampler import ViewSampler
import torch.distributed as dist
import cv2
import numpy as np
from ..geometry.util import Equirec2Cube
@dataclass
class DatasetHM3DCfg(DatasetCfgCommon):
    name: Literal["replica", "hm3d"]
    roots: list[Path]
    rgb_roots: list[Path]
    baseline_epsilon: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    test_len: int
    test_chunk_interval: int
    test_times_per_scene: int
    skip_bad_shape: bool = True
    near: float = -1.0
    far: float = -1.0
    baseline_scale_bounds: bool = True
    shuffle_val: bool = True
    debug: bool = False

class DatasetHM3D(IterableDataset):
    cfg: DatasetHM3DCfg
    stage: Stage
    view_sampler: ViewSampler
    to_tensor: tf.ToTensor
    chunks: list[Path]
    near: float = 0.5
    far: float = 1000.0
    def __init__(
        self,
        cfg: DatasetHM3DCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()

        self.cfg = cfg
        self.debug = cfg.debug
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()


        self.h, self.w = cfg.image_shape  
        self.e2c = Equirec2Cube(self.h, self.w, self.h // 2)

        # NOTE: update near & far; remember to DISABLE `apply_bounds_shim` in encoder
        if cfg.near != -1:
            self.near = cfg.near
        if cfg.far != -1:
            self.far = cfg.far
        # Collect chunks.
        self.chunks = []
        self.rgb_dirs = {}
        # TODO: cfg.rgb_roots
        assert len(cfg.rgb_roots) == len(cfg.roots)
        for root_idx, root in enumerate(cfg.roots):
            rgb_dir = cfg.rgb_roots[root_idx] / self.data_stage 
            root = root / self.data_stage           
            root_chunks = sorted(
                [path for path in root.iterdir() if path.suffix == ".torch"]
            )
            self.chunks.extend(root_chunks)
            self.rgb_dirs[str(root)] = rgb_dir
            
        if self.cfg.overfit_to_scene is not None:
            chunk_path = self.index[self.cfg.overfit_to_scene]
            self.chunks = [chunk_path] * len(self.chunks)

        if self.stage == "test":
            # NOTE: hack to skip some chunks in testing during training, but the index
            # is not change, this should not cause any problem except for the display
            self.chunks = self.chunks[:: cfg.test_chunk_interval]

    def shuffle(self, lst: list) -> list:
        indices = torch.randperm(len(lst))
        return [lst[x] for x in indices]

    def __iter__(self):
        # Chunks must be shuffled here (not inside __init__) for validation to show
        # random chunks.
        if self.stage in (("train", "val") if self.cfg.shuffle_val else ("train")):
            self.chunks = self.shuffle(self.chunks)       

        # When testing, the data loaders alternate chunks.
        worker_info = torch.utils.data.get_worker_info()
        if self.stage == "test" and worker_info is not None:
            self.chunks = [
                chunk
                for chunk_index, chunk in enumerate(self.chunks)
                if chunk_index % worker_info.num_workers == worker_info.id
            ]

        cubes = 6
        for chunk_path in self.chunks:
            chunk = torch.load(chunk_path)
            rgb_root = self.rgb_dirs[str(chunk_path.parent)]
            if self.cfg.overfit_to_scene is not None:
                item = [x for x in chunk if x["key"] == self.cfg.overfit_to_scene]
                assert len(item) == 1
                chunk = item * len(chunk)

            if self.stage in (("train", "val") if self.cfg.shuffle_val else ("train")):
                chunk = self.shuffle(chunk)


            times_per_scene = self.cfg.test_times_per_scene
            
            for run_idx in range(int(times_per_scene * len(chunk))):                
                example = chunk[run_idx // times_per_scene]   
                cube_shape = example["cube_shape"]
                extrinsics_sphere, extrinsics, intrinsics = self.convert_poses(example["cameras"], example["c2ws_cubes"], example["fxfycxcys"], cube_shape)
                # evaluation
                
                if times_per_scene > 1:  # specifically for DTU
                    scene = f"{example['key']}_{(run_idx % times_per_scene):02d}"
                else:
                    scene = example["key"]
                # try:
                if hasattr(self.view_sampler, "index"):
                    entry = self.view_sampler.index.get(scene)
                    if entry is None:
                        print(f"No indices available for scene {scene}... skip!")
                        continue;

                try:
                    context_indices, target_indices = self.view_sampler.sample(
                        scene,
                        # extrinsics_sphere=extrinsics_sphere,
                        extrinsics=extrinsics,
                        intrinsics=intrinsics,
                    )

                # clip_length = 100
                # if len(target_indices) > clip_length and self.view_sampler.cfg.name == "all":
                #     target_indices = target_indices[:clip_length] # clip length
                #     print("context_indices too long, might cause OOM! clip video")
                    
                except ValueError:
                    # Skip because the example doesn't have enough frames.
                    print("Skip because the example doesn't have enough frames.")
                    continue

                # Skip the example if the field of view is too wide.
                # if (get_fov(intrinsics).rad2deg() > self.cfg.max_fov).any():
                #     continue
                
                # panorama rgbs
                rgb_dir = rgb_root / scene / "pano" #/ rgbd_name

                rgb_files = sorted(
                    # [path for path in rgb_dir.iterdir() if path.suffix == ".torch"]
                    [path for path in rgb_dir.iterdir() if path.suffix == ".png"]
                )

                debug = True
                if debug:
                    cnt = extrinsics.shape[0]
                    rgb_files = rgb_files[:cnt]
                
                context_images = []
                context_depths = []

                def load_frame_data(files, indices):
                    pano_images = []
                    pano_depths = []
                    cube_images_input = []
                    cube_images_supervise = []
                    cube_depth_frames = []

                    for index in indices:
                        file_path = files[index]
                        pano_frame = cv2.imread(str(file_path))
                        pano_frame = cv2.cvtColor(pano_frame, cv2.COLOR_BGR2RGB)
                        cube_depth_frame = torch.load(str(file_path).replace("pano", "cubemaps_depth").replace(".png", ".torch"))
                        # 6 h w 1, original metric, no need to scale.

                        cube_frame = self.e2c.run(pano_frame)
                        cube_frame = torch.from_numpy(cube_frame) # h 6*w c
                        cube_frame = rearrange(cube_frame, "h (cubes w) c -> cubes h w c", cubes=cubes)

                        # reordering:
                        # [F R B L U D] <- [U B L F R D]                        
                        cube_frame_new = cube_frame.clone()
                        cube_frame_new[0] = cube_frame[4]
                        cube_frame_new[1:3] = cube_frame[2:4]
                        cube_frame_new[3:5] = cube_frame[0:2]
                        
                        # flip along h, w
                        cube_frame_new[0, :, :, :] = torch.flip(cube_frame_new[0, :, :, :], dims=[0, 1])
                        cube_frame_new[5, :, :, :] = torch.flip(cube_frame_new[5, :, :, :], dims=[0, 1])

                        cube_frame = cube_frame * 1.0 / 255
                        cube_frame_new = cube_frame_new * 1.0 / 255

                        pano_frame = pano_frame * 1.0 / 255.0
                        pano_frame = torch.from_numpy(pano_frame).to(torch.float32)
                        pano_frame = rearrange(pano_frame, "h w c -> () c h w")

                        # pano_depth
                        pano_depth_path = str(file_path).replace("pano", "pano_depth")
                        pano_depth = cv2.imread(str(pano_depth_path), cv2.IMREAD_UNCHANGED)
                        pano_depth = pano_depth.astype(np.float32) * 1.0 / 1000.0 # meters
                        pano_depth = torch.from_numpy(pano_depth).to(torch.float32) # h w

                        pano_images.append(pano_frame)
                        pano_depths.append(pano_depth)
                        cube_images_input.append(cube_frame)
                        cube_images_supervise.append(cube_frame_new)

                        cube_depth_frames.append(cube_depth_frame)

                    cube_depth_frames = torch.stack(cube_depth_frames, dim=0) # b h w
                    pano_images = torch.cat(pano_images, dim=0)
                    pano_depths = torch.stack(pano_depths, dim=0).unsqueeze(1) # n 1 h w
                    cube_images_input = torch.stack(cube_images_input)
                    cube_images_input = rearrange(cube_images_input, "b n h w c -> b n c h w")

                    cube_images_supervise = torch.stack(cube_images_supervise)
                    cube_images_supervise = rearrange(cube_images_supervise, "b n h w c -> b n c h w")
                    return pano_images, cube_images_input, cube_images_supervise, pano_depths, cube_depth_frames
                
                if self.view_sampler.cfg.name == "all":
                    # save time
                    # only for sample view index in getting evaluation index.
                    context_pano_images = torch.zeros((1, ))
                    context_cube_images_input = torch.zeros((1, ))
                    context_cube_images_supervise = torch.zeros((1, ))
                    context_pano_depths = torch.zeros((1, ))
                    target_pano_images = torch.zeros((1, ))
                    target_cube_images_input = torch.zeros((1, ))
                    target_cube_images_supervise = torch.zeros((1, ))
                    target_pano_depths = torch.zeros((1, ))
                    # target_cube_depths = torch.zeros((1, ))
                    context_cube_depth_frames = torch.zeros((1,))
                    target_cube_depth_frames = torch.zeros((1,))
                else:
                    if max(context_indices) >= len(rgb_files):
                        print(f"Skipping {scene} due to invalid frame index!")
                        continue
                    context_pano_images, context_cube_images_input, context_cube_images_supervise, context_pano_depths, context_cube_depth_frames = load_frame_data(rgb_files, context_indices)
                    target_pano_images, target_cube_images_input, target_cube_images_supervise, target_pano_depths, target_cube_depth_frames = load_frame_data(rgb_files, target_indices)

                context_extrinsics_sphere = extrinsics_sphere[context_indices]
                # print("extrinsics_sphere:", extrinsics_sphere[:, :3, 3]) # v 4 4
                if context_extrinsics_sphere.shape[0] == 2:
                    a, b = context_extrinsics_sphere[:, :3, 3]
                    scale = (a - b).norm()
                    # context_indices: {context_indices}, 
                    if scale < self.cfg.baseline_epsilon:
                        print(
                            f"Scale: {scale:.6f}, Skipped {scene} because of insufficient baseline, less than {self.cfg.baseline_epsilon}"
                        )
                        # print("context_extrinsics_sphere:", context_extrinsics_sphere[:, :3, 3])
                        continue
                nf_scale = 1 

                example = {
                    "context": {
                        "extrinsics_sphere": extrinsics_sphere[context_indices],
                        "extrinsics_cubes": extrinsics[context_indices],
                        "intrinsics_cubes": intrinsics[context_indices],
                        "image_sphere": context_pano_images,
                        "image_cubes_input": context_cube_images_input,
                        "image_cubes_supervise": context_cube_images_supervise,
                        "depth_cubes": context_cube_depth_frames,
                        "depth_sphere": context_pano_depths,
                        # "depth_cubes": context_perspec_depths,
                        "near": self.get_bound("near", len(context_indices)) / nf_scale,
                        "far": self.get_bound("far", len(context_indices)) / nf_scale,
                        "near_cubes": repeat(self.get_bound("near", len(context_indices)) / nf_scale, "v -> v c", c=6),
                        "far_cubes": repeat(self.get_bound("far", len(context_indices)) / nf_scale, "v -> v c", c=6),
                        "index": context_indices,
                    },
                    "target": {
                        "extrinsics_sphere": extrinsics_sphere[target_indices],
                        "extrinsics_cubes": extrinsics[target_indices],
                        "intrinsics_cubes": intrinsics[target_indices],
                        "image_sphere": target_pano_images,
                        "image_cubes_input": target_cube_images_input,
                        "image_cubes_supervise": target_cube_images_supervise,
                        "depth_cubes": target_cube_depth_frames,
                        "depth_sphere": target_pano_depths,
                        # "depth_cubes": target_perspec_depths,
                        "near": self.get_bound("near", len(target_indices)) / nf_scale,
                        "far": self.get_bound("far", len(target_indices)) / nf_scale,
                        "near_cubes": repeat(self.get_bound("near", len(target_indices)) / nf_scale, "v -> v c", c=6),
                        "far_cubes": repeat(self.get_bound("far", len(target_indices)) / nf_scale, "v -> v c", c=6),
                        "index": target_indices,
                    },
                    "scene": scene,
                }
                
                
                # we don't use reflect augmentation in 360 dataset
                # if self.stage == "train" and self.cfg.augment:
                #     example = apply_augmentation_shim(example)
                # we don't use crop in 360 dataset
                # yield apply_crop_shim(example, tuple(self.cfg.image_shape))
                yield example

    def convert_poses(
        self,
        # poses: Float[Tensor, "batch 18"],
        poses_sphere: Float[Tensor, "batch ..."], # sphere poses
        poses: Float[Tensor, "batch ..."], # cubemap poses
        fxfycxcys: Float[Tensor, "batch ..."],
        cube_shape:  Int[Tensor, "2"],
    ) -> tuple[
        Float[Tensor, "batch 4 4"],  # extrinsics
        Float[Tensor, "batch 6 4 4"],  # cube poses
        Float[Tensor, "batch 6 3 3"],  # cube intrinsics
    ]:
        b = poses.shape[0]
        # Convert the intrinsics to a 3x3 normalized K matrix.
        intrinsics = torch.eye(3, dtype=torch.float32)
        intrinsics = repeat(intrinsics, "h w -> b h w", b=b).clone()
        # important!!!: normalize intrinsics for gaussian rasterization
        cube_h, cube_w = cube_shape
        
        # fx
        intrinsics[..., 0, 0] = fxfycxcys[..., 0] * 1.0 / cube_w

        # fy
        intrinsics[..., 1, 1] = fxfycxcys[..., 1] * 1.0 / cube_h

        # cx
        intrinsics[..., 0, 2] = fxfycxcys[..., 2] * 1.0 / cube_w

        # cy
        intrinsics[..., 1, 2] = fxfycxcys[..., 3] * 1.0 / cube_h
        intrinsics = repeat(intrinsics, "b h w -> b n h w", n=6)
        return poses_sphere, poses, intrinsics

    def convert_images(
        self,
        images: list[UInt8[Tensor, "..."]],
    ) -> Float[Tensor, "batch 3 height width"]:
        torch_images = []
        for image in images:
            image = Image.open(BytesIO(image.numpy().tobytes()))
            torch_images.append(self.to_tensor(image))
        return torch.stack(torch_images)
    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ) -> Float[Tensor, " view"]:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)

    @property
    def data_stage(self) -> Stage:
        if self.cfg.overfit_to_scene is not None:
            return "test"
        if self.stage == "val":
            return "test"
        return self.stage

    @cached_property
    def index(self) -> dict[str, Path]:
        merged_index = {}
        data_stages = [self.data_stage]
        if self.cfg.overfit_to_scene is not None:
            data_stages = ("test", "train")
        for data_stage in data_stages:
            for root in self.cfg.roots:
                # Load the root's index.
                with (root / data_stage / "index.json").open("r") as f:
                    index = json.load(f)
                index = {k: Path(root / data_stage / v) for k, v in index.items()}

                # The constituent datasets should have unique keys.
                assert not (set(merged_index.keys()) & set(index.keys()))

                # Merge the root's index into the main index.
                merged_index = {**merged_index, **index}
        return merged_index

    def __len__(self) -> int:
        if self.debug:
            return 2 # only for debug
        return (
            min(len(self.index.keys()) *
                self.cfg.test_times_per_scene, self.cfg.test_len)
            if self.stage == "test" and self.cfg.test_len > 0
            else len(self.index.keys()) * self.cfg.test_times_per_scene
        )


    