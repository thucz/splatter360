import subprocess
import sys
from pathlib import Path
from typing import TypedDict, Literal
import numpy as np
import torch
from jaxtyping import Float, Int, UInt8, Int32
from torch import Tensor
from tqdm import tqdm
from einops import repeat, rearrange
import json
import cv2
import os
import math
import multiprocessing

down_rate = 1
dataset_name = "replica"
basedir = "/wudang_vuc_3dc_afs/chenzheng"
dataset = dataset_name + "_dataset"
INPUT_IMAGE_DIR = Path(basedir + "/" + dataset)


# Target 100 MB per chunk.
TARGET_BYTES_PER_CHUNK = int(1e8)

def get_rotation_x(angle, device='cuda:0'):
        # print(angle)
        angle = math.radians(angle)
        sin, cos = math.sin(angle), math.cos(angle)
        r_mat = torch.eye(3).to(device)
        r_mat[1, 1] = cos
        r_mat[1, 2] = -sin
        r_mat[2, 1] = sin
        r_mat[2, 2] = cos
        return r_mat
def get_rotation_y(angle, device='cuda:0'):
    
    # print(angle)
    angle = math.radians(angle)
    sin, cos = math.sin(angle), math.cos(angle)
    r_mat = torch.eye(3).to(device)
    r_mat[0, 0] = cos
    r_mat[0, 2] = sin
    r_mat[2, 0] = -sin
    r_mat[2, 2] = cos
    return r_mat
def get_rotation_z(angle, device='cuda:0'):
    # print('***', angle)
    angle = math.radians(angle)
    sin, cos = math.sin(angle), math.cos(angle)
    r_mat = torch.eye(3).to(device)
    r_mat[0, 0] = cos
    r_mat[0, 1] = -sin
    r_mat[1, 0] = sin
    r_mat[1, 1] = cos
    return r_mat

# : Literal["test", "train"]
def get_example_keys(stage) -> list[str]:
    image_keys = set(
        example.name
        for example in tqdm((INPUT_IMAGE_DIR / stage).iterdir(), desc="Indexing images")
    )

    metadata_keys = set(
        example.stem
        for example in tqdm(
            (INPUT_METADATA_DIR / stage).iterdir(), desc="Indexing metadata"
        )
    )
    missing_image_keys = metadata_keys - image_keys
    if len(missing_image_keys) > 0:
        print(
            f"Found metadata but no images for {len(missing_image_keys)} examples.",
            file=sys.stderr,
        )
    missing_metadata_keys = image_keys - metadata_keys
    if len(missing_metadata_keys) > 0:
        print(
            f"Found images but no metadata for {len(missing_metadata_keys)} examples.",
            file=sys.stderr,
        )

    keys = image_keys & metadata_keys
    print(f"Found {len(keys)} keys.")
    return keys


def get_size(path: Path) -> int:
    """Get file or folder size in bytes."""
    return int(subprocess.check_output(["du", "-b", path]).split()[0].decode("utf-8"))


def load_raw(path: Path) -> UInt8[Tensor, " length"]:
    # import pdb;pdb.set_trace()
    return torch.tensor(np.memmap(path, dtype="uint8", mode="r"))


def load_images(example_path: Path) -> dict[int, UInt8[Tensor, "..."]]:
    """Load JPG images as raw bytes (do not decode)."""
    return {int(path.stem): load_raw(path) for path in example_path.iterdir()}

# TypedDict
class Metadata():
    url: str
    timestamps: Int[Tensor, " camera"]
    cameras: Float[Tensor, "camera entry"]


class Example(Metadata):
    key: str
    images: list[UInt8[Tensor, "..."]]
    # worker_func(cfg, stage, keys, num_worker, worker_index):

def worker_func(cfg, stage, keys, num_worker, worker_index):
    for key_id, key in enumerate(keys):
        if (key_id % num_worker == worker_index):
            print("key:", key)
            scene_dir = INPUT_IMAGE_DIR / stage / key 
            try:
                rotations = torch.from_numpy(np.load(scene_dir / "rotation.npy"))
                translations = torch.from_numpy(np.load(scene_dir / "translation.npy"))
            except:
                print('invalid rotation or translation file, skip ...')
                continue

            translations = translations.unsqueeze(-1)
            c2ws = torch.cat([rotations, translations], dim=-1)
            bottom = torch.tensor([[0., 0., 0., 1.0]])
            bottom = repeat(bottom, "() r -> n () r", n=c2ws.shape[0])
            c2ws = torch.cat([c2ws, bottom], dim=1)

            n = c2ws.shape[0]
            config_hfov = 90
            cubemap_width, cubemap_height = 256, 256
            width_center = cubemap_width / 2 - 0.5
            height_center = cubemap_height / 2 - 0.5
            focal_len = (cubemap_height / 2) / np.tan(config_hfov * np.pi / 180.0 / 2)
            fxfycxcys = torch.tensor([width_center, height_center, focal_len, focal_len]).to(torch.float32)
            fxfycxcys = repeat(fxfycxcys, "r -> n r", n=n)

            c2ws_front = c2ws.clone()
            c2ws_cubes = []
            device = c2ws.device
            # Top
            rotations_top = get_rotation_x(90, device)
            
            c2ws_top = c2ws_front.clone()        
            c2ws_top[:, :, :3] = torch.bmm(c2ws_front[:, :, :3], repeat(rotations_top, "r1 r2 -> n r1 r2", n=n))
            c2ws_cubes.append(c2ws_top)

            # Front
            c2ws_cubes.append(c2ws_front)

            # Left
            rotations_left = get_rotation_y(-90, device)
            c2ws_left = c2ws_front.clone()
            c2ws_left[:, :, :3] = torch.bmm(c2ws_front[:, :, :3], repeat(rotations_left, "r1 r2 -> n r1 r2", n=n))
            c2ws_cubes.append(c2ws_left)

            # Back
            rotations_back = get_rotation_y(-180, device)
            c2ws_back = c2ws_front.clone()
            c2ws_back[:, :, :3] = torch.bmm(c2ws_front[:, :, :3], repeat(rotations_back, "r1 r2 -> n r1 r2", n=n))
            c2ws_cubes.append(c2ws_back)

            # Right
            rotations_right = get_rotation_y(-270, device)
            c2ws_right = c2ws_front.clone()
            c2ws_right[:, :, :3] = torch.bmm(c2ws_front[:, :, :3], repeat(rotations_right, "r1 r2 -> n r1 r2", n=n))
            c2ws_cubes.append(c2ws_right)

            # Bottom
            rotations_bottom = get_rotation_x(-90, device)
            c2ws_bottom = c2ws_front.clone()
            c2ws_bottom[:, :, :3] = torch.bmm(c2ws_front[:, :, :3], repeat(rotations_bottom, "r1 r2 -> n r1 r2", n=n))
            c2ws_cubes.append(c2ws_bottom)

            # merge            
            c2ws_cubes = torch.stack(c2ws_cubes, dim=1)
            # habitat coordinate convention to Opencv/Colmap.
            # flip y, z    
            c2ws_cubes[..., 1] = -c2ws_cubes[..., 1]
            c2ws_cubes[..., 2] = -c2ws_cubes[..., 2]

            example = {}
            example['cameras'] = c2ws
            example['c2ws_cubes'] = c2ws_cubes
            filenames = sorted(list(scene_dir.iterdir()))
            example["file_paths"] = filenames #[filename for filename in filenames]
            num_bytes = get_size(scene_dir)
            # fxfycxcys = torch.stack(fxfycxcys, dim=0)
            example['fxfycxcys'] = fxfycxcys # 
            example['cube_shape'] = torch.tensor([cubemap_height, cubemap_width])
            
            # Add the key to the example.
            example["key"] = key
            if os.path.exists(scene_dir / "meta.pt"):
                os.system("rm -f "+str(scene_dir / "meta.pt"))
            torch.save(example, scene_dir / "meta.pt")

if __name__ == "__main__":
    # 等待子进程结束以后再继续往下运行，通常用于进程间的同步
    cfg = {
        "height": 512,
        "width": 1024,
    }

    stage = "test"
    keys = sorted(os.listdir(INPUT_IMAGE_DIR / stage))
    num_worker = 32
    processes = []

    for worker_index in range(num_worker):
        p = multiprocessing.Process(target=worker_func, args=(cfg, stage, keys, \
            num_worker, worker_index, ))
        p.start()                 # 用start()方法启动进程，执行worker_func()方法
        processes.append(p)

    for p in processes:
        p.join() # p.join()
