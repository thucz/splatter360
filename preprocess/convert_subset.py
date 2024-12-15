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
down_rate = 1

dataset_name = "replica"
basedir = "/wudang_vuc_3dc_afs/chenzheng"
dataset = dataset_name + "_dataset"

output_basedir = "/wudang_vuc_3dc_afs/chenzheng/" + dataset + "_pt"

INPUT_IMAGE_DIR = Path(basedir + "/" + dataset)
OUTPUT_DIR = Path(output_basedir)

# Target 100 MB per chunk.
TARGET_BYTES_PER_CHUNK = int(1e8)

def get_rotation_x(angle, device='cuda:0'):
    angle = math.radians(angle)
    sin, cos = math.sin(angle), math.cos(angle)
    r_mat = torch.eye(3).to(device)
    r_mat[1, 1] = cos
    r_mat[1, 2] = -sin
    r_mat[2, 1] = sin
    r_mat[2, 2] = cos
    return r_mat

def get_rotation_y(angle, device='cuda:0'):
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


# def load_images(example_path: Path) -> dict[int, UInt8[Tensor, "..."]]:
#     """Load JPG images as raw bytes (do not decode)."""
#     return {int(path.stem): load_raw(path) for path in example_path.iterdir()}

# # TypedDict
# class Metadata():
#     url: str
#     timestamps: Int[Tensor, "camera"]
#     cameras: Float[Tensor, "camera entry"]


# class Example(Metadata):
#     key: str
#     images: list[UInt8[Tensor, "..."]]

# def load_metadata(example_path: Path) :
#     # data = torch.load(example_path)
#     # cameras = torch.tensor(np.stack(cameras), dtype=torch.float32)
#     # n, 3, 4
#     c2ws = torch.cat([data['r_mats'], data['t_vecs']], dim=-1) 
    
#     bottom = torch.from_numpy(np.array([[0, 0, 0, 1]]))
#     bottom = repeat(bottom, "h w -> b h w", b=c2ws.shape[0])
#     c2ws = torch.cat([c2ws, bottom], dim=1)

#     # import pdb;pdb.set_trace()
#     # c2ws = torch.cat([])
#     return {
#         "cameras": c2ws,
#     }

if __name__ == "__main__":
    # debug = True
    # cnt = 100
    cfg = {
        "height": 512,
        "width": 1024,
    }
    # "train"
    for stage in ("test",):
        keys = sorted(os.listdir(INPUT_IMAGE_DIR / stage))
        chunk_size = 0
        chunk_index = 0
        chunk: list[Example] = []

        def save_chunk():
            global chunk_size
            global chunk_index
            global chunk

            chunk_key = f"{chunk_index:0>6}"
            print(
                f"Saving chunk {chunk_key} of {len(keys)} ({chunk_size / 1e6:.2f} MB)."
            )
            # import pdb;pdb.set_trace()
            dir = OUTPUT_DIR / stage
            dir.mkdir(exist_ok=True, parents=True)
            
            torch.save(chunk, f"{chunk_key}.torch")
            if os.path.exists( str(dir) + "/" f"{chunk_key}.torch"):
                os.system("rm " + str(dir) + "/" f"{chunk_key}.torch") # avoid disk input/output error                  
            os.system("cp " + f"{chunk_key}.torch" + " " + str(dir) + "/" f"{chunk_key}.torch")            
            os.system("rm " + f"{chunk_key}.torch")            
            
            # Reset the chunk.
            chunk_size = 0
            chunk_index += 1
            chunk = []

        for key in keys:
            print("key:", key)
            scene_dir = INPUT_IMAGE_DIR / stage / key 
            example = torch.load(scene_dir / "meta.pt")
            
            # num_bytes = get_size(scene_dir) # very slow when files are too many
            num_bytes = os.path.getsize(scene_dir / "meta.pt") # 文件路径及文件名
            print(f"Added {key} to chunk ({num_bytes / 1e6:.2f} MB).")
            chunk.append(example)            
            chunk_size += num_bytes
            if chunk_size >= TARGET_BYTES_PER_CHUNK:
                save_chunk()

        if chunk_size > 0:
            save_chunk()
