
# from util import Equirec2Cube
import torch
from util import Equirec2Cube
import cv2
import os
from einops import rearrange
import numpy as np
if __name__ == "__main__":
    h, w = 512, 1024
    cube_h = h // 2
    e2c = Equirec2Cube(h, w, cube_h)
    pano_path = "/wudang_vuc_3dc_afs/chenzheng/hm3d_dataset/train/00000-kfPV7w3FaU5_0000/pano/00000.png"
    cubes = 6
    
    # original cubes
    cube_path = pano_path.replace("pano", "cubemaps").replace(".png", ".torch")
    cube_orig = torch.load(cube_path)
    cube_orig = cube_orig.data.cpu().numpy()

    # perspec_depth_path = str(file_path).replace("pano", "cubemaps_depth").replace(".png", ".torch")
    pano_frame = cv2.imread(pano_path, cv2.IMREAD_UNCHANGED)
    cv2.imwrite("./debug_e2c/pano_frame.png", pano_frame)
    pano_frame = cv2.cvtColor(pano_frame, cv2.COLOR_BGR2RGB)
    cube_frame = e2c.run(pano_frame) #

    cube_frame = torch.from_numpy(cube_frame) # h 6*w c
    cube_frame = rearrange(cube_frame, "h (cubes w) c -> cubes h w c", cubes=cubes)


    # convert

    # render

    # reordering:
    # [F R B L U D] <- [U B L F R D]
    
    cube_frame_new = cube_frame.clone()
    cube_frame_new[0] = cube_frame[4]
    cube_frame_new[1:3] = cube_frame[2:4]
    cube_frame_new[3:5] = cube_frame[0:2]
    # flip x,y
    cube_frame_new[0, :, :, :] = torch.flip(cube_frame_new[0, :, :, :], dims=[0, 1])
    cube_frame_new[5, :, :, :] = torch.flip(cube_frame_new[5, :, :, :], dims=[0, 1])
    
    # cube_frame_new[0, :, :, :] = cube_frame_new[0, ::-1, ::-1, :]
    # cube_frame_new[5, :, :, :] = cube_frame_new[5, ::-1, ::-1, :]

    cube_frame = cube_frame_new # 6 h w 3, RGB

    
    cube_frame = cube_frame.data.cpu().numpy()
    os.makedirs("./debug_e2c", exist_ok=True)
    for cube_id in range(cubes):
        cv2.imwrite(f"./debug_e2c/{cube_id}_e2c.png", cube_frame[cube_id])
        cv2.imwrite(f"./debug_e2c/{cube_id}_ori.png", cube_orig[cube_id])

    # for cube_idx
    # cube_frame = torch.from_numpy(cube_frame) * 1.0 / 255
    # print("cube_rgb.shape:", cube_frame.shape)
