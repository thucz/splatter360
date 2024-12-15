import os
import torch
from spt_utils import Utils
from einops import rearrange, repeat
import argparse
import cv2
import numpy as np
import math
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

# 1. check epipolar line from one panoramic pixel to another panoramic view.
# 2. check epipolar line from one panoramic pixel to another view (six cubemaps).
def warp_pano_to_pano(base_dir, rgb_base_dir, scene_name, configs, near, far, mode, c2ws):

    

    # meta_path = os.path.join(base_dir, mode, '000000.torch')
    # meta = torch.load(meta_path)

    # fxfycxcys = meta[0]['fxfycxcys']
    # c2ws = meta[0]['cameras']
    # c2ws_cubes = meta[0]['c2ws_cubes']
    # scene_name = meta[0]['key']

    # read images
    img1_idx = 0
    img2_idx = 10
    x, y = 600, 300
    # emit a ray from (x, y) in image1
    # spherical_depths = torch.linspace(0.1, 200, steps=100)
    spherical_depths = 1 / torch.linspace(1 / far, 1 / near, steps=100)

    pano_dir = os.path.join(rgb_base_dir, mode, scene_name, 'pano')
    # cube_dir = os.path.join(rgb_base_dir, mode, scene_name, 'cubemaps')
    files = sorted(os.listdir(pano_dir))

    data1 = torch.load(os.path.join(pano_dir, files[img1_idx]))[0]
    data2 = torch.load(os.path.join(pano_dir, files[img2_idx]))[0]

    data1 = rearrange(data1, "c h w -> h w c").data.cpu().numpy()
    data2 = rearrange(data2, "c h w -> h w c").data.cpu().numpy()

    c2w1 = c2ws[img1_idx]
    c2w2 = c2ws[img2_idx]
    c2w1 = c2w1.unsqueeze(0)
    c2w2 = c2w2.unsqueeze(0)

    # visualize    
    cv2.circle(data1, (x, y), radius=1, color=(0, 0, 255), thickness=-1)
    debug_dir = "debug_epipolar_360"
    os.makedirs(debug_dir, exist_ok=True)
    cv2.imwrite(debug_dir + "/" + files[img1_idx][:-6] + ".png", data1)
    
    # transfer the dict object to an ArgumentParser object
    configs = argparse.Namespace(**configs)
    utils = Utils(configs)
    pixel = torch.from_numpy(np.array([x, y]))
    pixel = pixel.unsqueeze(0)
    for radius in spherical_depths:
        spherical = utils.equi_2_spherical(pixel, radius=radius) 
        cartesian = utils.spherical_2_cartesian(spherical)
        # homogeneous
        ones = torch.ones((cartesian.shape[0], 1))
        cartesian = torch.cat([cartesian, ones], dim=-1)
        cartesian = cartesian.unsqueeze(-1)
        cartesian_world = c2w1 @ cartesian 
        cartesian_camera = torch.inverse(c2w2) @ cartesian_world
        cartesian_camera = cartesian_camera[:, :3].squeeze(-1)
        spherical = utils.cartesian_2_spherical(cartesian_camera)
        equi = utils.spherical_2_equi(spherical)
        equi = equi.squeeze(0).data.cpu().numpy()
        cv2.circle(data2, (int(equi[0]), int(equi[1])), radius=1, color=(0, 0, 255), thickness=-1)
    cv2.imwrite(debug_dir + "/" + files[img2_idx][:-6] + ".png", data2)

# 2. check epipolar line from one panoramic pixel to another view (six cubemaps).
# warp_pano_to_pano(base_dir, rgb_base_dir, scene_name, configs, near, far, mode):
def warp_pano_to_cubemaps(base_dir, rgb_base_dir, scene_name, configs, near, far, mode, c2ws, c2ws_cubes, fxfycxcys):
    # infomation
    # base_dir = "/wudang_vuc_3dc_afs/chenzheng/park_1_pt"
    # rgb_base_dir = "/wudang_vuc_3dc_afs/chenzheng/park_rgb_1_pt"

    # config_hfov = 90
    # cubemap_width = 256, 256
    # width_center = cubemap_width / 2 - 0.5
    # height_center = cubemap_height / 2 - 0.5
    # focal_len = (cubemap_height / 2) / np.tan(config_hfov * np.pi / 180.0 / 2)

    # # mode = "test"
    # # read meta data
    # meta_path = os.path.join(base_dir, mode, '000000.torch')
    # meta = torch.load(meta_path)
    # fxfycxcys = meta[0]['fxfycxcys']
    # c2ws = meta[0]['cameras']
    # c2ws_cubes = meta[0]['c2ws_cubes'].to("cpu")
    # scene_name = meta[0]['key']

    # read images
    img1_idx = 10
    img2_idx = 30 #10
    x, y = 600, 300
    # emit a ray from (x, y) in image1
    # spherical_depths = torch.linspace(0.1, 200, steps=100)
    # near = 0.1
    # far = 100
    spherical_depths = 1 / torch.linspace(1 / far, 1 / near, steps=100)

    pano_dir = os.path.join(rgb_base_dir, mode, scene_name, 'pano')
    cube_dir = os.path.join(rgb_base_dir, mode, scene_name, 'cubemaps')
    files = sorted(os.listdir(pano_dir))

    # data1 = torch.load(os.path.join(pano_dir, files[img1_idx]))[0]

    data1 = cv2.imread(os.path.join(pano_dir, files[img1_idx]))
    data1 = cv2.cvtColor(data1, cv2.COLOR_BGR2RGB)
    
    # data1 = torch.from_numpy(data1) #, "h w c -> c h w")
    # data1 = rearrange(data1, "c h w -> h w c").data.cpu().numpy()

    # import pdb;pdb.set_trace()
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # import pdb;pdb.set_trace()
    data2_pano = cv2.imread(os.path.join(pano_dir, files[img2_idx]))

    
    data2 = torch.load(os.path.join(cube_dir, files[img2_idx].replace(".png", ".torch")))
    # data2 = cv2.imread(os.path.join(pano_dir, files[img2_idx]))
    # data2 = cv2.cvtColor(data2, cv2.COLOR_BGR2RGB)
    # data2 = rearrange(torch.from_numpy(data2), "h w c -> c h w")    
    data2 = data2.data.cpu().numpy() # n h w c
    # import pdb;pdb.set_trace()

    fx, fy, cx, cy = fxfycxcys[img2_idx]
    K = torch.tensor([[fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]]
                )

    c2w1 = c2ws[img1_idx]
    c2w2_cubes = c2ws_cubes[img2_idx] # 6 4 4

    c2w1 = c2w1.unsqueeze(0)
    c2w2_cubes = c2w2_cubes.unsqueeze(0) # 1 6 4 4

    # visualize
    cv2.circle(data1, (x, y), radius=1, color=(0, 0, 255), thickness=-1)
    # os.path.join(cube_dir, files[img2_idx].replace(".png", ".torch"))

    debug_dir = "debug_epipolar_360_cubes"
    os.makedirs(debug_dir, exist_ok=True)
    # .replace(".torch", ".png")
    cv2.imwrite(debug_dir + "/" + files[img1_idx], data1)
    cv2.imwrite(debug_dir + "/" + files[img2_idx], data2_pano)

    # configs = {
    #     "dataset_name": "outdoor_colmap",
    #     "batch_size": 1,
    #     "img_wh": [1024, 512],
    # }


    # transfer the dict object to an ArgumentParser object
    configs = argparse.Namespace(**configs)
    utils = Utils(configs)
    pixel = torch.from_numpy(np.array([x, y]))
    pixel = pixel.unsqueeze(0)
    for cube_idx in range(6):
        c2w2_cube = c2w2_cubes[:, cube_idx]
        data2_cube = data2[cube_idx]
        data2_cube = np.ascontiguousarray(data2_cube)
        
        for radius in spherical_depths:
            # import pdb;pdb.set_trace()
            spherical = utils.equi_2_spherical(pixel, radius=radius) 
            cartesian = utils.spherical_2_cartesian(spherical)

            # homogeneous
            ones = torch.ones((cartesian.shape[0], 1))
            cartesian = torch.cat([cartesian, ones], dim=-1)
            cartesian = cartesian.unsqueeze(-1)
            cartesian_world = c2w1 @ cartesian 
        
            cartesian_camera = torch.inverse(c2w2_cube) @ cartesian_world
            cartesian_camera = cartesian_camera[:, :3]

            
            cube_pixel = K.unsqueeze(0) @ cartesian_camera
            cube_pixel = cube_pixel.squeeze(-1).squeeze(0)
            cube_pixel = cube_pixel[:2] / cube_pixel[2:]

            
            if cube_pixel[0] >= 0 and cube_pixel[0] < data2.shape[2] \
                and cube_pixel[1] >= 0 and cube_pixel[1] < data2.shape[1]:
                # import pdb;pdb.set_trace()
                cv2.circle(data2_cube, (int(cube_pixel[0]), int(cube_pixel[1])), radius=1, color=(0, 0, 255), thickness=-1)
        # import pdb
        cv2.imwrite(debug_dir + "/" + files[img2_idx].replace(".png", "") + "_cube_" + str(cube_idx) +".png", data2_cube)

if __name__ == "__main__":
    # infomation
    dataset = "replica"
    base_dir = "/wudang_vuc_3dc_afs/chenzheng/" + dataset + "_dataset2"
    rgb_base_dir = "/wudang_vuc_3dc_afs/chenzheng/" + dataset + "_dataset2"
    mode = "train"
    if dataset == "replica":
        scene_name = "frl_apartment_3_0000"
        configs = {
            "dataset_name": dataset,
            "batch_size": 1,
            "img_wh": [1024, 512],
        }
        near = 0.1
        far = 15.0
    elif dataset == "outdoor_colmap":
        pass
    else: 
        raise NotImplementedError

    # read meta data
    rotations = torch.from_numpy(np.load(os.path.join(base_dir, mode, scene_name, "rotation.npy")))
    translations = torch.from_numpy(np.load(os.path.join(base_dir, mode, scene_name, "translation.npy")))
    translations = translations.unsqueeze(-1)
    # import pdb;pdb.set_trace()
    c2ws = torch.cat([rotations, translations], dim=-1)
    bottom = torch.tensor([[0., 0., 0., 1.0]])
    bottom = repeat(bottom, "() r -> n () r", n=c2ws.shape[0])
    c2ws = torch.cat([c2ws, bottom], dim=1)
    option = "warp_pano_to_cubemaps"

    if option == "warp_pano_to_pano":
        warp_pano_to_pano(base_dir, rgb_base_dir, scene_name, configs, near, far, mode, c2ws)
    elif option == "warp_pano_to_cubemaps":
        n = c2ws.shape[0]
        # warp_pano_to_pano()
        config_hfov = 90
        cubemap_width, cubemap_height = 256, 256
        width_center = cubemap_width / 2 - 0.5
        height_center = cubemap_height / 2 - 0.5
        focal_len = (cubemap_height / 2) / np.tan(config_hfov * np.pi / 180.0 / 2)
        fxfycxcys = torch.tensor([width_center, height_center, focal_len, focal_len]).to(torch.float32)
        fxfycxcys = repeat(fxfycxcys, "r -> n r", n=n)
        # habitat coordinate convention to Opencv/Colmap.
        # flip y, z    
        # c2ws[..., 1] = -c2ws[..., 1]
        # c2ws[..., 2] = -c2ws[..., 2]

        c2ws_front = c2ws.clone()
        # habitat coordinate convention to Opencv/Colmap.
        # flip y, z    
        # c2ws_front[..., 1] = -c2ws_front[..., 1]
        # c2ws_front[..., 2] = -c2ws_front[..., 2]
        

   
        # cubemap_rotations = [
        #     Rotation.from_euler('x', 90, degrees=True),   # Top
        #     Rotation.from_euler('y', 0, degrees=True),
        #     Rotation.from_euler('y', -90, degrees=True),
        #     Rotation.from_euler('y', -180, degrees=True),
        #     Rotation.from_euler('y', -270, degrees=True),
        #     Rotation.from_euler('x', -90, degrees=True),  # Bottom
        # ]
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

        c2ws_cubes = torch.stack(c2ws_cubes, dim=1)

        # # habitat coordinate convention to Opencv/Colmap.
        # # flip y, z    
        c2ws_cubes[..., 1] = -c2ws_cubes[..., 1]
        c2ws_cubes[..., 2] = -c2ws_cubes[..., 2]
     
        warp_pano_to_cubemaps(base_dir, rgb_base_dir, scene_name, configs, near, far, mode, c2ws, c2ws_cubes, fxfycxcys)
