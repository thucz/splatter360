import os
import sys
import torch
sys.path.append(".")
from configs.create_rgb_dataset_panogrf import RandomImageGenerator
from scipy.spatial.transform.rotation import Rotation
from PIL import Image
import numpy as np
import quaternion
import torch.nn.functional as F
from configs.utils360 import Utils360
from torchvision.transforms import ToTensor
import imageio

from configs.hm3d.options import (
    ArgumentParser,
    get_log_path,
    get_model_path,
    get_timestamp,
)
import cv2
from einops import rearrange
from dataset_generation.interpolate_trajectory import path_to_poses, interpolate_render_poses_m9d
# this is update the path in opts
from configs.options import get_dataset


# c2w
def warp_pano(source_matrix, compare_matrix, source_pano, compare_pano, source_depth, index):
    H, W = 512, 1024
    compare_matrix_inverse = np.linalg.inv(compare_matrix)
    source2compare_matrix = np.matmul(compare_matrix_inverse, source_matrix)
    ##### pano #####
    xs, ys = np.meshgrid(np.linspace(-1,1,W), np.linspace(-1,1,H))
    source_depth = source_depth.reshape(1,H,W)
    xs = xs.reshape(1,H,W)
    ys = ys.reshape(1,H,W)

    xs = (xs + 1) / 2 * W        # (1,512,1024)
    ys = (ys + 1) / 2 * H        # (1,512,1024)
                    
    theta = (0.5 - (xs + 0.5) / W) * 2 * np.pi
    phi = -((ys + 0.5) / H - 0.5) * np.pi
    
    y_source = np.sin(phi) * source_depth
    x_source = np.cos(phi) * np.sin(theta) * source_depth
    z_source = np.cos(phi) * np.cos(theta) * source_depth

    xyz_source = np.vstack((x_source, y_source, z_source, np.ones_like(z_source)))   # (4,512,1024)
    xyz_source = xyz_source.reshape(4, -1)
    xyz_compare = np.matmul(source2compare_matrix, xyz_source)                        # (4, H*W)
    xyz_compare = xyz_compare.reshape(4, H, W)[0:3]                                    # (3, H, W)

    theta_compare = np.arctan2(xyz_compare[0], xyz_compare[2])[None, ...]              # (1, H, W)
    compare_depth = np.linalg.norm(xyz_compare, axis=0)[None, ...]                     # (1, H, W)
    xyz_compare = xyz_compare / (compare_depth + 1e-6)                                 # (3, H, W)
    phi_compare = np.arctan2(xyz_compare[1], np.sqrt(np.square(xyz_compare[0]) + np.square(xyz_compare[2])))[None, ...]   # (1, H, W)

    xs_compare = (-theta_compare / (2 * np.pi) + 0.5) * W - 0.5                        # (1, H, W)
    ys_compare = (-phi_compare / np.pi + 0.5) * H - 0.5                                # (1, H, W)

    xs_compare = xs_compare / W * 2 - 1
    ys_compare = ys_compare / H * 2 - 1
    xys_compare = np.vstack((xs_compare, ys_compare)).reshape(2, H*W)                   # (2, H*W)
    xys_newimg = xys_compare

    # Create sampler
    sampler = torch.Tensor(xys_newimg).view(2, H, W).permute(1,2,0).unsqueeze(0).float()
    img1_tensor = ToTensor()(source_pano).unsqueeze(0)
    img2_tensor = ToTensor()(compare_pano).unsqueeze(0)

    img2_warped = F.grid_sample(img2_tensor, sampler)
    differences = torch.abs(img2_warped.squeeze().permute(1,2,0) - img1_tensor.squeeze().permute(1,2,0))
    max_value = torch.max(differences)
    min_value = torch.min(differences)
    mean_value = torch.mean(differences)
    num = torch.sum(differences <= 0.02).item()
    total_elements = differences.numel()
    percentage = num / total_elements

    if percentage < 0.5:
        return True
    else:
        return False

def export_data(dataset_name, split, root_dir, total_scene, samples_before_reset, scene_name):
    opts, _ = ArgumentParser().parse()
    opts.isTrain = False
    opts.dataset = dataset_name
    Dataset = get_dataset(opts)
    frame_cnt_threshold = 50
    frame_cnt_max = 500
    zero_ratio_threshold = 1.0
    root_dir = root_dir + "/" + opts.dataset + "_dataset/" + split # "./dataset_generation/hm3d_dataset_different"

    # # configs
    # if opts.dataset == 'hm3d':
    #     samples_before_reset = 5 #100        # num of path per scene
    #     if split == "train":
    #         total_scene = 800                 # num of scene
    #     else:
    #         total_scene = 100
    # elif opts.dataset == "replica":
    #     samples_before_reset = 1             # num of path per scene
    #     if split == "train":
    #         total_scene = 13                 # num of scene
    #     else:
    #         total_scene = 5

    image_generator = RandomImageGenerator(
        split,
        opts.render_ids[0],
        opts,
        vectorize=True,
        seed=0,
    )


    angle = np.random.uniform(0, 2 * np.pi)
    source_rotation = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]
    shortest_path_success_distance = 0
    shortest_path_max_steps = 500

    spherical_utils = Utils360(512, 1024)
    cubemap_rotations = [
            Rotation.from_euler('x', 90, degrees=True),   # Top
            Rotation.from_euler('y', 0, degrees=True),
            Rotation.from_euler('y', -90, degrees=True),
            Rotation.from_euler('y', -180, degrees=True),
            Rotation.from_euler('y', -270, degrees=True),
            Rotation.from_euler('x', -90, degrees=True),  # Bottom
        ]


    scenes_set = set()
    for scene_id in range(total_scene):
        print("scene_id:", scene_id)
        image_generator.env.reset()
        current_scene = image_generator.env.get_current_scene(index=0)
        if opts.dataset == "replica":
            current_scene = current_scene.split("/")[-3]
        elif opts.dataset == "hm3d":
            current_scene = current_scene.split("/")[-2]
        else:        
            raise NotImplementedError
        
        if opts.dataset == "hm3d":
            if current_scene[:5]  != scene_name: # only compare id
                continue
        elif opts.dataset == "replica":
            if current_scene != scene_name: # compare full name
                continue
        else:
            raise NotImplementedError
        
        if current_scene not in scenes_set:
            scenes_set.add(current_scene)
        else:
            print(f"repeat scene: {current_scene}, skip...")
            continue

        num_samples = 0   
        sim = image_generator.env

        while num_samples < samples_before_reset:
            print("scene_id, num_samples:", scene_id, num_samples)
            num = 0
            os.makedirs(os.path.join(root_dir, f'{current_scene}_{num_samples:04}'), exist_ok = True)

            source_position = sim.sample_navigable_point(0)
            target_position = sim.sample_navigable_point(0)
            shortest_paths = [sim.get_action_shortest_path(
                                    index=0,
                                    source_position=source_position,
                                    source_rotation=source_rotation,
                                    goal_position=target_position,
                                    success_distance=shortest_path_success_distance,
                                    max_episode_steps=shortest_path_max_steps,
                                )]
            
            shortest_path = shortest_paths[0]
            # TODO: remove later shortest path
            
            if len(shortest_path) > 0:
                c2ws = path_to_poses(shortest_path)
                c2ws = interpolate_render_poses_m9d(c2ws) # n 3 4
                c2ws[..., 1, 3] += 1.0 # camera height increase 1.0 meter.

                depths = []
                rgbs = []
                translations = []
                rotations = []
                rgbs_cube = []
                source_matrix = np.eye(4)
                index = None
                if (len(c2ws) < frame_cnt_threshold):
                    print(f"shortest_path {len(c2ws)} is too short, regenerate path...")
                    continue

                c2ws = c2ws[:frame_cnt_max]
                for index in range(len(c2ws)):
                    print(f"index {index} of {len(c2ws)}...")
                    rand_rotation = Rotation.from_matrix(c2ws[index, :3, :3]) # 
                    temp_rot = rand_rotation.as_quat()
                    rand_location = c2ws[index, :, 3]
                    obs = sim.get_observations_at(
                            index=0,
                            position=rand_location,
                            rotation=temp_rot, # input must be quaternion!!!
                        )
                    
                    temp_img = obs["rgb"]
                    temp_depth = obs["depth"][..., 0]
                    temp_translation, temp_quaternion = sim.get_agent_state(index=0)
                
                    rgb_cubemap_sides = []
                    depth_cubemap_sides = []

                    for j in range(6):
                        my_rotation = (rand_rotation * cubemap_rotations[j]).as_quat()
                        obs = sim.get_observations_at(
                            index=0,
                            position=rand_location,
                            rotation=my_rotation, 
                        )                   
                        rgb_cubemap_sides.append(obs["rgb"])    
                        depth_cubemap_sides.append(obs["depth"])
                                        
                    rgb_cubemap_sides = np.stack(rgb_cubemap_sides, axis=0)
                    rgb_erp_image = spherical_utils.stitch_cubemap(rgb_cubemap_sides, clip=False)
                    depth_cubemap_sides = np.stack(depth_cubemap_sides, axis=0)
                    depth_erp_image = spherical_utils.stitch_cubemap(depth_cubemap_sides, clip=False)

                    # image_filter
                    check_depth = spherical_utils.zdepth_to_distance(depth_erp_image[..., 0][None, ..., None])
                    zero_count = np.sum(check_depth == 0.0)
                    total_count = np.prod(check_depth.shape)
                    zero_ratio = zero_count / total_count

                    if zero_ratio > zero_ratio_threshold:
                        continue

                    rgb_save = rgb_erp_image.astype(np.uint8)
                    rotation_matrix = quaternion.as_rotation_matrix(quaternion.from_float_array(temp_quaternion))
                    
                    if num == 0:
                        source_matrix[0:3, 0:3] = rotation_matrix
                        source_matrix[0:3, 3] = temp_translation
                        source_pano = rgb_save
                        source_depth = spherical_utils.zdepth_to_distance(depth_erp_image[..., 0][None, ..., None])[..., 0]
                        num += 1
                        rotations.append(rotation_matrix)
                        translations.append(temp_translation)
                        depths.append(depth_erp_image)
                        rgbs.append(rgb_save)
                        rgbs_cube.append(rgb_cubemap_sides)
                    else:
                        compare_matrix = np.eye(4)
                        compare_matrix[0:3, 0:3] = rotation_matrix
                        compare_matrix[0:3, 3] = temp_translation
                        compare_pano = rgb_save
                        # if warp_pano(source_matrix, compare_matrix, source_pano, compare_pano, source_depth, num):
                        num += 1
                        source_matrix[0:3, 0:3] = rotation_matrix
                        source_matrix[0:3, 3] = temp_translation
                        source_pano = compare_pano
                        source_depth = spherical_utils.zdepth_to_distance(depth_erp_image[..., 0][None, ..., None])[..., 0]
                        rotations.append(rotation_matrix)
                        translations.append(temp_translation)
                        depths.append(depth_erp_image)
                        rgbs.append(rgb_save)
                        rgbs_cube.append(rgb_cubemap_sides)
                    
                    
                    # if num == frame_cnt:
                    #     break
                
                # if index == len(shortest_path) - 1:
                #     continue

                frame_cnt = len(rgbs)
                if (len(rgbs) < frame_cnt_threshold):
                    print("black area is too large, regenerate path")
                    continue

                translations = np.stack(translations, axis=0).astype(np.float32)

                np.save(os.path.join(root_dir, f'{current_scene}_{num_samples:04}', 'translation.npy'), translations)
                rotations = np.stack(rotations, axis=0).astype(np.float32)
                np.save(os.path.join(root_dir, f'{current_scene}_{num_samples:04}', 'rotation.npy'), rotations)

                # meta_dir 
                rgbs = np.stack(rgbs, axis=0)
                
                pano_dir = os.path.join(root_dir, f'{current_scene}_{num_samples:04}', 'pano')
                os.makedirs(pano_dir, exist_ok=True)

                depth_pano_dir = os.path.join(root_dir, f'{current_scene}_{num_samples:04}', 'pano_depth')
                os.makedirs(depth_pano_dir, exist_ok=True)

                depths = np.stack(depths, axis=0)
                depths = depths[..., 0:1]
                depths = spherical_utils.zdepth_to_distance(depths) # (N, 512, 1024, 1)
                
                # depths_torch = rearrange(torch.from_numpy(depths), "b h w c -> b c h w")
                # rgbs_torch = rearrange(torch.from_numpy(rgbs), "b h w c -> b c h w") # n, 512, 1024, 3
                
                for i in range(frame_cnt):
                    print(f"save rgb and depth frame {i} of video...")
                    # rgb_save_PIL = Image.fromarray(rgbs[i]) # 
                    # rgb_save_PIL.save(os.path.join(root_dir, f'{current_scene}_{num_samples:04}', f'{i}.webp'))
                    tmp = cv2.cvtColor(rgbs[i], cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(pano_dir, f'{i:05}.png'), tmp)

                    # b c h w                
                    # tmp = rgbs_torch[i] # 1 c h w
                    # torch.save(rgbs_torch[i].unsqueeze(0), os.path.join(pano_dir, f'{i:05}.torch'))
                    # torch.save(depths_torch[i].unsqueeze(0), os.path.join(depth_pano_dir, f'{i:05}.torch'))

                rgbs_cube = np.stack(rgbs_cube, axis=0) # n, 6, 256, 256, 3            
                cube_dir = os.path.join(root_dir, f'{current_scene}_{num_samples:04}', 'cubemaps')
                os.makedirs(cube_dir, exist_ok=True)
                for i in range(frame_cnt):
                    print(f"save cube rgb frame {i}...")
                    tmp = torch.from_numpy(rgbs_cube[i]) # 6, 256, 256, 3
                    torch.save(tmp, os.path.join(cube_dir, f'{i:05}.torch'))
                    # for j in range(6):
                    #     imageio.imwrite(os.path.join(cube_dir , f'{i:05}_{j:05}.webp'), rgbs_cube[i][j])
            else:
                print("shortest path not found, regenerate.")
                continue

            num_samples += 1
    
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='ProgramName',
        description='What the program does',
        epilog='Text at the bottom of help')
    parser.add_argument("--dataset_name", "-d", type=str, default='')
    parser.add_argument("--split", "-s", type=str, default='')
    parser.add_argument("--root_dir", "-r", type=str, default='root directory')
    parser.add_argument("--total_scene", "-t", type=int, default='total scene')
    parser.add_argument("--scene_name", type=int, default='total scene')
    parser.add_argument("--samples_before_reset", type=int, default="number of paths per scene")
    args = parser.parse_args()
    export_data(args.dataset_name, args.split, args.root_dir, total_scene=args.total_scene, samples_before_reset=args.samples_before_reset, scene_name = args.scene_name)
