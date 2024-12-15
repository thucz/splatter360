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
import multiprocessing


def worker_func(opts, image_generator, samples_before_reset, \
    root_dir, worker_index):
    name = multiprocessing.current_process().name        # 获取当前进程的名字

    total_scene = len(image_generator.dataset[worker_index].episodes)

    print(
        name, 'starting',
        "total_scene: ", total_scene,  
        "worker_id: ", worker_index,
        )    
    for scene_id in range(total_scene):
        # image_generator.env.reset() # reset all envs
        image_generator.env.reset_at(worker_index) # reset specific env !!!
        current_scene = image_generator.env.get_current_scene(index=worker_index)
        print(f"current_scene: {current_scene} in {name}...")
        # import pdb;pdb.set_trace()
        if opts.dataset == "replica":
            current_scene = current_scene.split("/")[-3]
        elif opts.dataset == "hm3d":
            current_scene = current_scene.split("/")[-2]
        else:
            raise NotImplementedError    
        # hard-coded parameters
        frame_cnt_threshold = 50
        frame_cnt_max = 500
        zero_ratio_threshold = 1.0

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
        
        num_samples = 0   
        sim = image_generator.env
        while num_samples < samples_before_reset:
            print("scene_id, num_samples:", scene_id, num_samples)
            num = 0
            os.makedirs(os.path.join(root_dir, f'{current_scene}_{num_samples:04}'), exist_ok = True)
            source_position = sim.sample_navigable_point(worker_index)
            target_position = sim.sample_navigable_point(worker_index)
            shortest_path = sim.get_action_shortest_path(
                                    index=worker_index,
                                    source_position=source_position,
                                    source_rotation=source_rotation,
                                    goal_position=target_position,
                                    success_distance=shortest_path_success_distance,
                                    max_episode_steps=shortest_path_max_steps,
                                )
            
            # shortest_paths[0]        
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
                depths_cube = []
                source_matrix = np.eye(4)
                index = None
                if (len(c2ws) < frame_cnt_threshold):
                    print(f"shortest_path {len(c2ws)} is too short, regenerate path...")
                    continue
                
                cube_dir = os.path.join(root_dir, f'{current_scene}_{num_samples:04}', 'cubemaps')
                depth_cube_dir = os.path.join(root_dir, f'{current_scene}_{num_samples:04}', 'cubemaps_depth')
                os.makedirs(cube_dir, exist_ok=True)
                os.makedirs(depth_cube_dir, exist_ok=True)

                pano_dir = os.path.join(root_dir, f'{current_scene}_{num_samples:04}', 'pano')
                os.makedirs(pano_dir, exist_ok=True)

                depth_pano_dir = os.path.join(root_dir, f'{current_scene}_{num_samples:04}', 'pano_depth')
                os.makedirs(depth_pano_dir, exist_ok=True)


                c2ws = c2ws[:frame_cnt_max]
                for index in range(len(c2ws)):
                    print(f"index {index} of {len(c2ws)}...")
                    rand_rotation = Rotation.from_matrix(c2ws[index, :3, :3]) # 
                    temp_rot = rand_rotation.as_quat()
                    rand_location = c2ws[index, :, 3]
                    obs = sim.get_observations_at(
                            index=worker_index,
                            position=rand_location,
                            rotation=temp_rot, # input must be quaternion!!!
                        )
                    temp_img = obs["rgb"]
                    temp_depth = obs["depth"][..., 0]
                    temp_translation, temp_quaternion = sim.get_agent_state(index=worker_index)
                
                    rgb_cubemap_sides = []
                    depth_cubemap_sides = []

                    for j in range(6):
                        my_rotation = (rand_rotation * cubemap_rotations[j]).as_quat()
                        obs = sim.get_observations_at(
                            index=worker_index,
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

                    # rgb_save = rgb_erp_image.astype(np.uint8)
                    rotation_matrix = quaternion.as_rotation_matrix(quaternion.from_float_array(temp_quaternion))
                    
                    rotations.append(rotation_matrix)
                    translations.append(temp_translation)

                    # cube rgb
                    rgb_cubes = torch.from_numpy(rgb_cubemap_sides) # 6, 256, 256, 3
                    torch.save(rgb_cubes, os.path.join(cube_dir, f'{index:05}.torch'))

                    # cube depth
                    depth_cubes = torch.from_numpy(depth_cubemap_sides) # 6, 256, 256, 1                
                    torch.save(depth_cubes, os.path.join(depth_cube_dir, f'{index:05}.torch'))
                    # print("depth_cubes.shape:", depth_cubes.shape)

                    # erp rgb
                    tmp = cv2.cvtColor(rgb_erp_image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(pano_dir, f'{index:05}.png'), tmp)

                    # erp depth
                    depth_erp_image = depth_erp_image[..., 0:1]
                    depth_erp = spherical_utils.zdepth_to_distance(depth_erp_image[np.newaxis, ...]) # (512, 1024, 1)
                    depth_erp = depth_erp.squeeze(0).squeeze(-1)
                    # only for indoor scenes <= 25 meters
                    tmp_depth = np.uint16(depth_erp * 1000)
                    cv2.imwrite(os.path.join(depth_pano_dir, f'{index:05}.png'), tmp_depth)

                translations = np.stack(translations, axis=0).astype(np.float32)
                np.save(os.path.join(root_dir, f'{current_scene}_{num_samples:04}', 'translation.npy'), translations)
                rotations = np.stack(rotations, axis=0).astype(np.float32)
                np.save(os.path.join(root_dir, f'{current_scene}_{num_samples:04}', 'rotation.npy'), rotations)

                # meta_dir 
                # rgbs = np.stack(rgbs, axis=0)
                # depths = np.stack(depths, axis=0)
                
                
                # depths_torch = rearrange(torch.from_numpy(depths), "b h w c -> b c h w")
                # rgbs_torch = rearrange(torch.from_numpy(rgbs), "b h w c -> b c h w") # n, 512, 1024, 3
                
                # for i in range(frame_cnt):
                #     print(f"save rgb and depth frame {i} of video...")
                    # rgb_save_PIL = Image.fromarray(rgbs[i]) # 
                    # rgb_save_PIL.save(os.path.join(root_dir, f'{current_scene}_{num_samples:04}', f'{i}.webp'))
                    

                    # b c h w                
                    # tmp = rgbs_torch[i] # 1 c h w
                    # torch.save(rgbs_torch[i].unsqueeze(0), os.path.join(pano_dir, f'{i:05}.torch'))
                    # torch.save(depths_torch[i].unsqueeze(0), os.path.join(depth_pano_dir, f'{i:05}.torch'))

                # depths_cube = np.stack(depths_cube)
                # rgbs_cube = np.stack(rgbs_cube, axis=0) # n, 6, 256, 256, 3            

                # for i in range(frame_cnt):
                #     print(f"save cube rgb frame {i}...")
                    
             
                    # for j in range(6):
                    #     imageio.imwrite(os.path.join(cube_dir , f'{i:05}_{j:05}.webp'), rgbs_cube[i][j])
            else:
                print("shortest path not found, regenerate.")
                continue

            num_samples += 1


def process_mp(dataset_name, split, root_dir, samples_before_reset, num_worker):
    opts, _ = ArgumentParser().parse()
    opts.isTrain = False
    opts.dataset = dataset_name
    Dataset = get_dataset(opts)

    # work_num
    root_dir = root_dir + "/" + opts.dataset + "_dataset/" + split # "./dataset_generation/hm3d_dataset_different"

    image_generator = RandomImageGenerator(
        split,
        opts.render_ids[0],
        opts,
        vectorize=True,
        seed=0,
        num_parallel_envs=num_worker,
    )    

    # single_process
    # worker_func(opts, image_generator, samples_before_reset, root_dir, worker_index=0)
    # multi-process

    processes = []
    for worker_index in range(num_worker):
        # worker_func(opts, image_generator, samples_before_reset, \
        p = multiprocessing.Process(target=worker_func, args=(opts, image_generator, 
            samples_before_reset, root_dir, worker_index, ))       
        p.start()                 # 用start()方法启动进程，执行worker_func()方法
        processes.append(p)
    
    for p in processes:
        p.join() # p.join()                  # 等待子进程结束以后再继续往下运行，通常用于进程间的同步         

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", "-d", type=str, default='hm3d')
    parser.add_argument("--split", "-s", type=str, default='test')
    parser.add_argument("--root_dir", "-r", type=str, default='/wudang_vuc_3dc_afs/chenzheng')
    parser.add_argument("--samples_before_reset", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()
    process_mp(args.dataset_name, args.split, args.root_dir, samples_before_reset=args.samples_before_reset, num_worker=args.num_workers)
