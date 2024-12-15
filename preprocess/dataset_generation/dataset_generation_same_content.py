import os
import sys
sys.path.append(".")
from configs.create_rgb_dataset_panogrf import RandomImageGenerator
from scipy.spatial.transform.rotation import Rotation
from PIL import Image
import numpy as np
import quaternion
from configs.utils360 import Utils360
import imageio
from configs.hm3d.options import (
    ArgumentParser,
    get_log_path,
    get_model_path,
    get_timestamp,
)

opts, _ = ArgumentParser().parse()
opts.isTrain = False
split = "train" #'val'

# this is update the path in opts
from configs.options import get_dataset
Dataset = get_dataset(opts)

image_generator = RandomImageGenerator(
    split,
    opts.render_ids[0],
    opts,
    vectorize=True,
    seed=0,
)

# configs
root_dir = "./dataset_generation/hm3d_dataset_same" 
samples_before_reset = 50        # num of path per scene
total_scene = 800                 # num of scene 

angle = np.random.uniform(0, 2 * np.pi)
source_rotation = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]
shortest_path_success_distance = 0
shortest_path_max_steps = 7

spherical_utils = Utils360(512, 1024)
cubemap_rotations = [
        Rotation.from_euler('x', 90, degrees=True),   # Top
        Rotation.from_euler('y', 0, degrees=True),
        Rotation.from_euler('y', -90, degrees=True),
        Rotation.from_euler('y', -180, degrees=True),
        Rotation.from_euler('y', -270, degrees=True),
        Rotation.from_euler('x', -90, degrees=True),  # Bottom
    ]

for scene_id in range(total_scene):
    image_generator.env.reset()
    current_scene = image_generator.env.get_current_scene(index=0)
    flag = False
    
    num_samples = 0   
    sim = image_generator.env

    while num_samples < samples_before_reset:
        filter_flag = True
        os.makedirs(os.path.join(root_dir, f'{current_scene}_{num_samples:04}'), exist_ok = True)
        os.makedirs(os.path.join(root_dir, f'{current_scene}_{num_samples:04}', 'cube'), exist_ok = True)

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

        if len(shortest_path) > 0:
            depths = []
            rgbs = []
            translations = []
            rotations = []
            
            for index in range(len(shortest_path)):                
                temp_pos = shortest_path[index].position
                temp_pos[1] += 1.0
                temp_rot = shortest_path[index].rotation
                
                rand_rotation = Rotation.from_quat(temp_rot)
                rand_location = temp_pos

                obs = sim.get_observations_at(
                        index=0,
                        position=rand_location,
                        rotation=temp_rot,
                    )
                temp_img = obs["rgb"]
                temp_depth = obs["depth"][...,0]
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
                if zero_ratio > 0.002:
                    filter_flag = False
                    break

                rgb_save = rgb_erp_image.astype(np.uint8)
                rgb_save_PIL = Image.fromarray(rgb_save)
                rgb_save_PIL.save(os.path.join(root_dir, f'{current_scene}_{num_samples:04}', f'{index}.webp'))

                depths.append(depth_erp_image)
                rgbs.append(rgb_erp_image.astype(np.uint8))

                rotation_matrix = quaternion.as_rotation_matrix(quaternion.from_float_array(temp_quaternion))
                rotations.append(rotation_matrix)
                translations.append(temp_translation)
                
            if not filter_flag:
                continue

            for j in range(6):                 
                imageio.imwrite(os.path.join(root_dir, f'{current_scene}_{num_samples:04}', 'cube' , f'{j}.webp'), rgb_cubemap_sides[j])

            translations = np.stack(translations, axis=0).astype(np.float32)
            np.save(os.path.join(root_dir, f'{current_scene}_{num_samples:04}', 'translation.npy'), translations)
            rotations = np.stack(rotations, axis=0).astype(np.float32)
            np.save(os.path.join(root_dir, f'{current_scene}_{num_samples:04}', 'rotation.npy'), rotations)
            
            rgbs = np.stack(rgbs, axis=0)
            depths = np.stack(depths, axis=0)
            depths = depths[..., 0:1]
            depths = spherical_utils.zdepth_to_distance(depths)

        else:
            continue

        num_samples += 1