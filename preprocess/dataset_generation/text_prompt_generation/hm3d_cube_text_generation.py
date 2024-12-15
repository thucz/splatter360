import torch
from PIL import Image
import os
import json
from tqdm import tqdm
from lavis.models import load_model_and_preprocess


# setup device to use
device = torch.device("cuda")

# loads BLIP-2 pre-trained model
model, vis_processors, _ = load_model_and_preprocess(name="blip2_t5", model_type="pretrain_flant5xxl", is_eval=True, device=device)

dataset_path_same = "./dataset_generation/hm3d_dataset_same"
dataset_path_different = "./dataset_generation/hm3d_dataset_different"
same_folders = os.listdir(dataset_path_same)
different_folders = os.listdir(dataset_path_different)

def key_function(file_name):
    return [int(part) if part.isdigit() else part for part in file_name.split('_')]

captions = []
for i, folder in enumerate(same_folders):
    cubes_folder = os.path.join(dataset_path_same, folder, 'cube')
    cubes = sorted(os.listdir(cubes_folder), key=key_function)
    for j, cube in enumerate(cubes):
        cube_path = os.path.join(cubes_folder, cube)
        cube_image = Image.open(cube_path).convert("RGB")
        cube_image = vis_processors["eval"](cube_image).unsqueeze(0).to(device)
        caption = model.generate({"image": cube_image, "prompt": "a photo of"})
        caption[0] = caption[0].capitalize() + '. '

        if (j+1) % 6 == 0:
            captions += caption
            full_caption = [''.join(captions)[:-1]]
            prompt_path = os.path.join(dataset_path_same, folder, 'prompt.txt')
            with open(prompt_path, 'a') as file:
                file.write(full_caption[0]+'\n')
            captions = []
        else:
            captions += caption

for i, folder in enumerate(different_folders):
    cubes_folder = os.path.join(dataset_path_different, folder, 'cube')
    cubes = sorted(os.listdir(cubes_folder), key=key_function)
    for j, cube in enumerate(cubes):
        cube_path = os.path.join(cubes_folder, cube)
        cube_image = Image.open(cube_path).convert("RGB")
        cube_image = vis_processors["eval"](cube_image).unsqueeze(0).to(device)
        caption = model.generate({"image": cube_image, "prompt": "a photo of"})
        caption[0] = caption[0].capitalize() + '. '

        if (j+1) % 6 == 0:
            captions += caption
            full_caption = [''.join(captions)[:-1]]
            prompt_path = os.path.join(dataset_path_different, folder, 'prompt.txt')
            with open(prompt_path, 'a') as file:
                file.write(full_caption[0]+'\n')
            captions = []
        else:
            captions += caption
