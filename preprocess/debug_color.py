split = "test"
dataset = "replica"

root_dir = "/wudang_vuc_3dc_afs/chenzheng/" + dataset+"_dataset2/" + split # "./dataset_generation/hm3d_dataset_different"
scene_name = "hotel_0_0000"
frame_idx = 0
pano_path = root_dir + "/" + scene_name + "/" + "pano" + "/" + f"{frame_idx:05}" +".png"
import cv2
import os
import torch

pano_data = cv2.imread(pano_path)
pano_data = cv2.cvtColor(pano_data, cv2.COLOR_BGR2RGB)

os.makedirs("./debug_color", exist_ok=True)
pano_data = cv2.cvtColor(pano_data, cv2.COLOR_RGB2BGR)
cv2.imwrite("./debug_color/pano.png", pano_data)

cube_path = root_dir + "/" + scene_name + "/" + "cubemaps" + "/" + f"{frame_idx:05}" + ".torch"
cube_data = torch.load(cube_path).data.cpu().numpy() # 6 256 256 3

# cv2.cvtColor(data_one, cv2.COLOR_)
data_one = cube_data[0]
# cube_data = cube_data[..., ::-1]
data_one = cv2.cvtColor(data_one, cv2.COLOR_RGB2BGR)
cv2.imwrite("./debug_color/cube_0.png", data_one)




