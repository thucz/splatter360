
# first compute MSE

# sort by MSE (delta mse, high to low)

# list scene ids

# output top 20 ids.

import json
import numpy as np
if __name__ == "__main__":
    # we put our method at the first place
    method_names = ["splat360", "mvsplat"] # , "panogrf"
    dataset_name = "hm3d" # "hm3d"
    json_directorys = {
        "splat360": "/mnt/disk1/chenzheng/splat360/outputs/splat360_log_depth_near0.1-100k/" + dataset_name + "/mse_dict.json",
        "mvsplat": "/wudang_vuc_3dc_afs/chenzheng/mse_directory/mvsplat_"+ dataset_name +"_mse.json",
        "panogrf": "/wudang_vuc_3dc_afs/chenzheng/mse_directory/panogrf_"+ dataset_name +"_mse.json",
        }

    # sort by delta mse, high to low
    mse_dict = {}
    for method_name in method_names:
        with open(json_directorys[method_name], "r") as fp:
            data = json.load(fp)
            mse_dict[method_name] = data
    
    scene_names = list(mse_dict[method_names[0]].keys())


    delta_mse_dict = {}
    for scene_name in scene_names:
        mse_ours = mse_dict[method_names[0]][scene_name]
        mse_other_min = np.inf
        for method_name in method_names[1:]:
            mse_other = mse_dict[method_name][scene_name]
            if mse_other < mse_other_min:
                mse_other_min = mse_other
           
        delta_mse = mse_ours - mse_other_min # smaller is better
        print("scene name: ", scene_name)
        delta_mse_dict[scene_name] = delta_mse
        # for method_name in method_names:
            # print("{}, mse: {}, delta mse: {}".format(method_name, mse_dict[method_name][scene_name]["mse"], mse_dict[method_name][scene_name]["delta_mse"]))
    result_tuple = sorted(delta_mse_dict.items(), key=lambda x:x[1])
    # for i in range(len(result_tuple)):
    with open("result.txt", "a") as fp:
        for i in range(len(result_tuple)):
            fp.write(str(i) + "," + str(result_tuple[i][0]) + "," + str(result_tuple[i][1])+"\n")
    