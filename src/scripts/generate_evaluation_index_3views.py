import json
from pathlib import Path
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--index_input', type=str, help='depth directory')
parser.add_argument('--index_output', type=str, help='dataset directory') 
args = parser.parse_args()


# INDEX_INPUT = Path("assets/evaluation_index_re10k.json")
# INDEX_OUTPUT = Path("assets/evaluation_index_re10k_video.json")
INDEX_INPUT = Path(args.index_input)
INDEX_OUTPUT = Path(args.index_output)


if __name__ == "__main__":
    with INDEX_INPUT.open("r") as f:
        index_input = json.load(f)

    index_output = {}
    for scene, scene_index_input in index_input.items():
        # Handle scenes for which there's no index.
        if scene_index_input is None:
            index_output[scene] = None
            continue

        # Add all intermediate frames as target frames.
        a, b = scene_index_input["context"]

        # add middle view as c

        view_idx = (a + b) // 2
        if view_idx not in scene_index_input["context"]:
            pass    # scene_index_input["context"].append(view_idx)
        else:
            index_output[scene] = None
            continue
        
        index_target_left = torch.randint(
            a,
            view_idx,
            size=(3, ),
            
        )        
        index_target_right = torch.randint(
            view_idx,
            b,
            size=(3, ),
        )
        index_target = torch.cat([index_target_left, index_target_right], dim=0)
        index_output[scene] = {
            "context": [a, view_idx, b],
            "target": index_target.data.cpu().numpy().tolist(),
        }

    with INDEX_OUTPUT.open("w") as f:
        json.dump(index_output, f)
