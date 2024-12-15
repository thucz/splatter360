import json
from pathlib import Path
import torch
from tqdm import tqdm

dataset_name = "replica_dataset_pt"

DATASET_PATH = Path("/wudang_vuc_3dc_afs/chenzheng/" + dataset_name + "/")

if __name__ == "__main__":
    for stage in DATASET_PATH.iterdir():
        index = {}
        for chunk_path in tqdm(list(stage.iterdir()), desc=f"Indexing {stage.name}"):
            if chunk_path.suffix == ".torch":
                chunk = torch.load(chunk_path)
                for example in chunk:
                    index[example["key"]] = str(chunk_path.relative_to(stage))
        with (stage / "index.json").open("w") as f:
            json.dump(index, f)