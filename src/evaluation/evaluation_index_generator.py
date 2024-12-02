import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from einops import rearrange
from pytorch_lightning import LightningModule
from tqdm import tqdm

from ..geometry.epipolar_lines import project_rays
from ..geometry.projection import get_world_rays, sample_image_grid
from ..misc.image_io import save_image
from ..visualization.annotation import add_label
from ..visualization.layout import add_border, hcat

@dataclass
class EvaluationIndexGeneratorCfg:
    num_target_views: int
    num_context_views: int
    min_distance: int
    max_distance: int
    min_overlap: float
    max_overlap: float
    # min_camera_distance: float
    # max_camera_distance: float
    output_path: Path
    save_previews: bool
    seed: int
    frame_interval: int # 100

@dataclass
class IndexEntry:
    context: tuple[int, ...]
    target: tuple[int, ...]


class EvaluationIndexGenerator(LightningModule):
    generator: torch.Generator
    cfg: EvaluationIndexGeneratorCfg
    index: dict[str, IndexEntry | None]

    def __init__(self, cfg: EvaluationIndexGeneratorCfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.generator = torch.Generator()
        self.generator.manual_seed(cfg.seed)
        self.index = {}

    def test_step(self, batch, batch_idx):
        # print("batch_idx:", batch_idx)
        # b, v, _, h, w = batch["target"]["image_sphere"].shape
        # assert b == 1
        extrinsics = batch["target"]["extrinsics_sphere"][0]
        b, v = batch["target"]["extrinsics_sphere"].shape[:2]
        assert b == 1
        


        # intrinsics = batch["target"]["intrinsics"][0]
        scene = batch["scene"][0]

        
        # context_indices = torch.randperm(v, generator=self.generator)
        
        # for context_index in tqdm(context_indices, "Finding context pair"):
        #     # xy, _ = sample_image_grid((h, w), self.device)
        #     # context_origins, context_directions = get_world_rays(
        #     #     rearrange(xy, "h w xy -> (h w) xy"),
        #     #     extrinsics[context_index],
        #     #     intrinsics[context_index],
        #     #     unnormalized_ray_dir=False,
        #     # )
        #     # Step away from context view until the minimum overlap threshold is met.
        #     valid_indices = []
        #     for step in (1, -1):
        #         min_distance = self.cfg.min_distance
        #         max_distance = self.cfg.max_distance
        #         current_index = context_index + step * min_distance
        #         while 0 <= current_index.item() < v:
        #             # print("current_index:", current_index)
        #             # Compute overlap.                    
        #             # current_origins, current_directions = get_world_rays(
        #             #     rearrange(xy, "h w xy -> (h w) xy"),
        #             #     extrinsics[current_index],
        #             #     intrinsics[current_index],
        #             #     unnormalized_ray_dir=False,
        #             # )
        #             # projection_onto_current = project_rays(
        #             #     context_origins,
        #             #     context_directions,
        #             #     extrinsics[current_index],
        #             #     intrinsics[current_index],
        #             # )
        #             # projection_onto_context = project_rays(
        #             #     current_origins,
        #             #     current_directions,
        #             #     extrinsics[context_index],
        #             #     intrinsics[context_index],
        #             # )

        #             # overlap_a = projection_onto_context["overlaps_image"].float().mean()
        #             # overlap_b = projection_onto_current["overlaps_image"].float().mean()

        #             # overlap = min(overlap_a, overlap_b)
        #             delta = (current_index - context_index).abs()
        #             # torch.sqrt(torch.sum(torch.pow(torch.subtract(a, b), 2), dim=0))  
        #             # camera_distance = torch.linalg.norm(
        #             #     extrinsics[current_index][..., :3, 3] - \
        #             #     extrinsics[context_index][..., :3, 3])
                    
        #             # min_camera_distance = self.cfg.min_camera_distance
        #             # max_camera_distance = self.cfg.max_camera_distance

                    # if min_camera_distance <= camera_distance <= max_camera_distance:

        context_left = 40 # we skip the first 40 frames because they are almost keep at the same place with small translations.
        context_right = context_left + self.cfg.frame_interval
        if self.cfg.num_context_views > 2:
            # import pdb;pdb.set_trace()
            context_middle = torch.randint(context_left, context_right, size=(self.cfg.num_context_views-2, ))
            # print('context_middle:', context_middle)
            context_middle = [ data.item() for data in context_middle] # torch to ordinary scalar.
            # import pdb;pdb.set_trace()
            context_views = [context_left, *context_middle, context_right]
            context = tuple(sorted(context_views))
        else:
            context = (context_left, context_right)


        # context_left = 40
        # context_right = 70

        print('context_right:', context_right)
        # Pick non-repeated random target views.
        while True:
            target_views = torch.randint(
                context_left,
                context_right + 1,
                (self.cfg.num_target_views,),
                generator=self.generator,
            )
            if (target_views.unique(return_counts=True)[1] == 1).all():
                break

        target = tuple(sorted(target_views.tolist()))
        self.index[scene] = IndexEntry(
            context=context,
            target=target,
        )

              
        
            # # This happens if no starting frame produces a valid evaluation example.
            # self.index[scene] = None

    def save_index(self) -> None:
        self.cfg.output_path.mkdir(exist_ok=True, parents=True)
        with (self.cfg.output_path / "evaluation_index.json").open("w") as f:
            json.dump(
                {k: None if v is None else asdict(v) for k, v in self.index.items()}, f
            )
