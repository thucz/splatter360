from dataclasses import dataclass
from typing import Literal

import torch
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor

from ...dataset import DatasetCfg
from ..types import Gaussians
from .cuda_splatting import DepthRenderingMode, render_cuda, render_depth_cuda
from .decoder import Decoder, DecoderOutput


@dataclass
class DecoderSplattingCUDACfg:
    name: Literal["splatting_cuda"]

class DecoderSplattingCUDA(Decoder[DecoderSplattingCUDACfg]):
    background_color: Float[Tensor, "3"]

    def __init__(
        self,
        cfg: DecoderSplattingCUDACfg,
        dataset_cfg: DatasetCfg,
    ) -> None:
        super().__init__(cfg, dataset_cfg)
        self.register_buffer(
            "background_color",
            torch.tensor(dataset_cfg.background_color, dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        depth_mode: DepthRenderingMode | None = None,
    ) -> DecoderOutput:
        b, v, _, _ = extrinsics.shape # 1 18 4 4
        device = extrinsics.device
        colors = torch.zeros((b, v, 3, *image_shape), dtype=torch.float32, device=device)
        for view_idx in range(v):
            colors[:, view_idx, ...] = render_cuda(
                extrinsics[:, view_idx],
                intrinsics[:, view_idx],
                near[:, view_idx],
                far[:, view_idx],
                image_shape,
                repeat(self.background_color, "c -> b c", b=b),
                gaussians.means,
                gaussians.covariances,
                gaussians.harmonics,
                gaussians.opacities,
            )
            # color = rearrange(color, "(b v) c h w -> b v c h w", b=b, v=v)
        # import pdb;pdb.set_trace()

        return DecoderOutput(
            colors,
            None
            if depth_mode is None
            else self.render_depth(
                gaussians, extrinsics, intrinsics, near, far, image_shape, depth_mode
            ),
        )

    def render_depth(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        mode: DepthRenderingMode = "depth",
    ) -> Float[Tensor, "batch view height width"]:
        b, v, _, _ = extrinsics.shape
        device = extrinsics.device
        depths = torch.zeros((b, v, *image_shape), dtype=torch.float32, device=device)
        for view_idx in range(v):
            depths[:, view_idx] = render_depth_cuda(
                extrinsics[:, view_idx],
                intrinsics[:, view_idx],
                near[:, view_idx], 
                far[:, view_idx],
                image_shape,
                gaussians.means,
                gaussians.covariances,
                gaussians.opacities,
                mode=mode,
            )            
        return depths
