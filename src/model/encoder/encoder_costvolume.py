from dataclasses import dataclass
from typing import Literal, Optional, List

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn
from collections import OrderedDict

from ...dataset.shims.bounds_shim import apply_bounds_shim
from ...dataset.shims.patch_shim import apply_patch_shim
from ...dataset.types import BatchedExample, DataShim
from ...geometry.projection import sample_image_grid
from ..types import Gaussians
from .backbone import (
    BackboneMultiview,
)
# from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg
from .common.gaussian_adapter_erp import GaussianAdapterERP, GaussianAdapterERPCfg
from .encoder import Encoder
# from .costvolume.depth_predictor_multiview import DepthPredictorMultiView
from .costvolume.depth_predictor_multiview_360 import DepthPredictorMultiView360

from .visualization.encoder_visualizer_costvolume_cfg import EncoderVisualizerCostVolumeCfg

from ...global_cfg import get_cfg
from .epipolar.epipolar_sampler import EpipolarSampler
from ..encodings.positional_encoding import PositionalEncoding
from ...geometry.layers import Conv3x3, ConvBlock, upsample, Cube2Equirec, Concat, BiProj, CEELayer
from ...geometry.utils360 import Utils

import torch.nn.functional as F
# from .layers import 
@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int


@dataclass
class EncoderCostVolumeCfg:
    name: Literal["costvolume"]
    d_feature: int
    num_depth_candidates: int
    num_surfaces: int
    visualizer: EncoderVisualizerCostVolumeCfg
    gaussian_adapter: GaussianAdapterERPCfg
    opacity_mapping: OpacityMappingCfg
    gaussians_per_pixel: int
    unimatch_weights_path: str | None
    downscale_factor: int
    shim_patch_size: int
    multiview_trans_attn_split: int
    costvolume_unet_feat_dim: int
    costvolume_unet_channel_mult: List[int]
    costvolume_unet_attn_res: List[int]
    depth_unet_feat_dim: int
    depth_unet_attn_res: List[int]
    depth_unet_channel_mult: List[int]
    wo_depth_refine: bool
    wo_cost_volume: bool
    wo_backbone_cross_attn: bool
    wo_cost_volume_refine: bool
    use_epipolar_trans: bool

    use_cross_view_self_attn: bool
    depth_sampling_type: Literal["linear_depth", "log_depth", "inverse_depth"]

    add_mono_feat: bool
    vit_type: Literal["vits", "vitb", "vitl"]

    pretrained_monodepth: str

    wo_cube_encoder: bool
    wo_erp_encoder: bool

class EncoderCostVolume(Encoder[EncoderCostVolumeCfg]):
    backbone: BackboneMultiview
    depth_predictor_erp:  DepthPredictorMultiView360
    gaussian_adapter_erp: GaussianAdapterERP

    def __init__(self, cfg: EncoderCostVolumeCfg) -> None:
        super().__init__(cfg)
        if not cfg.wo_cube_encoder:
            self.backbone = BackboneMultiview(
                feature_channels=cfg.d_feature,
                downscale_factor=cfg.downscale_factor,
                no_cross_attn=cfg.wo_backbone_cross_attn,
                use_epipolar_trans=cfg.use_epipolar_trans,
            )

        global_cfg = get_cfg()
        self.cube_h = global_cfg.dataset.image_shape[0] // cfg.downscale_factor // 2
        self.equi_h = global_cfg.dataset.image_shape[0] // cfg.downscale_factor
        self.equi_w = global_cfg.dataset.image_shape[1] // cfg.downscale_factor
        self.dataset_name = global_cfg.dataset.name
        print(f"in encoder: self.cube_h, self.equi_h, self.equi_w: {self.cube_h, self.equi_h, self.equi_w}")
        self.c2e = Cube2Equirec(self.cube_h, self.equi_h, self.equi_w)
        
        # self.debug = False
        # if self.debug:
        #     self.c2e_debug = Cube2Equirec(256, 512, 1024)
        

        
        if cfg.add_mono_feat:
            self.pretrained = torch.hub.load("facebookresearch/dinov2", "dinov2_{:}14".format(cfg.vit_type))
            del self.pretrained.mask_token
            # freeze_mono model to reduce GPU memory.
            # for name, para in self.pretrained_monodepth_model.named_parameters():
            #     para.requires_grad = False
            # self.pretrained_monodepth_model.eval()
            mono_feature_dims = {
                "vits": 384, # 2, 5, 8, 
                "vitb": 768, # 2, 5, 8, 
                "vitl": 1024, # 4, 11, 17, 
            }
            self.rgbd_fusion = nn.Sequential(
                nn.Linear(mono_feature_dims[cfg.vit_type] + cfg.d_feature, cfg.d_feature, bias=False),
                # nn.GroupNorm(8, cfg.d_feature),
                nn.LayerNorm(cfg.d_feature),
                nn.ReLU(),
                nn.Linear(cfg.d_feature, cfg.d_feature, bias=False),
            )

        self.fusion_type = "cee"
        Fusion_dict = {"cat": Concat,
                "biproj": BiProj,
                "cee": CEELayer}
        FusionLayer = Fusion_dict[self.fusion_type]
        self.fuse1 = FusionLayer(cfg.d_feature, SE=True)
        self.fuse2 = FusionLayer(cfg.d_feature, SE=True)
        if not cfg.wo_erp_encoder:
            self.backbone_erp = BackboneMultiview(
                feature_channels=cfg.d_feature,
                downscale_factor=cfg.downscale_factor,
                no_cross_attn=cfg.wo_backbone_cross_attn,
                use_epipolar_trans=cfg.use_epipolar_trans,
            )
        ckpt_path = cfg.unimatch_weights_path
        if get_cfg().mode == 'train':
            if cfg.unimatch_weights_path is None:
                print("==> Init multi-view transformer backbone from scratch")
            else:
                print("==> Load multi-view transformer backbone checkpoint: %s" % ckpt_path)
                unimatch_pretrained_model = torch.load(ckpt_path)["model"]
                # NOTE: when wo cross attn, we added ffns into self-attn, but they have no pretrained weight
                is_strict_loading = not cfg.wo_backbone_cross_attn
                # import pdb;pdb.set_trace()
                if not self.cfg.wo_cube_encoder:
                    updated_state_dict = OrderedDict(
                        {
                            k: v
                            for k, v in unimatch_pretrained_model.items()
                            if k in self.backbone.state_dict()
                        }
                    )                
                
                    try:
                        self.backbone.load_state_dict(updated_state_dict, strict=is_strict_loading)
                        # self.backbone_erp.load_state_dict(updated_state_dict, strict=is_strict_loading)
                    except:
                        print("loading pretrained model failed.")
                        pass
                if not self.cfg.wo_erp_encoder:                    
                    updated_state_dict = OrderedDict(
                        {
                            k: v
                            for k, v in unimatch_pretrained_model.items()
                            if k in self.backbone_erp.state_dict()
                        }
                    )  
                    try:
                        # self.backbone.load_state_dict(updated_state_dict, strict=is_strict_loading)
                        self.backbone_erp.load_state_dict(updated_state_dict, strict=is_strict_loading)
                    except:
                        print("loading pretrained model failed.")
                        pass


        

        # gaussians convertor
        self.gaussian_adapter_erp = GaussianAdapterERP(cfg.gaussian_adapter)

        # # cost volume based depth predictor
        # self.depth_predictor = DepthPredictorMultiView(
        #     feature_channels=cfg.d_feature,
        #     upscale_factor=cfg.downscale_factor,
        #     num_depth_candidates=cfg.num_depth_candidates,
        #     costvolume_unet_feat_dim=cfg.costvolume_unet_feat_dim,
        #     costvolume_unet_channel_mult=tuple(cfg.costvolume_unet_channel_mult),
        #     costvolume_unet_attn_res=tuple(cfg.costvolume_unet_attn_res),
        #     gaussian_raw_channels=cfg.num_surfaces * (self.gaussian_adapter.d_in + 2),
        #     gaussians_per_pixel=cfg.gaussians_per_pixel,
        #     num_views=get_cfg().dataset.view_sampler.num_context_views,
        #     depth_unet_feat_dim=cfg.depth_unet_feat_dim,
        #     depth_unet_attn_res=cfg.depth_unet_attn_res,
        #     depth_unet_channel_mult=cfg.depth_unet_channel_mult,
        #     wo_depth_refine=cfg.wo_depth_refine,
        #     wo_cost_volume=cfg.wo_cost_volume,
        #     wo_cost_volume_refine=cfg.wo_cost_volume_refine,
        #     use_cross_view_self_attn=cfg.use_cross_view_self_attn,
        # )

        # cost volume based depth predictor
        # w/o erp decoder
        self.depth_predictor_erp = DepthPredictorMultiView360(
            feature_channels=cfg.d_feature,
            upscale_factor=cfg.downscale_factor,
            num_depth_candidates=cfg.num_depth_candidates,
            costvolume_unet_feat_dim=cfg.costvolume_unet_feat_dim,
            costvolume_unet_channel_mult=tuple(cfg.costvolume_unet_channel_mult),
            costvolume_unet_attn_res=tuple(cfg.costvolume_unet_attn_res),
            gaussian_raw_channels=cfg.num_surfaces * (self.gaussian_adapter_erp.d_in + 2),
            gaussians_per_pixel=cfg.gaussians_per_pixel,
            num_views=get_cfg().dataset.view_sampler.num_context_views,
            depth_unet_feat_dim=cfg.depth_unet_feat_dim,
            depth_unet_attn_res=cfg.depth_unet_attn_res,
            depth_unet_channel_mult=cfg.depth_unet_channel_mult,
            wo_depth_refine=cfg.wo_depth_refine,
            wo_cost_volume=cfg.wo_cost_volume,
            wo_cost_volume_refine=cfg.wo_cost_volume_refine,
            use_cross_view_self_attn=cfg.use_cross_view_self_attn,
        )

    def map_pdf_to_opacity(
        self,
        pdf: Float[Tensor, " *batch"],
        global_step: int,
    ) -> Float[Tensor, " *batch"]:
        # https://www.desmos.com/calculator/opvwti3ba9

        # Figure out the exponent.
        cfg = self.cfg.opacity_mapping
        x = cfg.initial + min(global_step / cfg.warm_up, 1) * (cfg.final - cfg.initial)
        exponent = 2**x

        # Map the probability density to an opacity.
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))

    def normalize_images(self, x):
        device = x.device
        std = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype, device=device).view(
            1, 3, 1, 1
        )
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype, device=device).view(
            1, 3, 1, 1
        )
        x = (x - mean) / std
        return x

    def forward(
        self,
        context: dict,
        # dataset_name: str,
        global_step: int,
        deterministic: bool = False,
        visualization_dump: Optional[dict] = None,
        scene_names: Optional[list] = None,
    ) -> tuple[Gaussians, Float[Tensor, "b v h w"]]:


        device = context["image_sphere"].device
        b, v, _, h, w = context["image_sphere"].shape

        cubes = 6
        if self.cfg.add_mono_feat:
            # mono feature
            normalized_images = self.normalize_images(context["image_cubes_input"])
            images_mono = rearrange(normalized_images, "b v cubes c h w -> (b v cubes) c h w")
            orig_h, orig_w = images_mono.shape[-2:]
            new_h, new_w = orig_h // 14 * 14, orig_w // 14 * 14
            images_mono = F.interpolate(
                images_mono, (new_h, new_w), mode="bilinear", align_corners=True
            )
            # get features at intermediate layers
            last_layer_idx = {
                "vits": [11], # 2, 5, 8, 
                "vitb": [11], # 2, 5, 8, 
                "vitl": [23], # 4, 11, 17, 
            }
            
            with torch.no_grad():
                features_mono = self.pretrained.get_intermediate_layers(
                    images_mono, last_layer_idx[self.cfg.vit_type], return_class_token=False
                )
                features_mono = features_mono[-1]
                features_mono = features_mono.reshape(b*v*cubes, new_h // 14, new_w//14, -1)
                features_mono = rearrange(features_mono, "bvc  h w c -> bvc c h w")

                # reshape to target resolution
                features_mono = F.interpolate(features_mono, (orig_h // self.cfg.downscale_factor, orig_w // self.cfg.downscale_factor), mode='bilinear', align_corners=True)
                features_mono = rearrange(features_mono, "(b v cubes) c h w -> (b v) c h (cubes w)", v=v, cubes=cubes)
                # cube_enc_feat4 = torch.cat(torch.split(cube_enc_feat4, input_equi_image.shape[0], dim=0), dim=-1)
                features_mono = self.c2e(features_mono)

        epipolar_kwargs = {}
        if not self.cfg.wo_cube_encoder:
            trans_features, cnn_features = self.backbone(
                rearrange(context["image_cubes_input"], "b v cubes c h w -> b (v cubes) c h w"),
                attn_splits=self.cfg.multiview_trans_attn_split,
                return_cnn_features=True,
                epipolar_kwargs=epipolar_kwargs,
            )
            # convert trans_features from cubemaps to erp
            cnn_features = rearrange(cnn_features, "b (v cubes) c h w -> (b v) c h (cubes w)", v=v, cubes=cubes)
            cnn_features = self.c2e(cnn_features)

            trans_features = rearrange(trans_features, "b (v cubes) c h w -> (b v) c h (cubes w)", v=v, cubes=cubes)
            trans_features = self.c2e(trans_features)
        else:
            # trans_features = torch.zeros((b*v, self.cfg.d_feature, h, w)).to(device)
            # cnn_features = torch.zeros((b*v, self.cfg.d_feature, h, w)).to(device)
            trans_features = torch.zeros((b*v, self.cfg.d_feature, h // self.cfg.downscale_factor, w // self.cfg.downscale_factor)).to(device)
            cnn_features = torch.zeros((b*v, self.cfg.d_feature, h // self.cfg.downscale_factor, w // self.cfg.downscale_factor)).to(device)

        if not self.cfg.wo_erp_encoder:        
            trans_features_erp, cnn_features_erp = self.backbone_erp(
                context["image_sphere"],
                attn_splits=self.cfg.multiview_trans_attn_split,
                return_cnn_features=True,
                epipolar_kwargs=epipolar_kwargs,
            )
        else:
            trans_features_erp = torch.zeros((b, v, self.cfg.d_feature, h // self.cfg.downscale_factor, w // self.cfg.downscale_factor)).to(device)
            cnn_features_erp = torch.zeros((b, v, self.cfg.d_feature, h // self.cfg.downscale_factor, w // self.cfg.downscale_factor)).to(device)


        # if self.debug:
        #     import numpy as np
        #     import cv2
        #     image_cubes = rearrange(context["image_cubes_input"], "b v cubes c h w -> (b v) c h (cubes w)")
        #     image_c2e = self.c2e_debug(image_cubes)

        #     orig = np.uint8(rearrange(context['image_sphere'], "b v c h w -> (b v) h w c").data.cpu().numpy() * 255)
        #     c2e = np.uint8(rearrange(image_c2e, "bv c h w -> bv h w c").data.cpu().numpy() * 255)

        #     cv2.imwrite("ori_0.png", orig[0])
        #     cv2.imwrite("c2e_0.png", c2e[0])

        #     cv2.imwrite("ori_1.png", orig[1])
        #     cv2.imwrite("c2e_1.png", c2e[1])
        #     import pdb;pdb.set_trace()


        # merge RGB transformer features and monocular features
        if self.cfg.add_mono_feat: 
            # import pdb;pdb.set_trace()
            trans_features = torch.cat([trans_features, features_mono], dim=1) # (b v) c h w
            # import pdb;pdb.set_trace() # 
            trans_features = self.rgbd_fusion(rearrange(trans_features, "bv c h w -> bv h w c"))
            trans_features = rearrange(trans_features, "bv h w c -> bv c h w")
        else:
            pass # torch.zeros((b, v, self.cfg.d_feature, h // self.cfg.downscale_factor, w // self.cfg.downscale_factor)).to(device)

        # Fuse
        trans_features_erp = self.fuse1(rearrange(trans_features_erp, "b v ... -> (b v) ..."), trans_features)
        cnn_features_erp = self.fuse2(rearrange(cnn_features_erp, "b v ... -> (b v) ..."), cnn_features)

        # fused features 
        trans_features = rearrange(trans_features_erp, "(b v) ... -> b v ...", b=b, v=v)
        cnn_features = rearrange(cnn_features_erp, "(b v) ... -> b v ...", b=b, v=v)

        in_feats = trans_features
        extra_info = {}
        extra_info['images_sphere'] = rearrange(context["image_sphere"], "b v c h w -> (v b) c h w")
        extra_info["scene_names"] = scene_names
        debug = True
        if debug:     
            extra_info['gt_depth'] = context["depth_sphere"]
            extra_info['gt_rgb'] = context['image_sphere']
        else:
            extra_info['gt_depth'] = None
            extra_info['gt_rgb'] = None # context['image_sphere']

        gpp = self.cfg.gaussians_per_pixel

        feature_util_config = {
            "dataset_name": self.dataset_name,
            "batch_size": context["image_sphere"].shape[0],
            "height": h // self.cfg.downscale_factor,
            "width": w // self.cfg.downscale_factor,
        }
        feature_utils360 = Utils(feature_util_config)

        depths, densities, raw_gaussians = self.depth_predictor_erp(
            feature_utils360,
            in_feats,
            # context["intrinsics"],
            context["extrinsics_sphere"],
            context["near"],
            context["far"],
            gaussians_per_pixel=gpp,
            deterministic=deterministic,
            extra_info=extra_info,
            cnn_features=cnn_features,
            depth_sampling_type=self.cfg.depth_sampling_type,
        )

        # # Convert the features and depths into Gaussians.
        # xy_ray, _ = sample_image_grid((h, w), device)
        # xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
        gaussians = rearrange(
            raw_gaussians,
            "... (srf c) -> ... srf c",
            srf=self.cfg.num_surfaces,
        )
        # offset_xy = gaussians[..., :2].sigmoid()
        # pixel_size = 1 / torch.tensor((w, h), dtype=torch.float32, device=device)
        # xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size
        gpp = self.cfg.gaussians_per_pixel
        gaussians = self.gaussian_adapter_erp.forward(
            self.dataset_name,
            rearrange(context["extrinsics_sphere"], "b v i j -> b v () () () i j"),
            # rearrange(context["intrinsics_sphere"], "b v i j -> b v () () () i j"),
            # rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
            depths,
            self.map_pdf_to_opacity(densities, global_step) / gpp,
            rearrange(
                gaussians[..., 2:],
                "b v r srf c -> b v r srf () c",
            ),
            (h, w),
        )
        
        # # cubemaps
        # # Sample depths from the resulting features.
        # in_feats = trans_features
        # extra_info = {}
        # extra_info['images'] = rearrange(context["image"], "b v c h w -> (v b) c h w")
        # extra_info["scene_names"] = scene_names
        # gpp = self.cfg.gaussians_per_pixel
        # depths, densities, raw_gaussians = self.depth_predictor(
        #     in_feats,
        #     context["intrinsics"],
        #     context["extrinsics"],
        #     context["near"],
        #     context["far"],
        #     gaussians_per_pixel=gpp,
        #     deterministic=deterministic,
        #     extra_info=extra_info,
        #     cnn_features=cnn_features,
        # )

        # # Convert the features and depths into Gaussians.
        # xy_ray, _ = sample_image_grid((h, w), device)
        # xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
        # gaussians = rearrange(
        #     raw_gaussians,
        #     "... (srf c) -> ... srf c",
        #     srf=self.cfg.num_surfaces,
        # )
        
        # offset_xy = gaussians[..., :2].sigmoid()
        # pixel_size = 1 / torch.tensor((w, h), dtype=torch.float32, device=device)
        # xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size
        # gpp = self.cfg.gaussians_per_pixel
        
        # gaussians = self.gaussian_adapter.forward(
        #     rearrange(context["extrinsics"], "b v i j -> b v () () () i j"),
        #     rearrange(context["intrinsics"], "b v i j -> b v () () () i j"),
        #     rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
        #     depths,
        #     self.map_pdf_to_opacity(densities, global_step) / gpp,
        #     rearrange(
        #         gaussians[..., 2:],
        #         "b v r srf c -> b v r srf () c",
        #     ),
        #     (h, w),
        # )


        # Dump visualizations if needed.
        if visualization_dump is not None:
            visualization_dump["depth"] = rearrange(
                depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w
            )
            visualization_dump["scales"] = rearrange(
                gaussians.scales, "b v r srf spp xyz -> b (v r srf spp) xyz"
            )
            visualization_dump["rotations"] = rearrange(
                gaussians.rotations, "b v r srf spp xyzw -> b (v r srf spp) xyzw"
            )

        # Optionally apply a per-pixel opacity.
        opacity_multiplier = 1

        gaussians = Gaussians(
            rearrange(
                gaussians.means,
                "b v r srf spp xyz -> b (v r srf spp) xyz",
            ),
            rearrange(
                gaussians.covariances,
                "b v r srf spp i j -> b (v r srf spp) i j",
            ),
            rearrange(
                gaussians.harmonics,
                "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
            ),
            rearrange(
                opacity_multiplier * gaussians.opacities,
                "b v r srf spp -> b (v r srf spp)",
            ),
        )
        return gaussians, rearrange(depths, "b v (h w) () ()-> b v h w", h=h, w=w)

    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            # pass
        #     # batch = apply_patch_shim(
        #     #     batch,
        #     #     patch_size=self.cfg.shim_patch_size
        #     #     * self.cfg.downscale_factor,
        #     # )

        #     # if self.cfg.apply_bounds_shim:
        #     #     _, _, _, h, w = batch["context"]["image"].shape
        #     #     near_disparity = self.cfg.near_disparity * min(h, w)
        #     #     batch = apply_bounds_shim(batch, near_disparity, self.cfg.far_disparity)

            return batch

        return data_shim

    @property
    def sampler(self):
        # hack to make the visualizer work
        return None
