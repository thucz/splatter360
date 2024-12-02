import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from ..backbone.unimatch.geometry import coords_grid
from .ldm_unet.unet import UNetModel


# def warp_with_pose_depth_candidates(
#     feature1,
#     intrinsics,
#     pose,
#     depth,
#     clamp_min_depth=1e-3,
#     warp_padding_mode="zeros",
# ):
#     """
#     feature1: [B, C, H, W]
#     intrinsics: [B, 3, 3]
#     pose: [B, 4, 4]
#     depth: [B, D, H, W]
#     """

#     assert intrinsics.size(1) == intrinsics.size(2) == 3
#     assert pose.size(1) == pose.size(2) == 4
#     assert depth.dim() == 4

#     b, d, h, w = depth.size()
#     c = feature1.size(1)

#     with torch.no_grad():
#         # pixel coordinates
#         grid = coords_grid(
#             b, h, w, homogeneous=True, device=depth.device
#         )  # [B, 3, H, W]
#         # back project to 3D and transform viewpoint
#         points = torch.inverse(intrinsics).bmm(grid.view(b, 3, -1))  # [B, 3, H*W]
#         points = torch.bmm(pose[:, :3, :3], points).unsqueeze(2).repeat(
#             1, 1, d, 1
#         ) * depth.view(
#             b, 1, d, h * w
#         )  # [B, 3, D, H*W]
#         points = points + pose[:, :3, -1:].unsqueeze(-1)  # [B, 3, D, H*W]
#         # reproject to 2D image plane
#         points = torch.bmm(intrinsics, points.view(b, 3, -1)).view(
#             b, 3, d, h * w
#         )  # [B, 3, D, H*W]
#         pixel_coords = points[:, :2] / points[:, -1:].clamp(
#             min=clamp_min_depth
#         )  # [B, 2, D, H*W]

#         # normalize to [-1, 1]
#         x_grid = 2 * pixel_coords[:, 0] / (w - 1) - 1
#         y_grid = 2 * pixel_coords[:, 1] / (h - 1) - 1

#         grid = torch.stack([x_grid, y_grid], dim=-1)  # [B, D, H*W, 2]

#     # sample features
#     warped_feature = F.grid_sample(
#         feature1,
#         grid.view(b, d * h, w, 2),
#         mode="bilinear",
#         padding_mode=warp_padding_mode,
#         align_corners=True,
#     ).view(
#         b, c, d, h, w
#     )  # [B, C, D, H, W]

#     return warped_feature


def warp_with_pose_depth_candidates(
    utils360,
    feature1,
    pose,
    depth,
    clamp_min_depth=1e-3,
    warp_padding_mode="zeros",
    debug=False,
    gt_rgb1=None,
    gt_depth0=None,
    gt_rgb0=None,
):
    """
    feature1: [B, C, H, W]
    intrinsics: [B, 3, 3]
    pose: [B, 4, 4]
    depth: [B, D, H, W]
    """
    if debug:
        b, d, h, w = gt_depth0.size()
        orig_w, orig_h = utils360.width, utils360.height
        utils360.width = w
        utils360.height = h

        # gt_depth0 = F.interpolate(gt_depth0, (h//8, w//8), mode="bilinear")
        # gt_rgb1 = F.interpolate(gt_rgb1, (h//8, w//8), mode="bilinear")
        # h = h // 8
        # w = w // 8
        
        c = gt_rgb1.size(1)
        with torch.no_grad():
            xy_locs = utils360.get_xy_coords()
            spherical_coords = utils360.equi_2_spherical(xy_locs, radius=1)
            spherical_coords = repeat(spherical_coords, "() h w c -> n c h w", n=gt_depth0.shape[0])
            num_depth_candidates = gt_depth0.shape[1]
            spherical_coords = repeat(spherical_coords, "b c h w -> b d h w c", d=num_depth_candidates).to(gt_depth0.device)
            spherical_coords[..., 2] = gt_depth0

            points = utils360.spherical_2_cartesian(spherical_coords)            
            points = rearrange(points, "b d h w c -> b d c (h w)")
            pose_reshape = repeat(pose, "b r1 r2 -> b d r1 r2", d=d)
            # points = torch.bmm(pose[..., :3, :3], points) + pose[..., :3, -1:].unsqueeze(-1)  # [B, 3, D, H*W]
            points = pose_reshape[..., :3, :3] @ points + pose_reshape[..., :3, -1:]  # [B, 3, D, H*W]
            points = rearrange(points, "b d c n -> b d n c")
            spherical_coords_warped = utils360.cartesian_2_spherical(points)
            equi_warped = utils360.spherical_2_equi(spherical_coords_warped)

            u = equi_warped[..., 0]
            v = equi_warped[..., 1]

            u = (u + 0.5) / w
            v = (v + 0.5) / h

            u = u * 2.0 - 1.0
            v = v * 2.0 - 1.0

            assert torch.logical_and(torch.logical_and(u >= -1, u <= 1), torch.logical_and(v >= -1, v <= 1)).all(),"Wrong UV mapping, UV must be in [-1, 1]!"
            grid = torch.stack([u, v], dim=-1) # b d h*w 2
            warped_rgb = F.grid_sample(
                gt_rgb1.contiguous(),
                grid.view(b, d * h, w, 2).contiguous(),
                mode="bilinear",
                padding_mode=warp_padding_mode,
                align_corners=True,
            ).view(
                b, c, d, h, w
            )  # [B, C, D, H, W]
            # visualize rgb0 & depth0

            # visualize rgb1 & depth1
            import os, cv2
            import numpy as np
            os.makedirs("./debug_warp/", exist_ok=True)
            batch_idx = 0
            warped_rgb_np = np.uint8(rearrange(warped_rgb, "b c () h w -> b h w c").data.cpu().numpy()[batch_idx]*255)
            cv2.imwrite("./debug_warp/warped_rgb.png", warped_rgb_np)

            gt_rgb1_np = np.uint8(rearrange(gt_rgb1, "b c h w -> b h w c").data.cpu().numpy()[batch_idx]*255)
            cv2.imwrite("./debug_warp/gt_rgb1.png", gt_rgb1_np)

            gt_rgb0_np = np.uint8(rearrange(gt_rgb0, "b c h w -> b h w c").data.cpu().numpy()[batch_idx]*255)
            cv2.imwrite("./debug_warp/gt_rgb0.png", gt_rgb0_np)
            print("pose relative translation:", pose_reshape[batch_idx, 0, :3, 3])

        utils360.width, utils360.height = orig_w, orig_h

    assert pose.size(1) == pose.size(2) == 4
    assert depth.dim() == 4
    b, d, h, w = depth.size()
    c = feature1.size(1)
    with torch.no_grad():
        xy_locs = utils360.get_xy_coords()
        
        spherical_coords = utils360.equi_2_spherical(xy_locs, radius=1)
        spherical_coords = repeat(spherical_coords, "() h w c -> n c h w", n=depth.shape[0])
        num_depth_candidates = depth.shape[1]
        spherical_coords = repeat(spherical_coords, "b c h w -> b d h w c", d=num_depth_candidates).to(depth.device)
        spherical_coords[..., 2] = depth

        points = utils360.spherical_2_cartesian(spherical_coords)
        
        points = rearrange(points, "b d h w c -> b d c (h w)")

        pose = repeat(pose, "b r1 r2 -> b d r1 r2", d=d)
        # points = torch.bmm(pose[..., :3, :3], points) + pose[..., :3, -1:].unsqueeze(-1)  # [B, 3, D, H*W]
        points = pose[..., :3, :3] @ points + pose[..., :3, -1:]  # [B, 3, D, H*W]
        points = rearrange(points, "b d c n -> b d n c")


        spherical_coords_warped = utils360.cartesian_2_spherical(points)
        equi_warped = utils360.spherical_2_equi(spherical_coords_warped)

        u = equi_warped[..., 0]
        v = equi_warped[..., 1]

        u = (u + 0.5) / w
        v = (v + 0.5) / h

        u = u * 2.0 - 1.0
        v = v * 2.0 - 1.0
        assert torch.logical_and(torch.logical_and(u >= -1, u <= 1), torch.logical_and(v >= -1, v <= 1)).all(),"Wrong UV mapping, UV must be in [-1, 1]!"
        grid = torch.stack([u, v], dim=-1) # b d h*w 2
        
    try:
        warped_feature = F.grid_sample(
            feature1.contiguous(),
            grid.view(b, d * h, w, 2).contiguous(),
            mode="bilinear",
            padding_mode=warp_padding_mode,
            align_corners=True,
        ).view(
            b, c, d, h, w
        )  # [B, C, D, H, W]
    except:
        warped_feature = F.grid_sample(feature1, grid.view(b, d * h, w, 2).contiguous(),  mode="bilinear",padding_mode=warp_padding_mode,align_corners=False,)
        warped_feature = warped_feature.view(
            b, c, d, h, w
        ) 
        # padding_mode=warp_padding_mode,
            # align_corners=True,
        # )
    return warped_feature

# def prepare_feat_proj_data_lists(
#     features, intrinsics, extrinsics, near, far, num_samples
# ):
#     # prepare features
#     b, v, _, h, w = features.shape

#     feat_lists = []
#     pose_curr_lists = []
#     init_view_order = list(range(v))
#     feat_lists.append(rearrange(features, "b v ... -> (v b) ..."))  # (vxb c h w)
#     for idx in range(1, v):
#         cur_view_order = init_view_order[idx:] + init_view_order[:idx]

#         cur_feat = features[:, cur_view_order]
#         feat_lists.append(rearrange(cur_feat, "b v ... -> (v b) ..."))  # (vxb c h w)

#         # calculate reference pose
#         # NOTE: not efficient, but clearer for now
#         if v > 2:
#             cur_ref_pose_to_v0_list = []
#             for v0, v1 in zip(init_view_order, cur_view_order):
#                 cur_ref_pose_to_v0_list.append(
#                     extrinsics[:, v1].clone().detach().inverse()
#                     @ extrinsics[:, v0].clone().detach()
#                 )
#             cur_ref_pose_to_v0s = torch.cat(cur_ref_pose_to_v0_list, dim=0)  # (vxb c h w)
#             pose_curr_lists.append(cur_ref_pose_to_v0s)
    
#     # get 2 views reference pose
#     # NOTE: do it in such a way to reproduce the exact same value as reported in paper
#     if v == 2:
#         pose_ref = extrinsics[:, 0].clone().detach()
#         pose_tgt = extrinsics[:, 1].clone().detach()
#         pose = pose_tgt.inverse() @ pose_ref
#         pose_curr_lists = [torch.cat((pose, pose.inverse()), dim=0),]

#     # unnormalized camera intrinsic
#     intr_curr = intrinsics[:, :, :3, :3].clone().detach()  # [b, v, 3, 3]
#     intr_curr[:, :, 0, :] *= float(w)
#     intr_curr[:, :, 1, :] *= float(h)
#     intr_curr = rearrange(intr_curr, "b v ... -> (v b) ...", b=b, v=v)  # [vxb 3 3]

#     if depth_sampling_type == "inverse_depth":
#         # prepare depth bound (inverse depth) [v*b, d]
#         min_depth = rearrange(1.0 / far.clone().detach(), "b v -> (v b) 1")
#         max_depth = rearrange(1.0 / near.clone().detach(), "b v -> (v b) 1")
#         depth_candi_curr = (
#             min_depth
#             + torch.linspace(0.0, 1.0, num_samples).unsqueeze(0).to(min_depth.device)
#             * (max_depth - min_depth)
#         ).type_as(features)
#         depth_candi_curr = repeat(depth_candi_curr, "vb d -> vb d () ()")  # [vxb, d, 1, 1]
#     elif depth_sampling_type == "log_depth":
#         d_min = rearrange(far.clone().detach(), "b v -> (v b) 1")
#         d_max = rearrange(near.clone().detach(), "b v -> (v b) 1") 
#         # depth_candi_curr = (
#         #     min_depth
#         #     + torch.linspace(0.0, 1.0, num_samples).unsqueeze(0).to(min_depth.device)
#         #     * (max_depth - min_depth)
#         # ).type_as(features)

#         # map from [0, 1] to [d_min, d_max]
#         linear_depth = d_min + depth_array * (d_max - d_min)

#         # adjust depth value with log function
#         log_depth = d_min + (torch.log2(linear_depth - d_min + 1) / torch.log2(d_max - d_min + 1)) * (d_max - d_min)

def prepare_feat_proj_data_lists_360(
    features, extrinsics_sphere, near, far, num_samples, depth_sampling_type, gt_rgbs=None, gt_depths=None
):
    # prepare features
    b, v, _, h, w = features.shape
    feat_lists = []
    pose_curr_lists = []
    init_view_order = list(range(v))
    feat_lists.append(rearrange(features, "b v ... -> (v b) ..."))  # (v*b c h w)
    if gt_depths is not None and gt_rgbs is not None:
        rgb_lists = []
        depth_lists = []

        rgb_lists.append(rearrange(gt_rgbs, "b v ... -> (v b) ...")) # (v*b 3 h w)
        depth_lists.append(rearrange(gt_depths, "b v ... -> (v b) ...")) # (v*b 3 h w)

    for idx in range(1, v):
        cur_view_order = init_view_order[idx:] + init_view_order[:idx]
        cur_feat = features[:, cur_view_order]

        feat_lists.append(rearrange(cur_feat, "b v ... -> (v b) ..."))  # (v*b c h w)
        
        if gt_depths is not None and gt_rgbs is not None:
            cur_rgb = gt_rgbs[:, cur_view_order]
            rgb_lists.append(rearrange(cur_rgb, "b v ... -> (v b) ...")) # (v*b, c h w)
            cur_depth = gt_depths[:, cur_view_order]
            depth_lists.append(rearrange(cur_depth, "b v ... -> (v b) ...")) # (v*b, c h w)

        # calculate reference pose
        # NOTE: not efficient, but clearer for now
        if v > 2:
            cur_ref_pose_to_v0_list = []
            for v0, v1 in zip(init_view_order, cur_view_order):
                cur_ref_pose_to_v0_list.append(
                    extrinsics_sphere[:, v1].clone().detach().inverse()
                    @ extrinsics_sphere[:, v0].clone().detach()
                )
            cur_ref_pose_to_v0s = torch.cat(cur_ref_pose_to_v0_list, dim=0)  # (v*b c h w)
            pose_curr_lists.append(cur_ref_pose_to_v0s)
    
    # get 2 views reference pose
    # NOTE: do it in such a way to reproduce the exact same value as reported in paper
    if v == 2:
        pose_ref = extrinsics_sphere[:, 0].clone().detach()
        pose_tgt = extrinsics_sphere[:, 1].clone().detach()
        pose = pose_tgt.inverse() @ pose_ref
        pose_curr_lists = [torch.cat((pose, pose.inverse()), dim=0),]

    # unnormalized camera intrinsic
    # intr_curr = intrinsics[:, :, :3, :3].clone().detach()  # [b, v, 3, 3]
    # intr_curr[:, :, 0, :] *= float(w)
    # intr_curr[:, :, 1, :] *= float(h)
    # intr_curr = rearrange(intr_curr, "b v ... -> (v b) ...", b=b, v=v)  # [v*b 3 3]
    # prepare depth bound (inverse depth) [v*b, d]

    # depth_sampling_type = 
    if depth_sampling_type == "inverse_depth":
        # prepare depth bound (inverse depth) [v*b, d]
        min_depth = rearrange(1.0 / far.clone().detach(), "b v -> (v b) 1")
        max_depth = rearrange(1.0 / near.clone().detach(), "b v -> (v b) 1")
        depth_candi_curr = (
            min_depth
            + torch.linspace(0.0, 1.0, num_samples).unsqueeze(0).to(min_depth.device)
            * (max_depth - min_depth)
        ).type_as(features)     
        depth_candi_curr = repeat(depth_candi_curr, "vb d -> vb d () ()")  # [vxb, d, 1, 1]
        depth_candi_curr = 1 / depth_candi_curr
    elif depth_sampling_type == "log_depth":
        d_min = rearrange(near.clone().detach(), "b v -> (v b) 1")
        d_max = rearrange(far.clone().detach(), "b v -> (v b) 1") 
        log_d_min = torch.log(d_min)
        log_d_max = torch.log(d_max)
        # log_samples = torch.linspace(log_d_min, log_d_max, num_samples)
        log_samples = log_d_min + torch.linspace(0.0, 1.0, num_samples).unsqueeze(0).to(d_min.device) \
            * (log_d_max - log_d_min)
        depth_candi_curr = torch.exp(log_samples)
        depth_candi_curr = repeat(depth_candi_curr, "vb d -> vb d () ()")

    elif depth_sampling_type == "linear_depth": # uniform
        # prepare depth bound (inverse depth) [v*b, d]
        min_depth = rearrange(near.clone().detach(), "b v -> (v b) 1")
        max_depth = rearrange(far.clone().detach(), "b v -> (v b) 1")
        depth_candi_curr = (
            min_depth
            + torch.linspace(0.0, 1.0, num_samples).unsqueeze(0).to(min_depth.device)
            * (max_depth - min_depth)
        ).type_as(features)
        depth_candi_curr = repeat(depth_candi_curr, "vb d -> vb d () ()")  # [vxb, d, 1, 1]
        # depth_candi_curr = 1 / depth_candi_curr
    else:
        raise NotImplementedError
    if gt_depths is not None and gt_rgbs is not None:
        return feat_lists, pose_curr_lists, depth_candi_curr, rgb_lists, depth_lists
    else:
        return feat_lists, pose_curr_lists, depth_candi_curr




class DepthPredictorMultiView360(nn.Module):
    """IMPORTANT: this model is in (v b), NOT (b v), due to some historical issues.
    keep this in mind when performing any operation related to the view dim"""

    def __init__(
        self,
        feature_channels=128,
        upscale_factor=4,
        num_depth_candidates=32,
        costvolume_unet_feat_dim=128,
        costvolume_unet_channel_mult=(1, 1, 1),
        costvolume_unet_attn_res=(),
        gaussian_raw_channels=-1,
        gaussians_per_pixel=1,
        num_views=2,
        depth_unet_feat_dim=64,
        depth_unet_attn_res=(),
        depth_unet_channel_mult=(1, 1, 1),
        wo_depth_refine=False,
        wo_cost_volume=False,
        wo_cost_volume_refine=False,
        use_cross_view_self_attn=True,
        **kwargs,
    ):
        super(DepthPredictorMultiView360, self).__init__()
        self.num_depth_candidates = num_depth_candidates
        self.regressor_feat_dim = costvolume_unet_feat_dim
        self.upscale_factor = upscale_factor
        # ablation settings
        # Table 3: base
        self.wo_depth_refine = wo_depth_refine
        # Table 3: w/o cost volume
        self.wo_cost_volume = wo_cost_volume
        # Table 3: w/o U-Net
        self.wo_cost_volume_refine = wo_cost_volume_refine

        # Cost volume refinement: 2D U-Net
        input_channels = feature_channels if wo_cost_volume else (num_depth_candidates + feature_channels)
        channels = self.regressor_feat_dim
        if wo_cost_volume_refine:
            self.corr_project = nn.Conv2d(input_channels, channels, 3, 1, 1)
        else:
            modules = [
                nn.Conv2d(input_channels, channels, 3, 1, 1),
                nn.GroupNorm(8, channels),
                nn.GELU(),
                UNetModel(
                    image_size=None,
                    in_channels=channels,
                    model_channels=channels,
                    out_channels=channels,
                    num_res_blocks=1,
                    attention_resolutions=costvolume_unet_attn_res,
                    channel_mult=costvolume_unet_channel_mult,
                    num_head_channels=32,
                    dims=2,
                    postnorm=True,
                    num_frames=num_views,
                    use_cross_view_self_attn=use_cross_view_self_attn,
                ),
                nn.Conv2d(channels, num_depth_candidates, 3, 1, 1)
            ]
            self.corr_refine_net = nn.Sequential(*modules)
            # cost volume u-net skip connection
            self.regressor_residual = nn.Conv2d(
                input_channels, num_depth_candidates, 1, 1, 0
            )

        # Depth estimation: project features to get softmax based coarse depth
        self.depth_head_lowres = nn.Sequential(
            nn.Conv2d(num_depth_candidates, num_depth_candidates * 2, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_depth_candidates * 2, num_depth_candidates, 3, 1, 1),
        )

        # CNN-based feature upsampler
        proj_in_channels = feature_channels + feature_channels
        upsample_out_channels = feature_channels
        self.upsampler = nn.Sequential(
            nn.Conv2d(proj_in_channels, upsample_out_channels, 3, 1, 1),
            nn.Upsample(
                scale_factor=upscale_factor,
                mode="bilinear",
                align_corners=True,
            ),
            nn.GELU(),
        )
        self.proj_feature = nn.Conv2d(
            upsample_out_channels, depth_unet_feat_dim, 3, 1, 1
        )

        # Depth refinement: 2D U-Net
        input_channels = 3 + depth_unet_feat_dim + 1 + 1
        channels = depth_unet_feat_dim
        if wo_depth_refine:  # for ablations
            self.refine_unet = nn.Conv2d(input_channels, channels, 3, 1, 1)
        else:
            self.refine_unet = nn.Sequential(
                nn.Conv2d(input_channels, channels, 3, 1, 1),
                nn.GroupNorm(4, channels),
                nn.GELU(),
                UNetModel(
                    image_size=None,
                    in_channels=channels,
                    model_channels=channels,
                    out_channels=channels,
                    num_res_blocks=1, 
                    attention_resolutions=depth_unet_attn_res,
                    channel_mult=depth_unet_channel_mult,
                    num_head_channels=32,
                    dims=2,
                    postnorm=True,
                    num_frames=num_views,
                    use_cross_view_self_attn=use_cross_view_self_attn,
                ),
            )

        # Gaussians prediction: covariance, color
        gau_in = depth_unet_feat_dim + 3 + feature_channels
        self.to_gaussians = nn.Sequential(
            nn.Conv2d(gau_in, gaussian_raw_channels * 2, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(
                gaussian_raw_channels * 2, gaussian_raw_channels, 3, 1, 1
            ),
        )

        # Gaussians prediction: centers, opacity
        if not wo_depth_refine:
            channels = depth_unet_feat_dim
            disps_models = [
                nn.Conv2d(channels, channels * 2, 3, 1, 1),
                nn.GELU(),
                nn.Conv2d(channels * 2, gaussians_per_pixel * 2, 3, 1, 1),
            ]
            self.to_disparity = nn.Sequential(*disps_models)

    def forward(
        self,
        utils360,
        features,
        # intrinsics,
        extrinsics,
        near,
        far,
        gaussians_per_pixel=1,
        deterministic=True,
        extra_info=None,
        cnn_features=None,
        depth_sampling_type="inverse_depth",
    ):
        """IMPORTANT: this model is in (v b), NOT (b v), due to some historical issues.
        keep this in mind when performing any operation related to the view dim"""

        # format the input
        b, v, c, h, w = features.shape
        debug = False
        if debug:
            feat_comb_lists, pose_curr_lists, depth_candi_curr, rgb_comb_lists, gt_depth_comb_lists = (
                prepare_feat_proj_data_lists_360(
                    features,
                    extrinsics,
                    near,
                    far,
                    num_samples=self.num_depth_candidates,
                    depth_sampling_type=depth_sampling_type,
                    gt_rgbs=extra_info["gt_rgb"],
                    gt_depths=extra_info["gt_depth"],

                )
            )
        else:
            feat_comb_lists, pose_curr_lists, depth_candi_curr = (
                prepare_feat_proj_data_lists_360(
                    features,
                    extrinsics,
                    near,
                    far,
                    num_samples=self.num_depth_candidates,
                    depth_sampling_type=depth_sampling_type,
                    gt_rgbs=None,
                    gt_depths=None,

                )
            )

        # feat_comb_lists, intr_curr, pose_curr_lists, disp_candi_curr = (
        #     prepare_feat_proj_data_lists(
        #         features,
        #         intrinsics,
        #         extrinsics,
        #         near,
        #         far,
        #         num_samples=self.num_depth_candidates,
        #     )
        # )

        if cnn_features is not None:
            cnn_features = rearrange(cnn_features, "b v ... -> (v b) ...")

        # cost volume constructions
        feat01 = feat_comb_lists[0]

        if self.wo_cost_volume:
            raw_correlation_in = feat01
        else:
            raw_correlation_in_lists = []
            # for feat10, pose_curr, rgb10 in zip(feat_comb_lists[1:], pose_curr_lists, rgb_comb_lists[1:]):
            for feat10, pose_curr in zip(feat_comb_lists[1:], pose_curr_lists):
                if debug:
                    feat01_warped = warp_with_pose_depth_candidates(
                        utils360,
                        # dataset_name,
                        feat10,
                        # intr_curr,
                        pose_curr,
                        depth_candi_curr.repeat([1, 1, *feat10.shape[-2:]]),
                        warp_padding_mode="zeros",
                        debug=debug,
                        gt_rgb1 = rgb10,
                        gt_depth0 = gt_depth_comb_lists[0],
                        gt_rgb0 = rgb_comb_lists[0],
                    )  # [B, C, D, H, W]
                else:
                    feat01_warped = warp_with_pose_depth_candidates(
                        utils360,
                        # dataset_name,
                        feat10,
                        # intr_curr,
                        pose_curr,
                        depth_candi_curr.repeat([1, 1, *feat10.shape[-2:]]),
                        warp_padding_mode="zeros",
                        # debug=debug,
                        # gt_rgb1 = rgb10,
                        # gt_depth0 = gt_depth_comb_lists[0],
                        # gt_rgb0 = rgb_comb_lists[0],
                    )  # [B, C, D, H, W]
                
                # calculate similarity
                raw_correlation_in = (feat01.unsqueeze(2) * feat01_warped).sum(
                    1
                ) / (
                    c**0.5
                )  # [vB, D, H, W]
                raw_correlation_in_lists.append(raw_correlation_in)
            # average all cost volumes
            raw_correlation_in = torch.mean(
                torch.stack(raw_correlation_in_lists, dim=0), dim=0, keepdim=False
            )  # [vxb d, h, w]
            raw_correlation_in = torch.cat((raw_correlation_in, feat01), dim=1)
        # refine cost volume via 2D u-net
        if self.wo_cost_volume_refine:
            raw_correlation = self.corr_project(raw_correlation_in)
        else:
            raw_correlation = self.corr_refine_net(raw_correlation_in)  # (vb d h w)
            # apply skip connection
            raw_correlation = raw_correlation + self.regressor_residual(
                raw_correlation_in
            )

        # softmax to get coarse depth and density
        pdf = F.softmax(
            self.depth_head_lowres(raw_correlation), dim=1
        )  # [2xB, D, H, W]
        coarse_depths = (depth_candi_curr * pdf).sum(
            dim=1, keepdim=True
        )  # (vb, 1, h, w)
        # safe_divide(coarse_depths)
        coarse_disps = 1 / coarse_depths
        pdf_max = torch.max(pdf, dim=1, keepdim=True)[0]  # argmax
        pdf_max = F.interpolate(pdf_max, scale_factor=self.upscale_factor)
        fullres_disps = F.interpolate(
            coarse_disps,
            scale_factor=self.upscale_factor,
            mode="bilinear",
            align_corners=True,
        )

        # depth refinement
        proj_feat_in_fullres = self.upsampler(torch.cat((feat01, cnn_features), dim=1))
        proj_feature = self.proj_feature(proj_feat_in_fullres)
        refine_out = self.refine_unet(torch.cat(
            (extra_info["images_sphere"], proj_feature, fullres_disps, pdf_max), dim=1
        ))

        # gaussians head
        raw_gaussians_in = [refine_out,
                            extra_info["images_sphere"], proj_feat_in_fullres]
        raw_gaussians_in = torch.cat(raw_gaussians_in, dim=1)
        raw_gaussians = self.to_gaussians(raw_gaussians_in)
        raw_gaussians = rearrange(
            raw_gaussians, "(v b) c h w -> b v (h w) c", v=v, b=b
        )

        if self.wo_depth_refine:
            densities = repeat(
                pdf_max,
                "(v b) dpt h w -> b v (h w) srf dpt",
                b=b,
                v=v,
                srf=1,
            )
            depths = 1.0 / fullres_disps
            depths = repeat(
                depths,
                "(v b) dpt h w -> b v (h w) srf dpt",
                b=b,
                v=v,
                srf=1,
            )
        else:
            # delta fine depth and density
            delta_disps_density = self.to_disparity(refine_out)
            delta_disps, raw_densities = delta_disps_density.split(
                gaussians_per_pixel, dim=1
            )

            # combine coarse and fine info and match shape
            densities = repeat(
                F.sigmoid(raw_densities),
                "(v b) dpt h w -> b v (h w) srf dpt",
                b=b,
                v=v,
                srf=1,
            )

            fine_disps = (fullres_disps + delta_disps).clamp(
                1.0 / rearrange(far, "b v -> (v b) () () ()"),
                1.0 / rearrange(near, "b v -> (v b) () () ()"),
            )
            depths = 1.0 / fine_disps
            depths = repeat(
                depths,
                "(v b) dpt h w -> b v (h w) srf dpt",
                b=b,
                v=v,
                srf=1,
            )

        return depths, densities, raw_gaussians
