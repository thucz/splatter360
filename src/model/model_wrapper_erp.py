from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import moviepy.editor as mpy
import torch
from einops import pack, rearrange, repeat
from jaxtyping import Float
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import rank_zero_only
from torch import Tensor, nn, optim
import numpy as np
import json

from ..dataset.data_module import get_data_shim
from ..dataset.types import BatchedExample
from ..dataset import DatasetCfg
from ..evaluation.metrics import compute_lpips, compute_psnr, compute_ssim
from ..global_cfg import get_cfg
from ..loss import Loss
from ..misc.benchmarker import Benchmarker
from ..misc.image_io import prep_image, save_image, save_video
# from ..misc.LocalLogger import LOG_PATH, LocalLogger
from ..misc.step_tracker import StepTracker
from ..visualization.annotation import add_label
from ..visualization.camera_trajectory.interpolation import (
    interpolate_extrinsics,
    interpolate_intrinsics,
)
from ..visualization.camera_trajectory.wobble import (
    generate_wobble,
    generate_wobble_transformation,
)
from ..visualization.color_map import apply_color_map_to_image
from ..visualization.layout import add_border, hcat, vcat
from ..visualization import layout
from ..visualization.validation_in_3d import render_cameras, render_projections
from ..visualization.camera_trajectory.interpolate_trajectory import interpolate_render_poses_m9d
from .decoder.decoder import Decoder, DepthRenderingMode
from .encoder import Encoder
from .encoder.visualization.encoder_visualizer import EncoderVisualizer
import imageio
from skimage.io import imsave
import torch.nn.functional as F
from .model_wrapper_helper import compute_l1_sphere_loss, erode
from ..geometry.layers import Cube2Equirec #, Concat, BiProj, CEELayer
from ..scripts.compute_depth_metrics import compute_depth_metrics_batched
from ..geometry.z_depth_to_distance import depth_to_distance_map_batch
@dataclass
class OptimizerCfg:
    lr: float
    warm_up_steps: int
    cosine_lr: bool


@dataclass
class TestCfg:
    output_path: Path
    compute_scores: bool
    save_image: bool
    save_video: bool
    eval_time_skip_steps: int

    eval_depth: bool # default is False

@dataclass
class TrainCfg:
    depth_mode: DepthRenderingMode | None
    extended_visualization: bool
    print_log_every_n_steps: int
    wo_depth_supervise: bool


@runtime_checkable
class TrajectoryFn(Protocol):
    def __call__(
        self,
        t: Float[Tensor, " t"],
    ) -> tuple[
        Float[Tensor, "batch view 4 4"],  # extrinsics
        Float[Tensor, "batch view 3 3"],  # intrinsics
    ]:
        pass


import matplotlib as mpl
import matplotlib.cm as cm
def get_colormap(input_data): # H, W, 3
    normalizer = mpl.colors.Normalize(vmin=0., vmax=1.0)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='viridis') # 
    colormapped_im = (mapper.to_rgba(input_data)[:, :, :3] * 255).astype(np.uint8)
    return colormapped_im

def convert_colormap(data):
    import cv2
    rgbs = []
    for idx in range(data.shape[0]):
        rgb = np.uint8(data[idx][0].data.cpu().numpy() * 255)
        # rgb = data[idx].data.cpu().numpy()
        # rgb = get_colormap(rgb)
        rgb = cv2.applyColorMap(rgb, cv2.COLORMAP_RAINBOW)
        rgbs.append(rgb)
    rgbs = np.stack(rgbs)

    rgbs = torch.from_numpy(rgbs).float() / 255.
    rgbs = rearrange(rgbs, "v h w c -> v c h w")
    return rgbs

def convert_single_colormap(data):
    import cv2
    # for idx in range(data.shape[0]):
    # rgb = np.uint8(data[0].data.cpu().numpy() * 255)
    rgb = data[0].data.cpu().numpy()
    rgb = get_colormap(rgb)
    # rgb = cv2.applyColorMap(rgb, cv2.COLORMAP_RAINBOW)
    # rgbs.append(rgb)
    # rgbs = np.stack(rgbs)
    rgb = torch.from_numpy(rgb).float() / 255.
    rgb = rearrange(rgb, "h w c -> c h w")
    return rgb
# Color-map the result.
def depth_map(result):
    try:
        near = result[result > 0][:16_000_000].quantile(0.01).log()
        far = result.view(-1)[:16_000_000].quantile(0.99).log()
        result = result.log()
        result = 1 - (result - near) / (far - near)
    except Exception:
        near = result.min()
        far = result.max()
        result = 1 - (result - near) / (far - near)
    
    return apply_color_map_to_image(result, "turbo")

def change_order(cubes):
    assert len(cubes.shape) == 4
    # [U B L F R D] -> [F R B L U D]
    cubes[0, :, :, :] = torch.flip(cubes[0, :, :, :], dims=[-1, -2])
    cubes[5, :, :, :] = torch.flip(cubes[5, :, :, :], dims=[-1, -2])
    cubes_new = cubes.clone()
    cubes_new[4] = cubes[0]
    cubes_new[2:4] = cubes[1:3]
    cubes_new[0:2] = cubes[3:5]
    # flip along h, w
    return cubes_new

def change_order_batch(cubes):
    # B V C H W or B V H W
    # [U B L F R D] -> [F R B L U D]
    assert len(cubes.shape) == 5 or len(cubes.shape) == 4
    cubes[:, 0, ...] = torch.flip(cubes[:, 0, ...], dims=[-1, -2])
    cubes[:, 5, ...] = torch.flip(cubes[:, 5, ...], dims=[-1, -2])
    cubes_new = cubes.clone()
    cubes_new[:, 4] = cubes[:, 0]
    cubes_new[:, 2:4] = cubes[:, 1:3]
    cubes_new[:, 0:2] = cubes[:, 3:5]
    # flip along h, w
    return cubes_new

class ModelWrapperERP(LightningModule):
    # logger: Optional[WandbLogger]
    encoder: nn.Module
    encoder_visualizer: Optional[EncoderVisualizer]
    decoder: Decoder
    losses: nn.ModuleList
    optimizer_cfg: OptimizerCfg
    test_cfg: TestCfg
    train_cfg: TrainCfg
    step_tracker: StepTracker | None

    def __init__(
        self,
        optimizer_cfg: OptimizerCfg,
        test_cfg: TestCfg,
        train_cfg: TrainCfg,
        encoder: Encoder,
        encoder_visualizer: Optional[EncoderVisualizer],
        decoder: Decoder,
        losses: list[Loss],
        step_tracker: StepTracker | None,
    ) -> None:
        super().__init__()
        self.optimizer_cfg = optimizer_cfg
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.step_tracker = step_tracker

        # Set up the model.
        self.encoder = encoder
        self.encoder_visualizer = encoder_visualizer
        self.decoder = decoder
        self.data_shim = get_data_shim(self.encoder)
        self.losses = nn.ModuleList(losses)

        # This is used for testing.
        self.benchmarker = Benchmarker()
        self.eval_cnt = 0


        # cube2equi
        global_cfg = get_cfg()
        self.cube_h = global_cfg.dataset.image_shape[0] // 2
        self.equi_h = global_cfg.dataset.image_shape[0] 
        self.equi_w = global_cfg.dataset.image_shape[1]  
        self.c2e = [Cube2Equirec(self.cube_h, self.equi_h, self.equi_w)] # hide from lightning 

        if self.test_cfg.compute_scores:
            self.test_step_outputs = {}
            self.time_skip_steps_dict = {"encoder": 0, "decoder": 0}
            self.test_step_mse_dict = {} # for image comparison


    def training_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)
        b, v, _, _, h, w = batch["target"]["image_cubes_supervise"].shape
        # Run the model.
        gaussians, pred_depth = self.encoder(
            batch["context"], self.global_step, False, scene_names=batch["scene"]
        )        

        output = self.decoder.forward(
            gaussians,
            rearrange(batch["target"]["extrinsics_cubes"], "b v cubes ... -> b (v cubes) ..."),
            rearrange(batch["target"]["intrinsics_cubes"], "b v cubes ... -> b (v cubes) ..."),
            rearrange(batch["target"]["near_cubes"], "b v cubes ... -> b (v cubes) ..."),
            rearrange(batch["target"]["far_cubes"], "b v cubes ... -> b (v cubes) ..."),
            (h, w),
            depth_mode=self.train_cfg.depth_mode,
        )
        target_gt = rearrange(batch["target"]["image_cubes_supervise"], "b v cubes ... -> b (v cubes) ...")


        # Compute metrics.
        psnr_probabilistic = compute_psnr(
            rearrange(target_gt, "b v c h w -> (b v) c h w"),
            rearrange(output.color, "b v c h w -> (b v) c h w"),
        )
        self.log("train/psnr_probabilistic", psnr_probabilistic.mean())

        # Compute and log loss.
        total_loss = 0
        def compute_context_depth_loss():
            context_depth_loss_weight = 0.1
            context_depth_sphere = rearrange(batch['context']['depth_sphere'], "b v () h w-> b v h w")
            near_threshold = 0.1
            context_depth_sphere_mask = context_depth_sphere > near_threshold # b v h w
            
            # import pdb;pdb.set_trace()
            far = batch['context']['far']
            far = far[0, 0] # assume all the far has the same value in one batch.
            context_depth_sphere[context_depth_sphere < 1e-7] = far

            context_depth_sphere_mask = context_depth_sphere_mask.float()
            if  not context_depth_sphere_mask.all(): # only apply when depth has holes.
                v_context = context_depth_sphere_mask.shape[1]
                mask_before = rearrange(context_depth_sphere_mask, "b v h w -> (b v) () h w")
                mask_after = erode(mask_before)
                mask_after = rearrange(mask_after, "(b v) () h w -> b v h w", b=b, v=v_context)
                context_depth_sphere_mask = mask_after
                # # visualize
                # import cv2
                # mask_before_np = mask_before.data.cpu().numpy()
                # cv2.imwrite("./mask_before.png", np.uint8(mask_before_np[0, 0] * 255))
                # mask_after_np = mask_after.data.cpu().numpy()
                # cv2.imwrite("./mask_after.png", np.uint8(mask_after_np[0, 0] * 255))   

            # if self.cfg.supervise_disparity:                
            #     context_disparity_sphere = safe_divide(1.0, context_depth_sphere)
            #     pred_disparity = safe_divide(1.0, pred_depth)
            #     total_context_depth_loss = context_depth_loss_weight * \
            #         compute_l1_sphere_loss(pred_disparity, context_disparity_sphere, mask=context_depth_sphere_mask, keep_batch=False)
            # else:
            total_context_depth_loss = context_depth_loss_weight * \
                compute_l1_sphere_loss(pred_depth, context_depth_sphere, mask=context_depth_sphere_mask, keep_batch=False)
            return total_context_depth_loss

        # we don't supervise depth with GT depth after half training process to focus on appearance learning.
        # if self.cfg.depth_warmup_training and self.global_step >= self.trainer.max_steps // 2:
            # total_context_depth_loss = 0.0
        # else:
        if not self.train_cfg.wo_depth_supervise:
            total_context_depth_loss = compute_context_depth_loss()
        else:
            total_context_depth_loss = torch.zeros((1, )).to(pred_depth)[0]

        total_loss += total_context_depth_loss
        self.log("loss/d_loss", total_context_depth_loss)

        for loss_fn in self.losses:
            loss = loss_fn.forward(output, batch, gaussians, self.global_step)
            self.log(f"loss/{loss_fn.name}", loss)
            total_loss = total_loss + loss
        self.log("loss/total", total_loss)

        if (
            self.global_rank == 0
            and self.global_step % self.train_cfg.print_log_every_n_steps == 0
        ):
            print(
                f"train step {self.global_step}; "
                f"scene = {[x[:20] for x in batch['scene']]}; "
                f"context = {batch['context']['index'].tolist()}; "
                f"bound = [{batch['context']['near_cubes'].detach().cpu().numpy().mean()} "
                f"{batch['context']['far_cubes'].detach().cpu().numpy().mean()}]; "
                f"loss = {total_loss:.6f}; "
                f"d_loss = {total_context_depth_loss:.6f}"
            )

        self.log("info/near", batch["context"]["near_cubes"].detach().cpu().numpy().mean())
        self.log("info/far", batch["context"]["far_cubes"].detach().cpu().numpy().mean())
        self.log("info/global_step", self.global_step)  # hack for ckpt monitor

        # Tell the data loader processes about the current step.
        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)

        return total_loss

    def test_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)
        b, v, _, _, h, w = batch["target"]["image_cubes_supervise"].shape
        num_cubes = 6
        assert b == 1

        # Render Gaussians.
        with self.benchmarker.time("encoder"):
            gaussians, pred_depth = self.encoder(
                batch["context"],
                self.global_step,
                deterministic=False,
            )
        
        
        depth_mode = None if not self.test_cfg.eval_depth else "depth"

        with self.benchmarker.time("decoder", num_calls=v*num_cubes): # b*v
            output = self.decoder.forward(
                gaussians,
                rearrange(batch["target"]["extrinsics_cubes"], "b v c r1 r2 -> b (v c) r1 r2"),
                rearrange(batch["target"]["intrinsics_cubes"], "b v c r1 r2 -> b (v c) r1 r2"),
                rearrange(batch["target"]["near_cubes"], "b v c -> b (v c)"),
                rearrange(batch["target"]["far_cubes"], "b v c -> b (v c)"),
                (h, w),
                depth_mode=depth_mode,
            )

        (scene,) = batch["scene"]
        name = get_cfg()["wandb"]["name"]
        path = self.image_dir / name
        if self.test_cfg.eval_depth:
            depths_prob = output.depth[0]
            depths_prob = rearrange(depths_prob, "(v cubes)... -> v cubes ...", v=v, cubes=num_cubes) # v cubes h w
            depth_gt = batch['target']["depth_cubes"][0].squeeze(-1) # v cubes h w

        images_prob = output.color[0]
        images_prob = rearrange(images_prob, "(v cubes)... -> v cubes ...", v=v, cubes=num_cubes) # v cubes c h w

        # rgb_gt = rearrange(batch["target"]["image_cubes_supervise"][0],  "(v cubes) ... -> v cubes ...", v=v, cubes=num_cubes)  # v cubes c h w
        rgb_gt = batch["target"]["image_cubes_supervise"][0] # v cubes c h w
        
        if self.test_cfg.save_image:
            for index in range(len(batch["target"]["index"][0])):
                cubes = []
                for cube_index in range(num_cubes):
                    color_pred = images_prob[index][cube_index]
                    color = rgb_gt[index][cube_index] # in zip(batch["target"]["index"][0], rearrange(batch['target']['image_cubes'][0], "v cubes ... -> (v cubes) ...")):
                    save_image(color_pred, path / "render" / scene / f"color/{index:0>6}_{cube_index:0>6}.png")
                    save_image(color, path / "render" / scene / f"color/{index:0>6}_{cube_index:0>6}_gt.png")
                    rgb_residual = torch.abs(color - color_pred)
                    rgb_residual = rgb_residual.mean(0, keepdim=True) # 1 H W
                    rgb_residual = convert_single_colormap(rgb_residual)
                    save_image(rgb_residual, path / "render" / scene / f"color/{index:0>6}_{cube_index:0>6}_err.png")
                    cubes.append(color_pred)
                    
                    if self.test_cfg.eval_depth:
                        depth_pred = depths_prob[index][cube_index]
                        depth = depth_gt[index][cube_index]
                        # near = batch["target"]["near_cubes"][0][index][cube_index]    
                        # far = batch["target"]["far_cubes"][0][index][cube_index]
                        # import pdb;pdb.set_trace()

                        # vis_disparity
                        depth_pred_vis = depth_map(depth_pred)
                        depth_vis = depth_map(depth)

                        # depth_residual = torch.abs(depth - depth_pred) 
                        save_image(depth_pred_vis, path / "render" / scene / f"color/{index:0>6}_{cube_index:0>6}_depth.png")
                        save_image(depth_vis, path / "render" / scene / f"color/{index:0>6}_{cube_index:0>6}_depth_gt.png")
                        # import pdb;pdb.set_trace()

                cubes = torch.stack(cubes, dim=0)
                

                # cubes are changed!
                cubes_new = change_order(cubes)
                cubes_new = rearrange(cubes_new, "cubes c h w -> () c h (cubes w)")
                self.c2e[0] = self.c2e[0].to(cubes_new.device)
                erp_new = self.c2e[0](cubes_new)
                # visualize the cube maps
                save_image(erp_new[0], path / "render" / scene / f"color/{index:0>6}_erp.png")
                save_image(batch["target"]["image_sphere"][0][index], path / "render" / scene / f"color/{index:0>6}_erp_gt.png")



            # save_input_views
            # if self.test_cfg.save_input_views:
            for index in range(len(batch["context"]["index"][0])):
                color = batch["context"]["image_sphere"][0][index]
                save_image(color, path / "render" / scene / f"color/{index:0>6}_input.png")

        # save video
        if self.test_cfg.save_video:
            # import pdb;pdb.set_trace()
            frame_str = "_".join([str(x.item()) for x in batch["context"]["index"][0]])
            # render cubes.
            
            for cube_idx in range(num_cubes):
                 # 3 c h w
                save_video(
                    [a for a in images_prob[:, cube_idx, ...]],
                    path / "video" / f"{scene}_frame_{frame_str}_cube_{cube_idx}.mp4",
                )

            # stitch into a panorama video            
            images_prob_reorder = change_order_batch(images_prob) # [b cubes c h w] 
            images_prob_reorder = rearrange(images_prob_reorder, "b cubes c h w -> b c h (cubes w)")
            self.c2e[0] = self.c2e[0].to(images_prob_reorder.device)
            images_prob_erp = self.c2e[0](images_prob_reorder) # b c h w
            save_video(
                [a for a in images_prob_erp],
                path / "video" / f"{scene}_frame_{frame_str}_erp.mp4",
            )

            if self.test_cfg.eval_depth:
                # import pdb;pdb.set_trace()
                depths_prob = output.depth[0] # b (v cubes) h w
                depths_prob = rearrange(depths_prob, "(v cubes)... -> v cubes ...", v=v, cubes=num_cubes) # v cubes c h w
                for cube_idx in range(num_cubes):
                    # 3 c h w
                    save_video(
                        [depth_map(a) for a in depths_prob[:, cube_idx, ...]],
                        path / "video" / f"{scene}_frame_{frame_str}_cube_{cube_idx}_depth.mp4",
                    )

                # stitch into a panorama video
                depths_prob_reorder = change_order_batch(depths_prob) # b cubes h w
                intrinsics_cubes = rearrange(batch['target']['intrinsics_cubes'][0], "v cubes r1 r2 -> (v cubes) r1 r2") # v cubes 3 3
                depths_prob_reorder = rearrange(depths_prob_reorder, "v cubes h w -> (v cubes) h w")
                height, width = depths_prob_reorder.shape[-2:]
                fx = intrinsics_cubes[:, 0, 0] * width
                fy = intrinsics_cubes[:, 1, 1] * height
                cx = intrinsics_cubes[:, 0, 2] * width
                cy = intrinsics_cubes[:, 1, 2] * height
                fxfycxcy = torch.stack([fx, fy, cx, cy], dim=1) # (v cubes) 4
                fxfycxcy = repeat(fxfycxcy, "vc r ->  vc r h w", h=height, w=width)                
                # depth to distance map
                depths_prob_reorder = depth_to_distance_map_batch(depths_prob_reorder, fxfycxcy) # vc h w 
                # import pdb;pdb.set_trace()
                depths_prob_reorder = rearrange(depths_prob_reorder, "(v cubes) h w -> v () h (cubes w)", v=v, cubes=num_cubes)

                self.c2e[0] = self.c2e[0].to(depths_prob_reorder.device) 
                depths_prob_erp = self.c2e[0](depths_prob_reorder) # b c h w
                depths_prob_erp = depths_prob_erp.squeeze(1)

                save_video(
                    [depth_map(a) for a in depths_prob_erp],
                    path / "video" / f"{scene}_frame_{frame_str}_erp_depth.mp4",
                )
                
        # compute scores
        if self.test_cfg.compute_scores:
            if batch_idx < self.test_cfg.eval_time_skip_steps:
                self.time_skip_steps_dict["encoder"] += 1
                self.time_skip_steps_dict["decoder"] += v
            rgb = images_prob
            rgb = rearrange(rgb, "v cubes ... -> (v cubes) ...")
            rgb_gt = rearrange(rgb_gt, "v cubes ... -> (v cubes) ...")

            if f"psnr" not in self.test_step_outputs:
                self.test_step_outputs[f"psnr"] = []
            if f"ssim" not in self.test_step_outputs:
                self.test_step_outputs[f"ssim"] = []
            if f"lpips" not in self.test_step_outputs:
                self.test_step_outputs[f"lpips"] = []
            self.test_step_outputs[f"psnr"].append(
                compute_psnr(rgb_gt, rgb).mean().item()
            )
            self.test_step_outputs[f"ssim"].append(
                compute_ssim(rgb_gt, rgb).mean().item()
            )
            self.test_step_outputs[f"lpips"].append(
                compute_lpips(rgb_gt, rgb).mean().item()
            )
            
            # psnr = compute_psnr(rgb_gt, rgb).mean().item()
            mse = F.mse_loss(rgb_gt, rgb).item()
            # we save mse for image comparison
            self.test_step_mse_dict[scene] = mse

            if self.test_cfg.eval_depth and 'depth_cubes' in batch['target'].keys():
                # novel view cube depth
                depth_gt = batch['target']['depth_cubes'] # b v 6 h w 1
                # order: U B L F R D
                depth_gt = depth_gt[:, :, 1:] # drop top view due to empty depth (cause NaNs)
                depth_gt = rearrange(depth_gt, "b v cubes h w () -> (b v cubes) () h w")

                depth_pred = output.depth # b (v cubes) h w
                depth_pred = rearrange(depth_pred, "b (v c) h w -> b v c h w", v=v, c=num_cubes)[:, :, 1:, ...]
                depth_pred = rearrange(depth_pred, "b v c h w -> (b v c) () h w")
                if depth_pred.shape != depth_gt.shape:
                    depth_pred = F.interpolate(
                            depth_pred, 
                            size=(depth_gt.shape[-2], depth_gt.shape[-1]),
                            mode="nearest",
                        )
                # using minimum of 0.5m
                valid_mask_b = (depth_gt > 0.1)
                # assert not (depth_gt == depth_pred).all()
                # we remove top view as its depths are all zeros more frequently in our dataset.
                # only evaluate > 1e-10
                valid_mask_batch = torch.any((depth_gt >= 0.1).reshape(b*v*(num_cubes-1), -1), dim=-1)

                # Check if there any valbid gt points in this sample
                if (valid_mask_b).any():
                    # compute metrics
                    depth_metrics_dict = compute_depth_metrics_batched(
                        depth_gt.flatten(start_dim=1).float(), 
                        depth_pred.flatten(start_dim=1).float(), 
                        valid_mask_b.flatten(start_dim=1),
                        mult_a=True,
                    )
                    for key in depth_metrics_dict.keys():
                        if key in ["abs_diff", "abs_rel", "rmse", "a25"]: # only evaluate with three metrics
                            if key not in self.test_step_outputs:
                                self.test_step_outputs[key] = []
                            
                            depth_metrics_dict[key][~valid_mask_batch] = 0
                            # non zero
                            batch_valid_cnt = valid_mask_batch.count_nonzero()
                            assert batch_valid_cnt > 0
                            self.test_step_outputs[key].append( (depth_metrics_dict[key].sum() / batch_valid_cnt).item() )

                    # # torch.isnan()
                    # if torch.isnan((depth_metrics_dict[key] * valid_mask_batch).mean()).any():
                    #     # import pdb;pdb.set_trace()
                    #     torch.isnan

    def on_test_end(self) -> None:
        name = get_cfg()["wandb"]["name"]
        # out_dir = self.test_cfg.output_path / name
        # name = get_cfg()["wandb"]["name"]
        # import pdb;pdb.set_trace()

        out_dir = self.image_dir / name
        saved_scores = {}
        if self.test_cfg.compute_scores:
            self.benchmarker.dump_memory(out_dir / "peak_memory.json")
            self.benchmarker.dump(out_dir / "benchmark.json")
            for metric_name, metric_scores in self.test_step_outputs.items():
                avg_scores = sum(metric_scores) / len(metric_scores)
                saved_scores[metric_name] = avg_scores
                print(metric_name, avg_scores)
                with (out_dir / f"scores_{metric_name}_all.json").open("w") as f:
                    json.dump(metric_scores, f)
                metric_scores.clear()
            for tag, times in self.benchmarker.execution_times.items():
                times = times[int(self.time_skip_steps_dict[tag]) :]
                saved_scores[tag] = [len(times), np.mean(times)]
                print(
                    f"{tag}: {len(times)} calls, avg. {np.mean(times)} seconds per call"
                )
                self.time_skip_steps_dict[tag] = 0

            with (out_dir / f"scores_all_avg.json").open("w") as f:
                json.dump(saved_scores, f)

            # for image comparison
            with (out_dir / f"mse_dict.json").open("w") as f:
                json.dump(self.test_step_mse_dict, f)  
            self.benchmarker.clear_history()
        else:
            self.benchmarker.dump(self.test_cfg.output_path / name / "benchmark.json")
            self.benchmarker.dump_memory(
                self.test_cfg.output_path / name / "peak_memory.json"
            )
            self.benchmarker.summarize()
        

    @rank_zero_only
    def validation_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)

        if self.global_rank == 0:
            print(
                f"validation step {self.global_step}; "
                f"scene = {[a[:20] for a in batch['scene']]}; "
                f"context = {batch['context']['index'].tolist()}"
            )

        # Render Gaussians.
        b, _, _, _, h, w = batch["target"]["image_cubes_supervise"].shape
        assert b == 1
        gaussians_softmax, pred_depths = self.encoder(
            batch["context"],
            self.global_step,
            deterministic=False,
        )

        import cv2
        output_dir = self.image_dir / "depth"
        output_dir.mkdir(exist_ok=True, parents=True)
        pred_depths_norm = rearrange(depth_map(pred_depths[0, 0]), "c h w -> h w c")
        pred_depths_norm = np.uint8(pred_depths_norm.data.cpu().numpy() * 255)
        cv2.imwrite(str(output_dir / f"{self.global_step}_pred.jpg"), pred_depths_norm)

        output_softmax = self.decoder.forward(
            gaussians_softmax,
            rearrange(batch["target"]["extrinsics_cubes"], "b v c r1 r2 -> b (v c) r1 r2"),
            rearrange(batch["target"]["intrinsics_cubes"], "b v c r1 r2 -> b (v c) r1 r2"),
            rearrange(batch["target"]["near_cubes"], "b v c -> b (v c)"),
            rearrange(batch["target"]["far_cubes"], "b v c -> b (v c)"),
            (h, w),
        )
        
        rgb_softmax = output_softmax.color[0]
        # Compute validation metrics.
        rgb_gt = rearrange(batch["target"]["image_cubes_supervise"], "b v cubes ... -> b (v cubes) ...")[0]
        for tag, rgb in zip(
            ("val",), (rgb_softmax,)
        ):
            psnr = compute_psnr(rgb_gt, rgb).mean()
            self.log(f"val/psnr_{tag}", psnr)
            lpips = compute_lpips(rgb_gt, rgb).mean()
            self.log(f"val/lpips_{tag}", lpips)
            ssim = compute_ssim(rgb_gt, rgb).mean()
            self.log(f"val/ssim_{tag}", ssim)
        
        rgb_residual = torch.abs(rgb_gt - rgb_softmax)
        rgb_residual = rgb_residual.mean(1, keepdim=True) # V 1 H W                
        rgb_residual = convert_colormap(rgb_residual)


        # Construct comparison image.
        comparison = hcat(
            # add_label(vcat(*batch["context"]["image"][0]), "Context"),
            add_label(vcat(*batch['context']["image_sphere"][0]), "Target ERP"),
            add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
            add_label(vcat(*rgb_softmax), "Target (Softmax)"),
            add_label(vcat(*rgb_residual), "Target (Difference)"),

        )
        subfolder = "val"

        output_dir = Path(self.image_dir) / "images" / subfolder
        self.log_image(
            "comparison",
            prep_image(add_border(comparison)),
            step=self.global_step,
            output_dir=output_dir,
        )

        # # Render projections and construct projection image.
        # projections = hcat(*render_projections(
        #                         gaussians_softmax,
        #                         256,
        #                         extra_label="(Softmax)",
        #                     )[0])
        # self.logger.log_image(
        #     "projection",
        #     [prep_image(add_border(projections))],
        #     step=self.global_step,
        # )

        # # Draw cameras.
        # cameras = hcat(*render_cameras(batch, 256))
        # self.logger.log_image(
        #     "cameras", [prep_image(add_border(cameras))], step=self.global_step
        # )

        # if self.encoder_visualizer is not None:
        #     for k, image in self.encoder_visualizer.visualize(
        #         batch["context"], self.global_step
        #     ).items():
        #         self.logger.log_image(k, [prep_image(image)], step=self.global_step)

        # Run video validation step.
        self.render_video_interpolation(batch, name='rgb_val')
        # self.render_video_wobble(batch)
        # if self.train_cfg.extended_visualization:
        #     self.render_video_interpolation_exaggerated(batch)

    # @rank_zero_only
    # def render_video_wobble(self, batch: BatchedExample) -> None:
    #     # Two views are needed to get the wobble radius.
    #     _, v, _, _ = batch["context"]["extrinsics"].shape
    #     if v != 2:
    #         return

    #     def trajectory_fn(t):
    #         origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
    #         origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
    #         delta = (origin_a - origin_b).norm(dim=-1)
    #         extrinsics = generate_wobble(
    #             batch["context"]["extrinsics"][:, 0],
    #             delta * 0.25,
    #             t,
    #         )
    #         intrinsics = repeat(
    #             batch["context"]["intrinsics"][:, 0],
    #             "b i j -> b v i j",
    #             v=t.shape[0],
    #         )
    #         return extrinsics, intrinsics

    #     return self.render_video_generic(batch, trajectory_fn, "wobble", num_frames=60)


    @rank_zero_only
    def log_image(self, name, x_sample, step, output_dir):
        # output_dir = Path(output_dir)
        image_dir = Path(output_dir / name)
        image_dir.mkdir(parents=True, exist_ok=True)
        imsave(str(image_dir / f'{step}.jpg'), x_sample)

    @rank_zero_only
    def render_video_interpolation(self, batch: BatchedExample, name: str="") -> None:
        _, v, _, _ = batch["context"]["extrinsics_sphere"].shape
        cube_idx = 1

        extrinsics = interpolate_render_poses_m9d(batch["context"]["extrinsics_cubes"][0, :, cube_idx], view_num_interp=120)

        # supposing all the views share the same camera intrinsics and near, far
        intrinsics = batch["context"]["intrinsics_cubes"][0, 0, cube_idx] # 1 3 3, 
        v_interp = len(extrinsics)
        intrinsics = repeat(intrinsics, " r1 r2 -> v r1 r2", v=v_interp)
        # near = rearrange(batch["context"]["near_cubes"][0, :], "b v cubes ... -> b (v cubes) ... ")
        # near = rearrange(batch["context"]["near_cubes"][0, 0, 1], "b v cubes ... -> b (v cubes) ... ")

        near = repeat(batch["context"]["near_cubes"][0, 0, cube_idx].unsqueeze(0), "() -> () v", v=v_interp)
        far = repeat(batch["context"]["far_cubes"][0, 0, cube_idx].unsqueeze(0), "() -> () v", v=v_interp)

        return self.render_video_generic(batch, extrinsics[None], intrinsics[None], near, far, name)

        # def trajectory_fn(t):
        #     extrinsics = interpolate_extrinsics(
        #         batch["context"]["extrinsics"][0, 0],
        #         (
        #             batch["context"]["extrinsics"][0, 1]
        #             if v == 2
        #             else batch["target"]["extrinsics"][0, 0]
        #         ),
        #         t,
        #     )
        #     intrinsics = interpolate_intrinsics(
        #         batch["context"]["intrinsics"][0, 0],
        #         (
        #             batch["context"]["intrinsics"][0, 1]
        #             if v == 2
        #             else batch["target"]["intrinsics"][0, 0]
        #         ),
        #         t,
        #     )
        #     return extrinsics[None], intrinsics[None]
        # return self.render_video_generic(batch, trajectory_fn, "rgb")

    # @rank_zero_only
    # def render_video_interpolation_exaggerated(self, batch: BatchedExample) -> None:
    #     # Two views are needed to get the wobble radius.
    #     _, v, _, _ = batch["context"]["extrinsics"].shape
    #     if v != 2:
    #         return

    #     def trajectory_fn(t):
    #         origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
    #         origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
    #         delta = (origin_a - origin_b).norm(dim=-1)
    #         tf = generate_wobble_transformation(
    #             delta * 0.5,
    #             t,
    #             5,
    #             scale_radius_with_t=False,
    #         )
    #         extrinsics = interpolate_extrinsics(
    #             batch["context"]["extrinsics"][0, 0],
    #             (
    #                 batch["context"]["extrinsics"][0, 1]
    #                 if v == 2
    #                 else batch["target"]["extrinsics"][0, 0]
    #             ),
    #             t * 5 - 2,
    #         )
    #         intrinsics = interpolate_intrinsics(
    #             batch["context"]["intrinsics"][0, 0],
    #             (
    #                 batch["context"]["intrinsics"][0, 1]
    #                 if v == 2
    #                 else batch["target"]["intrinsics"][0, 0]
    #             ),
    #             t * 5 - 2,
    #         )
    #         return extrinsics @ tf, intrinsics[None]

    #     return self.render_video_generic(
    #         batch,
    #         trajectory_fn,
    #         "interpolation_exagerrated",
    #         num_frames=300,
    #         smooth=False,
    #         loop_reverse=False,
    #     )

    # @rank_zero_only
    # def render_video_generic(
    #     self,
    #     batch: BatchedExample,
    #     trajectory_fn: TrajectoryFn,
    #     name: str,
    #     num_frames: int = 30,
    #     smooth: bool = True,
    #     loop_reverse: bool = True,
    # ) -> None:
    #     # Render probabilistic estimate of scene.
    #     gaussians_prob = self.encoder(batch["context"], self.global_step, False)
    #     # gaussians_det = self.encoder(batch["context"], self.global_step, True)

    #     t = torch.linspace(0, 1, num_frames, dtype=torch.float32, device=self.device)
    #     if smooth:
    #         t = (torch.cos(torch.pi * (t + 1)) + 1) / 2

    #     extrinsics, intrinsics = trajectory_fn(t)

    #     _, _, _, h, w = batch["context"]["image"].shape

    #     # Color-map the result.
    #     def depth_map(result):
    #         near = result[result > 0][:16_000_000].quantile(0.01).log()
    #         far = result.view(-1)[:16_000_000].quantile(0.99).log()
    #         result = result.log()
    #         result = 1 - (result - near) / (far - near)
    #         return apply_color_map_to_image(result, "turbo")

    #     # TODO: Interpolate near and far planes?
    #     near = repeat(batch["context"]["near"][:, 0], "b -> b v", v=num_frames)
    #     far = repeat(batch["context"]["far"][:, 0], "b -> b v", v=num_frames)
    #     output_prob = self.decoder.forward(
    #         gaussians_prob, extrinsics, intrinsics, near, far, (h, w), "depth"
    #     )
    #     images_prob = [
    #         vcat(rgb, depth)
    #         for rgb, depth in zip(output_prob.color[0], depth_map(output_prob.depth[0]))
    #     ]
    #     # output_det = self.decoder.forward(
    #     #     gaussians_det, extrinsics, intrinsics, near, far, (h, w), "depth"
    #     # )
    #     # images_det = [
    #     #     vcat(rgb, depth)
    #     #     for rgb, depth in zip(output_det.color[0], depth_map(output_det.depth[0]))
    #     # ]
    #     images = [
    #         add_border(
    #             hcat(
    #                 add_label(image_prob, "Softmax"),
    #                 # add_label(image_det, "Deterministic"),
    #             )
    #         )
    #         for image_prob, _ in zip(images_prob, images_prob)
    #     ]

    #     video = torch.stack(images)
    #     video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
    #     if loop_reverse:
    #         video = pack([video, video[::-1][1:-1]], "* c h w")[0]
    #     visualizations = {
    #         f"video/{name}": wandb.Video(video[None], fps=30, format="mp4")
    #     }

    #     # Since the PyTorch Lightning doesn't support video logging, log to wandb directly.
    #     try:
    #         wandb.log(visualizations)
    #     except Exception:
    #         assert isinstance(self.logger, LocalLogger)
    #         for key, value in visualizations.items():
    #             tensor = value._prepare_video(value.data)
    #             clip = mpy.ImageSequenceClip(list(tensor), fps=value._fps)
    #             dir = LOG_PATH / key
    #             dir.mkdir(exist_ok=True, parents=True)
    #             clip.write_videofile(
    #                 str(dir / f"{self.global_step:0>6}.mp4"), logger=None
    #             )

    @rank_zero_only
    def render_video_generic(
        self,
        batch: BatchedExample,
        # trajectory_fn: TrajectoryFn,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        near, 
        far,
        name: str,
        num_frames: int = 60,
        smooth: bool = True,
        loop_reverse: bool = True,
        key: Optional[str] = None,
    ) -> None:
        # Render probabilistic estimate of scene.
        gaussians_prob, pred_depth = self.encoder(batch["context"], self.global_step, False)

        h, w = batch['target']['image_cubes_supervise'].shape[-2:]
        # t = torch.linspace(0, 1, num_frames, dtype=torch.float32, device=self.device)

        # if smooth:
        #     t = (torch.cos(torch.pi * (t + 1)) + 1) / 2
        # extrinsics, intrinsics = trajectory_fn(t)


        # _, _, _, h, w = batch["context"]["image"].shape

        # Color-map the result.
        # def depth_map(result):
        #     # try:
        #     #     near = result[result > 0][:16_000_000].quantile(0.01).log()
        #     #     far = result.view(-1)[:16_000_000].quantile(0.99).log()
        #     #     result = result.log()
        #     #     result = 1 - (result - near) / (far - near)
        #     # except:
        #     near = result.min()
        #     far = result.max()
        #     result = 1 - (result - near) / (far - near)
        #     return apply_color_map_to_image(result, "turbo")
        # TODO: Interpolate near and far planes?

        # near = repeat(batch["context"]["near_cubes"][:, 0], "b -> b v", v=num_frames)
        # far = repeat(batch["context"]["far_cubes"][:, 0], "b -> b v", v=num_frames)

        # near = rearrange(batch["context"]["near_cubes"][0, :], "b v cubes -> b (v cubes", )

        # rearrange(batch["target"]["extrinsics_cubes"], "b v cubes ... -> b (v cubes) ..."),

        output_prob = self.decoder.forward(
            gaussians_prob, extrinsics, intrinsics, near, far, (h, w), "depth", 
        )
        

        images_prob = [
            vcat(rgb, depth)
            for rgb, depth in zip(output_prob.color[0], depth_map(output_prob.depth[0]))
        ]
        
        images = [
            add_border(
                hcat(
                    add_label(image_prob, "Softmax"),
                    # add_label(image_det, "Deterministic"),
                )
            )
            for image_prob, _ in zip(images_prob, images_prob)
        ]

        video = torch.stack(images)
        video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
        if loop_reverse:
            video = pack([video, video[::-1][1:-1]], "* c h w")[0]
        
        output_dir = Path(self.image_dir) / "video" / name

        # output_dir = LOG_PATH / "video" / name 
                # dir.mkdir(exist_ok=True, parents=True)
                # clip.write_videofile(
                #     str(dir / f"{self.global_step:0>6}.mp4"), logger=None
                # )

        # output_dir = Path(self.image_dir) / "video" / name 
        output_dir.mkdir(exist_ok=True, parents=True)
        video = video.transpose((0, 2, 3, 1))
        imageio.mimsave(output_dir / f"{self.global_step:0>6}.mp4", video, fps=30)
        # clip.write_videofile(
        #     str(dir / f"{self.global_step:0>6}.mp4"), logger=None
        # )


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.optimizer_cfg.lr)
        if self.optimizer_cfg.cosine_lr:
            warm_up = torch.optim.lr_scheduler.OneCycleLR(
                            optimizer, self.optimizer_cfg.lr,
                            self.trainer.max_steps + 10,
                            pct_start=0.01,
                            cycle_momentum=False,
                            anneal_strategy='cos',
                        )
        else:
            warm_up_steps = self.optimizer_cfg.warm_up_steps
            warm_up = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                1 / warm_up_steps,
                1,
                total_iters=warm_up_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": warm_up,
                "interval": "step",
                "frequency": 1,
            },
        }
