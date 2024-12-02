import torch
from einops import repeat, rearrange
from jaxtyping import Float
from torch import Tensor

def get_world_points_with_sphere_projection(
        utils,
        depth: Float[Tensor, "b v h w"], 
        pose: Float[Tensor, "b v r1 r2"], 
        h: int, 
        w: int,
    ):
    device, dtype = depth.device, depth.dtype
    with torch.no_grad():
        # pixel coordinates
        # grid = coords_grid(
        #     b, h, w, homogeneous=True, device=depth.device
        # )  # [B, 3, H, W]

        # phi = torch.arange(
        #     0,
        #     h,
        #     device=device,
        #     dtype=dtype
        # )
        # theta = torch.arange(
        #     0,
        #     w,
        #     device=device,
        #     dtype=dtype
        # )

        xy_locs = utils.get_xy_coords()
        spherical_coords = utils.equi_2_spherical(xy_locs)

        # # equi2spherical
        # if dataset_name in ["CoffeeArea", "outdoor_colmap"]:
        #     theta = (-2 * torch.pi / (w - 1)) * theta + 2 * torch.pi
        #     phi = (torch.pi / (h - 1)) * phi
        # elif dataset_name in ["replica_new", "hm3d_new"]:           
        #     theta = (0.5 - (theta + 0.5) / w) * 2 * torch.pi
        #     phi = -((phi + 0.5) / h - 0.5) * torch.pi
        # else:
        #     raise Exception
        # phi, theta = torch.meshgrid(phi, theta)

        # spherical to camera cartesian
        # points = my_torch_helpers.spherical_to_cartesian(dataset_name, theta, phi, r=1)        
        # r = depth
        b, v = depth.shape[:2]
        # depth = rearrange(depth, "b v h w -> b v h w")
        spherical_coords = repeat(spherical_coords, "b h w c -> b v h w c", v=v)
        spherical_coords = spherical_coords.to(depth.device)
        # import pdb;pdb.set_trace()
        spherical_coords[..., 2] = depth

        # import pdb;pdb.set_trace()
        # rearrange(r, "b v h w -> b v () h w")

        # spherical_coords = torch.cat([depth, ])
        # rearrange(spherical_coords, )
        # spherical_coords[..., 2] = r

        # points = utils.spherical_2_cartesian()

        # phi = repeat(phi, "h w -> b v () h w", b=b, v=v)
        # theta = repeat(theta, "h w -> b v () h w", b=b, v=v)

        # points = spherical_to_cartesian(dataset_name, r, phi, theta) # b 1 h w c
        points = utils.spherical_2_cartesian(spherical_coords) # b v h w c

        # points = points[None, :, :, :].expand(batch_size, h, w, 3)        
        # back project to 3D and transform viewpoint
        # points = torch.inverse(intrinsics).bmm(grid.view(b, 3, -1))  # [B, 3, H*W]
        points = rearrange(points, "b v h w c -> b v c (h w)")
        # (b 3 3) x (b 3 n) -> (b 3 n) -> (b 3 d n) * (b 1 d n) -> (b 3 d n)
        # points = torch.bmm(pose[:, :3, :3], points).unsqueeze(2).repeat(
        #     1, 1, d, 1
        # ) * depth.view(
        #     b, 1, d, h * w
        # )  # [B, 3, D, H*W]
        
        # pose = repeat(pose, "b v r1 r2 -> b v r1 r2")
        # points = torch.bmm(pose[..., :3, :3], points) + pose[..., :3, -1:].unsqueeze(-1)  # [B, 3, D, H*W]
        points = pose[..., :3, :3] @ points + pose[..., :3, -1:]  
        return rearrange(points, "b v c n -> b v n c")        
