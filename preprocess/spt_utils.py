import math
import torch
import copy

import torch.nn.functional as F
from torch.distributions.uniform import Uniform

class Utils(object):
    def __init__(self, configs):
        super().__init__()
        dataset = configs.dataset_name
        batch_size = configs.batch_size
        height = configs.img_wh[1]
        width = configs.img_wh[0]
        batch_size = 1

        self.height, self.width = height, width
        self.batch_size = batch_size
        self.dataset = dataset

    def get_xy_coords(self, device='cpu'):
        """Short summary.
        :return: Description of returned object.
        :rtype: torch.Tensor of shape [B, H, W, 2]
        """
        height, width = self.height, self.width
        batch_size = 1
        x_locs = torch.linspace(0, width-1, width).view(1, width, 1)
        y_locs = torch.linspace(0, height-1, height).view(height, 1, 1)
        x_locs, y_locs = map(lambda x: x.to(device), [x_locs, y_locs])
        x_locs, y_locs = map(lambda x: x.expand(
            height, width, 1), [x_locs, y_locs])
        xy_locs = torch.cat([x_locs, y_locs], dim=2)
        xy_locs = xy_locs.unsqueeze(0).expand(batch_size, height, width, 2)
        return xy_locs

    def equi_2_spherical(self, equi_coords, radius=1):
        """
        """
        height, width = self.height, self.width
        input_shape = equi_coords.shape
        assert input_shape[-1] == 2, 'last coordinate should be 2'
        # if 'replica' in self.dataset:
        #     x_locs, y_locs = torch.split(
        #         tensor=equi_coords, dim=-1, split_size_or_sections=1)
        #     theta = (2*math.pi / (width-1)) * x_locs - math.pi
        #     phi = (math.pi/(height-1))*(y_locs)
        #     spherical_coords = torch.cat(
        #         [theta, phi, torch.ones_like(theta).mul(radius)], dim=-1)
        if self.dataset == 'residential':
            x_locs, y_locs = torch.split(
                tensor=equi_coords, dim=-1, split_size_or_sections=1)
            x_locs = x_locs.clamp(min=0, max=width-1)
            y_locs = y_locs.clamp(min=0, max=height-1)
            theta = math.pi*(2*x_locs/(width-1) - 1.5)
            phi = math.pi*(0.5-y_locs/(height-1))
            spherical_coords = torch.cat(
                [theta, phi, torch.ones_like(theta).mul(radius)], dim=-1)
        elif self.dataset == 'replica' or self.dataset == 'hm3d':
            x_locs, y_locs = torch.split(
                tensor=equi_coords, dim=-1, split_size_or_sections=1)
            x_locs = x_locs.clamp(min=0, max=width-1)
            y_locs = y_locs.clamp(min=0, max=height-1)

            # theta = (-2*math.pi / (width-1)) * x_locs + 2*math.pi
            # phi = (math.pi/(height-1))*(y_locs)
            
            theta = (0.5 - (x_locs + 0.5) / width) * 2 * math.pi
            phi = -((y_locs + 0.5) / height - 0.5) * math.pi

            spherical_coords = torch.cat(
                [theta, phi, torch.ones_like(theta).mul(radius)], dim=-1)

        else:
            x_locs, y_locs = torch.split(
                tensor=equi_coords, dim=-1, split_size_or_sections=1)
            x_locs = x_locs.clamp(min=0, max=width-1)
            y_locs = y_locs.clamp(min=0, max=height-1)
            theta = (-2*math.pi / (width-1)) * x_locs + 2*math.pi
            phi = (math.pi/(height-1))*(y_locs)
            spherical_coords = torch.cat(
                [theta, phi, torch.ones_like(theta).mul(radius)], dim=-1)

        return spherical_coords

    def spherical_2_cartesian(self, spherical_coords):  # checked
        '''
        '''
        input_shape = spherical_coords.shape
        assert input_shape[-1] in [2,
                                   3], 'last dimension of input should be 3 or 2'
        coordinate_split = torch.split(
            spherical_coords, split_size_or_sections=1, dim=-1)
        theta, phi = coordinate_split[:2]
        if input_shape[-1] == 3:
            rad = coordinate_split[2]
        else:
            rad = torch.ones_like(theta).to(theta.device)
        if self.dataset == 'residential':
            # theta
            x_locs = rad * torch.cos(theta) * torch.cos(phi)
            z_locs = rad * torch.sin(theta) * torch.cos(phi)
            y_locs = rad * torch.sin(phi)
        elif self.dataset == 'replica' or self.dataset == 'hm3d':
            # x_locs = rad * torch.cos(theta) * torch.cos(phi)
            # z_locs = rad * torch.sin(theta) * torch.cos(phi)
            # y_locs = rad * torch.sin(phi)
            y_locs = torch.sin(phi) * rad
            x_locs = torch.cos(phi) * torch.sin(theta) * rad
            z_locs = torch.cos(phi) * torch.cos(theta) * rad
        else:
            # self.dataset in ['replica', 'd3dkit', 'coffee', 'coffee_1d', 'coffee_2d', 'coffee_12v', 'coffee_r2', 'coffee_r3', 'coffee_r1', 'coffee3x3', 'dense_replica']:
            x_locs = rad * torch.sin(phi) * torch.cos(theta)
            y_locs = rad * torch.sin(phi) * torch.sin(theta)
            z_locs = rad * torch.cos(phi)

        xyz_locs = torch.cat([x_locs, y_locs, z_locs], dim=-1)
        return xyz_locs

    def cartesian_2_spherical(self, input_points, normalized=False):
        '''
        '''
        last_coord_one = False
        if input_points.shape[-1] == 1:
            input_points = input_points.squeeze(-1)
            last_coord_one = True
        if not normalized:
            input_points = self.normalize_3d_vectors(input_points)
        x_c, y_c, z_c = torch.split(
            input_points, split_size_or_sections=1, dim=-1)
        r = torch.sqrt(x_c**2 + y_c**2 + z_c**2)
        # if 'replica' in self.dataset:
        #     theta = torch.atan2(y_c, x_c)
        #     phi = torch.acos(z_c/r)
        #     mask1 = theta.gt(math.pi)
        #     theta[mask1] = theta[mask1] - 2*math.pi
        #     mask2 = theta.lt(-1*math.pi)
        #     theta[mask2] = theta[mask2] + 2*math.pi
        if self.dataset == 'residential':
            theta = -torch.atan2(-z_c, x_c)
            phi = torch.asin(y_c/r)
            mask = torch.logical_and(
                theta.gt(math.pi*0.5), theta.le(2*math.pi))
            theta[mask] = theta[mask] - 2*math.pi
        elif self.dataset == 'replica' or self.dataset == 'hm3d':
            theta = torch.atan2(x_c, z_c) #[None, ...]              # (1, H, W)
            # compare_depth = np.linalg.norm(xyz_compare, axis=0)[None, ...]                     # (1, H, W)
            # xyz_compare = xyz_compare / (compare_depth + 1e-6) 
            # r = torch.linalg.norm()                                # (3, H, W)
            phi = torch.atan2(y_c, torch.sqrt(x_c**2 + z_c**2))   # (1, H, W)
        else:
            theta = torch.atan2(y_c, x_c)
            phi = torch.acos(z_c/r)
            mask1 = theta.lt(0)
            theta[mask1] = theta[mask1] + 2*math.pi
        spherical_coords = torch.cat(
            [theta, phi, torch.ones_like(theta)], dim=-1)
        return spherical_coords

    def spherical_2_equi(self, spherical_coords, height=None, width=None):
        """spherical coordinates to equirectangular coordinates
        :param spherical_coords: tensor of shape [B, ..., 3], [B, ..., 3, 1], [B, ..., 2] or [B, ..., 2, 1], 
        :param height: image height, optional when not given self.height will be used
        :param width: image width, optional when not given self.width will be used
        """
        height = height if not height is None else self.height
        width = width if not width is None else self.width
        last_coord_one = False
        if spherical_coords.shape[-1] == 1:
            spherical_coords = spherical_coords.squeeze(-1)
            last_coord_one = True
        spherical_coords = torch.split(
            spherical_coords, split_size_or_sections=1, dim=-1)
        theta, phi = spherical_coords[0], spherical_coords[1]
        # if 'replica' in self.dataset:
        #     x_locs = ((width-1)/(2.0*math.pi)) * (theta + math.pi)
        #     y_locs = (height-1)/math.pi * phi
        if self.dataset == 'residential':
            x_locs = ((1/(2.0*math.pi))*theta + (3/4.0))*(width-1)
            y_locs = (0.5 - phi/math.pi)*(height-1)
        elif self.dataset == 'replica' or self.dataset == 'hm3d':
            x_locs = (-theta / (2 * torch.pi) + 0.5) * (width-1) - 0.5                        # (1, H, W)
            y_locs = (-phi / torch.pi + 0.5) * (height-1) - 0.5                                # (1, H, W)
        else:
            x_locs = (width-1) * (1 - theta/(2.0*math.pi))
            y_locs = phi*(height-1)/math.pi
        xy_locs = torch.cat([x_locs, y_locs], dim=-1)
        if last_coord_one:
            xy_locs = xy_locs.unsqueeze(-1)
        return xy_locs

    def normalize_3d_vectors(self, input_points, p=2, eps=1e-12):
        '''normalises input 3d points along the last dimension
        :param input_points: 3D points of shape [B, ..., 3]
        :param p: norm power
        :param eps: epsilone to avoid division by 0
        '''
        last_coord_one = False
        p_norm = torch.norm(input_points, p=p, dim=-1,
                            keepdim=True).clamp(min=eps)
        normalized_points = input_points / p_norm
        return normalized_points

    def safe_ray_sphere_intersection(self, ray_origin, direction, sphere_radii):
        """ this is same as ray_sphere_intersection, 
            but we know that the sphere and ray intersect 
            at exactly one point 
        """
        assert ray_origin.shape == direction.shape, 'shapes of origin and direction should'
        assert len(sphere_radii.shape) in [
            2, 4], 'sphere radii should be 2 or 4 D'
        b_size, height, width, _ = ray_origin.shape
        num_spheres = sphere_radii.shape[-1]
        if len(sphere_radii.shape) == 2:
            # change sphere_radii  to shape [B, D, 1, 1]
            sphere_radii = sphere_radii.unsqueeze(
                -1).unsqueeze(-1).unsqueeze(-1)
        ray_origin = ray_origin.unsqueeze(1)
        direction = direction.unsqueeze(1)
        # solve the quadratic equation of ray sphere intersection
        # torch.sum(torch.mul(direction, direction), dim=-1, keepdim=True)
        a = 1
        b = 2*torch.mul(direction, ray_origin).sum(dim=-1, keepdim=True)
        # .sum(dim=-1, keepdim=True)
        r_square = torch.mul(sphere_radii, sphere_radii)
        r_square = r_square.expand(b_size, num_spheres, height, width, 1)
        c = torch.mul(ray_origin, ray_origin).sum(
            dim=-1, keepdim=True) - r_square
        discriminant_ = torch.mul(b, b).sum(dim=-1, keepdim=True) - 4*a*c
        discriminant = discriminant_  # .clamp(min=0.0)
        sqrt_discriminant = torch.sqrt(discriminant)
        denominator = (2.0*a)  # .clamp(min=1e-08)
        # t_values = (-b - sqrt_discriminant) / denominator
        t_values = (-b + sqrt_discriminant) / denominator
        intersection = ray_origin + torch.mul(t_values, direction)
        return intersection  # .view(b_size, num_spheres, height, width, 3)

    def ray_sphere_intersection(self, ray_origin, direction, sphere_radii):
        """
        :param torch.FloatTensor ray_origin: 3D points for the origin of each ray
        ray_origin shape [B, H, W, 3]
        :param torch.FloatTensor direction: 3D points where each ray ends
        direction shape [B, H, W, 3]
        :param torch.FloatTensor sphere_radii: radii of each shere
        [B, 1] or [B, D], [B, D, H, W], [B, 1, H, W]
        :return: Description of returned object.
        :rtype: torch.FloatTensor intersection: points of intersection
        intersection shape [B, D, H, W, 3]
        :rtype: torch.LongTensor mask: validity of each itnersection point
        mask shape [B, D, H, W, 1]
        """
        assert ray_origin.shape == direction.shape, 'shapes of origin and direction should'
        assert len(sphere_radii.shape) in [
            2, 4], 'sphere radii should be 2 or 4 D'
        b_size, height, width, _ = ray_origin.shape
        num_spheres = sphere_radii.shape[-1]
        if len(sphere_radii.shape) == 2:
            # change sphere_radii  to shape [B, D, 1, 1]
            sphere_radii = sphere_radii.unsqueeze(-1).unsqueeze(-1)
        # change ray and direction to shape [B, 1, H, W, 3]
        ray_origin = ray_origin.unsqueeze(1)
        # .expand(b_size, num_spheres, height, width, 3)
        direction = direction.unsqueeze(1)
        # solve the quadratic equation of ray sphere intersection
        a = torch.sum(torch.mul(direction, direction), dim=-1, keepdim=True)
        b = 2*torch.mul(direction, ray_origin).sum(dim=-1, keepdim=True)
        r_square = torch.mul(sphere_radii, sphere_radii).sum(
            dim=-1, keepdim=True)
        c = torch.mul(ray_origin, ray_origin).sum(
            dim=-1, keepdim=True) - r_square
        discriminant = torch.mul(b, b).sum(dim=-1, keepdim=True) - 4*a*c
        mask = discriminant.lt(0)
        not_mask = torch.logical_not(mask)
        discriminant[mask] = 0.0
        sqrt_discriminant = torch.sqrt(discriminant)
        denominator = (2.0*a).clamp(min=1e-07)
        sol_1 = (-b + sqrt_discriminant) / denominator
        sol_2 = (-b - sqrt_discriminant) / denominator
        sol_1_mask = sol_1 < sol_2
        sol_2_mask = torch.logical_not(sol_1_mask)
        solution = torch.zeros_like(sqrt_discriminant)
        solution[sol_1_mask] = sol_1[sol_1_mask]
        solution[sol_2_mask] = sol_2[sol_2_mask]
        t_values = torch.zeros_like(solution)
        t_values[not_mask] = solution[not_mask]
        # now we need intesection points from t, ray_origin and direction
        intersection = ray_origin + torch.mul(t_values, direction)
        # ignore any negative value of t
        neg_t = t_values.lt(0)
        not_mask[neg_t] = False
        return intersection.view(b_size, num_spheres, height, width, 3), not_mask

    def normalize_grid(self, grid):
        has_D = len(grid.shape) == 5
        if has_D:
            b_size, num_d_chans = grid.shape[0], grid.shape[1]
            grid = grid.view(
                b_size*grid.shape[1], grid.shape[2], grid.shape[3], grid.shape[4])
        b = grid.shape[0]
        h, w = self.height, self.width
        x_locs, y_locs = torch.split(grid, split_size_or_sections=1, dim=-1)

        x_locs = torch.clamp(x_locs, min=0.0, max=w-1)
        y_locs = torch.clamp(y_locs, min=0.0, max=h-1)
        x_half = (w-1)/2.0
        y_half = (h-1)/2.0
        u_locs = (x_locs - x_half) / x_half
        v_locs = (y_locs - y_half) / y_half
        norm_grid = torch.cat([u_locs, v_locs], dim=-1)
        if has_D:
            norm_grid = norm_grid.view(
                b_size, num_d_chans, grid.shape[1], grid.shape[2], grid.shape[3])
        return norm_grid

    def sample_equi(self, input, grid, mode='bilinear', normalize_grid=True):
        """Samples input equi-rectangula image using a sampling grid
        :param input: tensor of shape [B, F, H, W] or [B, D, F, H, W] # D is when we process sphere sweep volumes
        :param grid: tensor of shape [B, H, W, 2] or [B, D, H, W, 2]
        :param normalize_grid: if true grid locs will be converted to range [-1, 1]
        """
        assert grid.shape[-1] == 2, 'grid last dim should be 2'
        h, w = input.shape[-2:]
        b_size = input.shape[0]
        assert h == self.height, 'input/grid height should be same as self.height'
        assert w == self.width, 'input/grid width should be same as self.width'
        has_D = len(input.shape) == 5
        if has_D:
            # assert len(grid.shape)==5, 'input grid should be 5D too'
            num_d_chans = input.shape[1]
            input = input.view(
                b_size*input.shape[1], input.shape[2], input.shape[3], input.shape[4])
            grid = grid.view(
                b_size*grid.shape[1], grid.shape[2], grid.shape[3], grid.shape[4])
        norm_grid = self.normalize_grid(grid) if normalize_grid else grid
        sampled = F.grid_sample(input=input, grid=norm_grid, mode=mode,
                                align_corners=True)
        if has_D:
            sampled = sampled.view(
                b_size, num_d_chans, sampled.shape[1], sampled.shape[2], sampled.shape[3])
        return sampled

    # def _get_unit_rays_on_sphere(self, device='cpu', b_size=1):
    #     x_y_locs = self.get_xy_coords(device=device)  # B, H, W, 2
    #     sph_locs = self.equi_2_spherical(x_y_locs)
    #     _, h, w, c = sph_locs.shape
    #     xyz_locs = self.spherical_2_cartesian(
    #         sph_locs).expand(b_size, h, w, c)  # B, H, W, 3
    #     return xyz_locs