import math
import torch
import copy

import torch.nn.functional as F
from torch.distributions.uniform import Uniform

class Utils(object):
    def __init__(self, configs):
        super().__init__()
        dataset = configs["dataset_name"]
        batch_size = configs["batch_size"]
        height = configs["height"]
        width = configs["width"]
        self.configs = configs
        # batch_size = 1
        self.height, self.width = height, width
        self.batch_size = batch_size
        self.dataset = dataset

    def get_xy_coords(self, device='cpu'):
        """Short summary.
        :return: Description of returned object.
        :rtype: torch.Tensor of shape [B, H, W, 2]: (1, H, W, 2)
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
        # import pdb;pdb.set_trace()
        """
        """
        height, width = self.height, self.width
        input_shape = equi_coords.shape
        assert input_shape[-1] == 2, 'last coordinate should be 2'
        # import ipdb;ipdb.set_trace()
        if self.dataset == 'm3d':#[-pi, pi]
            # B, H, W, 2
            # the bug is fixed by adjust the theta calculation from (+pi/2)%(2*pi) to -pi.

            x_locs, y_locs = torch.split(
                tensor=equi_coords, dim=-1, split_size_or_sections=1)
            x_locs = x_locs.clamp(min=0, max=width-1)
            y_locs = y_locs.clamp(min=0, max=height-1)            
            theta = x_locs / (width-1) * 2 * math.pi
            theta = theta - 0.5* math.pi 
            phi = y_locs / (height-1) * math.pi
            #todo!
            spherical_coords = torch.cat(
                [theta, phi, torch.ones_like(theta).mul(radius)], dim=-1)
        # elif 'replica_test'== self.dataset:
        #     x_locs, y_locs = torch.split(
        #         tensor=equi_coords, dim=-1, split_size_or_sections=1)
        #     theta = x_locs * 2*math.pi / (width-1) - math.pi
        #     phi = -y_locs*math.pi/(height-1)+math.pi*0.5
        #     spherical_coords = torch.cat(
        #         [theta, phi, torch.ones_like(theta).mul(radius)], dim=-1)
        elif self.dataset == 'residential':
            x_locs, y_locs = torch.split(
                tensor=equi_coords, dim=-1, split_size_or_sections=1)
            x_locs = x_locs.clamp(min=0, max=width-1)
            y_locs = y_locs.clamp(min=0, max=height-1)
            theta = math.pi*(2*x_locs/(width-1) - 1.5)
            phi = math.pi*(0.5-y_locs/(height-1))
            spherical_coords = torch.cat(
                [theta, phi, torch.ones_like(theta).mul(radius)], dim=-1)
        elif self.dataset in ["CoffeeArea"]:
            x_locs, y_locs = torch.split(
                tensor=equi_coords, dim=-1, split_size_or_sections=1)
            x_locs = x_locs.clamp(min=0, max=width-1)
            y_locs = y_locs.clamp(min=0, max=height-1)
            theta = (-2*math.pi / (width-1)) * x_locs + 2*math.pi
            phi = (math.pi/(height-1))*(y_locs)
            spherical_coords = torch.cat(
                [theta, phi, torch.ones_like(theta).mul(radius)], dim=-1)
        elif self.dataset in ["outdoor_colmap"]:
            x_locs, y_locs = torch.split(
                tensor=equi_coords, dim=-1, split_size_or_sections=1)
            x_locs = x_locs.clamp(min=0, max=width-1)
            y_locs = y_locs.clamp(min=0, max=height-1)
            theta = (-2*math.pi / (width-1)) * x_locs + 2*math.pi
            phi = (math.pi/(height-1))*(y_locs)
            spherical_coords = torch.cat(
                [theta, phi, torch.ones_like(theta).mul(radius)], dim=-1)
        elif self.dataset in ['replica', 'hm3d']:
            x_locs, y_locs = torch.split(
                tensor=equi_coords, dim=-1, split_size_or_sections=1)
            x_locs = x_locs.clamp(min=0, max=width-1)
            y_locs = y_locs.clamp(min=0, max=height-1)

            
            theta = (0.5 - (x_locs + 0.5) / width ) * 2 * math.pi
            phi = -((y_locs + 0.5) / height - 0.5) * math.pi
            # [0, height] -> [0, 1] - 0.5 -> [-0.5, 0.5]
            spherical_coords = torch.cat(
                [theta, phi, torch.ones_like(theta).mul(radius)], dim=-1)
        else:
            raise Exception
        return spherical_coords

    def spherical_2_cartesian(self, spherical_coords):  # checked
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
        # import ipdb;ipdb.set_trace()
        if self.dataset == 'residential':
            # theta
            x_locs = rad * torch.cos(theta) * torch.cos(phi)
            z_locs = rad * torch.sin(theta) * torch.cos(phi)
            y_locs = rad * torch.sin(phi)
        elif self.dataset == "m3d":
            # print("rad:", rad)
            tmp = rad * torch.sin(phi)            
            x_locs = tmp * torch.cos(theta)
            y_locs = rad * torch.cos(phi)
            z_locs = tmp * torch.sin(theta)
            # return torch.stack((x_locs, y_locs, z_locs), axis=-1)
        # elif 'replica_test' == self.dataset:
        #     # self.dataset in ['replica', 'd3dkit', 'coffee', 'coffee_1d', 'coffee_2d', 'coffee_12v', 'coffee_r2', 'coffee_r3', 'coffee_r1', 'coffee3x3', 'dense_replica']:
        #     x_locs = rad * torch.sin(theta) * torch.cos(phi)
        #     y_locs = - rad * torch.sin(phi)
        #     z_locs = rad * torch.cos(theta) * torch.cos(phi)
        elif self.dataset in ["CoffeeArea"]:
            # self.dataset in ['replica', 'd3dkit', 'coffee', 'coffee_1d', 'coffee_2d', 'coffee_12v', 'coffee_r2', 'coffee_r3', 'coffee_r1', 'coffee3x3', 'dense_replica']:
            x_locs = rad * torch.sin(phi) * torch.cos(theta)
            y_locs = rad * torch.sin(phi) * torch.sin(theta)
            z_locs = rad * torch.cos(phi)
        elif self.dataset in ["outdoor_colmap"]:
            # self.dataset in ['replica', 'd3dkit', 'coffee', 'coffee_1d', 'coffee_2d', 'coffee_12v', 'coffee_r2', 'coffee_r3', 'coffee_r1', 'coffee3x3', 'dense_replica']:
            x_locs = rad * torch.sin(phi) * torch.cos(theta)
            y_locs = rad * torch.sin(phi) * torch.sin(theta)
            z_locs = rad * torch.cos(phi)
        elif self.dataset in ['replica', 'hm3d']:
            

            y_locs = torch.sin(phi) * rad
            x_locs = torch.cos(phi) * torch.sin(theta) * rad
            z_locs = torch.cos(phi) * torch.cos(theta) * rad
        else:
            raise Exception

        xyz_locs = torch.cat([x_locs, y_locs, z_locs], dim=-1)
        return xyz_locs

    def cartesian_2_spherical(self, input_points, normalized=False):
        x_c, y_c, z_c = torch.split(
            input_points, split_size_or_sections=1, dim=-1)
        radius = torch.linalg.norm(input_points, axis=-1).unsqueeze(-1)#

        if self.dataset == "m3d":            
            theta = torch.atan2(z_c, x_c)
            phi = torch.acos(y_c/(radius+1e-5))
            if torch.isnan(phi).any():
                import ipdb;ipdb.set_trace()
        # elif "replica_test" == self.dataset:
        #     theta = torch.atan2(x_c, z_c)
        #     phi = -torch.asin(y_c/radius)
            # mask1 = theta.gt(math.pi)
            # theta[mask1] = theta[mask1] - 2*math.pi
            # mask2 = theta.lt(-1*math.pi)
            # theta[mask2] = theta[mask2] + 2*math.pi
        elif self.dataset == 'residential':
            theta = -torch.atan2(-z_c, x_c)
            phi = torch.asin(y_c/radius)
            mask = torch.logical_and(
                theta.gt(math.pi*0.5), theta.le(2*math.pi))
            theta[mask] = theta[mask] - 2*math.pi
        elif self.dataset in ["CoffeeArea"]:
            theta = torch.atan2(y_c, x_c)
            phi = torch.acos(z_c/radius)
            mask1 = theta.lt(0)
            theta[mask1] = theta[mask1] + 2*math.pi
        elif self.dataset in ["outdoor_colmap"]:
            theta = torch.atan2(y_c, x_c)
            phi = torch.acos(z_c/radius)
            mask1 = theta.lt(0)
            theta[mask1] = theta[mask1] + 2*math.pi
        elif self.dataset in ['replica', 'hm3d']:
            theta = torch.atan2(x_c, z_c) #[None, ...]              # (1, H, W)
            # compare_depth = np.linalg.norm(xyz_compare, axis=0)[None, ...]                     # (1, H, W)
            # xyz_compare = xyz_compare / (compare_depth + 1e-6) 
            # r = torch.linalg.norm()                                # (3, H, W)
            phi = torch.atan2(y_c, torch.sqrt(x_c**2 + z_c**2))   # (1, H, W)

            # y_locs = torch.sin(phi) * rad
            # x_locs = torch.cos(phi) * torch.sin(theta) * rad
            # z_locs = torch.cos(phi) * torch.cos(theta) * rad
            
            # mask1 = theta.lt(0)
            # # import pdb;pdb.set_trace()
            # theta[mask1] = theta[mask1] + 2*math.pi
            # import pdb;pdb.set_trace()
            # import pdb;pdb.set_trace()


        else:
            raise Exception
        spherical_coords = torch.cat(
            [theta, phi, radius], dim=-1)
            
        return spherical_coords

    def spherical_2_equi(self, spherical_coords, height=None, width=None):
        """spherical coordinates to equirectangular coordinates
        :param spherical_coords: tensor of shape [B, ..., 3], [B, ..., 3, 1], [B, ..., 2] or [B, ..., 2, 1], 
        :param height: image height, optional when not given self.height will be used
        :param width: image width, optional when not given self.width will be used
        """

        height = height if not height is None else self.height
        width = width if not width is None else self.width
        spherical_coords = torch.split(
            spherical_coords, split_size_or_sections=1, dim=-1)        
        theta, phi = spherical_coords[0], spherical_coords[1]        
        if self.dataset == "m3d":
            theta = (theta +0.5*math.pi) % (2*math.pi)            
            x_locs = theta / (2*math.pi) * (width-1)
            y_locs = phi / (math.pi) * (height-1)            
        # elif "replica_test" == self.dataset:
        #     x_locs = ((width-1)/(2*math.pi)) * (theta + math.pi)
        #     # y_locs = (height-1)/math.pi * phi

        #     # theta  / 2*math.pi * (width-1) + 1/2 * (width-1)
        #     #  = -(height-1)* phi /math.pi + 1/2 * (height-1)
        #     y_locs = (height-1) / math.pi * (-phi + 0.5 * math.pi)
        elif self.dataset == 'residential':
            x_locs = ((1/(2.0*math.pi))*theta + (3/4.0))*(width-1)
            y_locs = (0.5 - phi/math.pi)*(height-1)
        elif self.dataset in ["CoffeeArea"]:
            x_locs = (width-1) * (1 - theta/(2.0*math.pi))
            y_locs = phi*(height-1)/math.pi
        elif self.dataset in ["outdoor_colmap"]:
            x_locs = (width-1) * (1 - theta/(2.0*math.pi))
            y_locs = phi*(height-1)/math.pi
        elif self.dataset in ['replica', 'hm3d']:
            # theta = (0.5 - (x_locs + 0.5) / width) * 2 * math.pi
            # phi = -((y_locs + 0.5) / height - 0.5) * math.pi
            # x_locs = (0.5 - theta / (2 * math.pi)) * width - 0.5
            # y_locs = (- phi / math.pi + 0.5) * height - 0.5

            # phi = -((y_locs + 0.5) / height - 0.5) * math.pi
            # # [0, height] -> [0, 1] - 0.5 -> [-0.5, 0.5]

            # theta = (0.5 - (x_locs + 0.5) / width) * 2 * math.pi
            # phi = -((y_locs + 0.5) / height - 0.5) * math.pi

            x_locs = (-theta / (2 * torch.pi) + 0.5) * width - 0.5                        # (1, H, W)
            y_locs = (-phi / torch.pi + 0.5) * height - 0.5
            
        else:
            raise Exception
        xy_locs = torch.cat([x_locs, y_locs], dim=-1)
        return xy_locs