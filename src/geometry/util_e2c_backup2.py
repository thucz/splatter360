import numpy as np
from scipy.ndimage import map_coordinates
import cv2

# Based on https://github.com/sunset1995/py360convert
class Equirec2Cube:
    def __init__(self, equ_h, equ_w, face_w):
        '''
        equ_h: int, height of the equirectangular image
        equ_w: int, width of the equirectangular image
        face_w: int, the length of each face of the cubemap
        '''

        self.equ_h = equ_h
        self.equ_w = equ_w
        self.face_w = face_w

        self._xyzcube()
        self._xyz2coor()

        # For convert R-distance to Z-depth for CubeMaps
        cosmap = 1 / torch.sqrt((2 * self.grid[..., 0]) ** 2 + (2 * self.grid[..., 1]) ** 2 + 1)
        self.cosmaps = torch.cat(6 * [cosmap], dim=1).unsqueeze(-1)

    def _xyzcube(self):
        '''
        Compute the xyz cordinates of the unit cube in [U F R B L D] format.
        '''
        self.xyz = torch.zeros((6, self.face_w, self.face_w, 3), torch.float32)
        rng = torch.linspace(-0.5, 0.5, num=self.face_w, dtype=torch.float32)
        self.grid = torch.stack(torch.meshgrid(rng, -rng, indexing='xy'), dim=-1)

        # Up face (y = 0.5)
        self.xyz[0:1, :, [0, 2]] = self.grid[::-1, :]
        self.xyz[0:1, :, 1] = 0.5

        # Front face (z = 0.5)
        self.xyz[1:2, :, [0, 1]] = self.grid
        self.xyz[1:2, :, 2] = 0.5

        # Right face (x = 0.5)
        self.xyz[2:3, :, [2, 1]] = self.grid[:, ::-1]
        self.xyz[2:3, :, 0] = 0.5

        # Back face (z = -0.5)
        self.xyz[3:4, :, [0, 1]] = self.grid[:, ::-1]
        self.xyz[3:4, :, 2] = -0.5

        # Left face (x = -0.5)
        self.xyz[4:5, :, [2, 1]] = self.grid
        self.xyz[4:5, :, 0] = -0.5

        # Down face (y = -0.5)
        self.xyz[5:6, :, [0, 2]] = self.grid
        self.xyz[5:6, :, 1] = -0.5

    def _xyz2coor(self):

        # x, y, z to longitude and latitude
        x, y, z = torch.split(self.xyz, 3, dim=-1)
        lon = torch.atan2(x, z)
        c = torch.sqrt(x ** 2 + z ** 2)
        lat = torch.atan2(y, c)

        # longitude and latitude to equirectangular coordinate
        self.coor_x = (lon / (2 * torch.pi) + 0.5) * self.equ_w - 0.5
        self.coor_y = (-lat / torch.pi + 0.5) * self.equ_h - 0.5

    def sample_equirec(self, e_img, order=0):
        grid = torch.cat([self.coor_x, self.coor_y], dim=-1)
        grid[..., 0] = (grid[..., 0] + 0.5) / self.equ_w * 2.0 - 1.0
        grid[..., 1] = (grid[..., 1] + 0.5) / self.equ_h * 2.0 - 1.0
        return F.grid_sample(e_img, grid, padding_mode='border', align_corners=True)    

        # return map_coordinates(e_img, [self.coor_y, self.coor_x],
        #                        order=order, mode='wrap')[..., 0]

    def run_pt(self, equ_img, equ_dep=None):
        # h, w = equ_img.shape[:2] # b c h w 
        # if h != self.equ_h or w != self.equ_w:
        #     equ_img = cv2.resize(equ_img, (self.equ_w, self.equ_h))
        #     if equ_dep is not None:
        #         equ_dep = cv2.resize(equ_dep, (self.equ_w, self.equ_h), interpolation=cv2.INTER_NEAREST)

        cube_img = torch.stack([self.sample_equirec(equ_img[..., i])
                             for i in range(equ_img.shape[2])], dim=-1)

        # if equ_dep is not None:
        #     cube_dep = np.stack([self.sample_equirec(equ_dep[..., i], order=0)
        #                          for i in range(equ_dep.shape[2])], axis=-1)
        #     cube_dep = cube_dep * self.cosmaps

        # if equ_dep is not None:
        #     return cube_img, cube_dep
        # else:
        #     return cube_img
    def run(self, equ_img, equ_dep=None):
        h, w = equ_img.shape[:2]
        if h != self.equ_h or w != self.equ_w:
            equ_img = cv2.resize(equ_img, (self.equ_w, self.equ_h))
            if equ_dep is not None:
                equ_dep = cv2.resize(equ_dep, (self.equ_w, self.equ_h), interpolation=cv2.INTER_NEAREST)

        cube_img = np.stack([self.sample_equirec(equ_img[..., i], order=1)
                             for i in range(equ_img.shape[2])], axis=-1)

        if equ_dep is not None:
            cube_dep = np.stack([self.sample_equirec(equ_dep[..., i], order=0)
                                 for i in range(equ_dep.shape[2])], axis=-1)
            cube_dep = cube_dep * self.cosmaps

        if equ_dep is not None:
            return cube_img, cube_dep
        else:
            return cube_img