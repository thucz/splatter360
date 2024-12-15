
import torch.nn.functional as F

import torch
import numpy as np
from einops import rearrange, repeat
from scipy.spatial.transform.rotation import Rotation
def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))

def interpolate_views(n_views_add, start_pose, end_pose):
    # 3x4
    # R,T
    delta = (end_pose - start_pose) / (n_views_add + 1)
    new_poses_add = []
    for i in range(n_views_add):
        pose_add = start_pose + delta * (i+1)
        new_poses_add.append(pose_add)
    return new_poses_add

def interpolate_render_poses(poses, view_num):
    # poses = [database.get_w2c(str(img_id)) for img_id in inter_img_ids]
    add_poses_len = view_num - len(poses) 
    add = add_poses_len // (len(poses)-1) 
    rest = add_poses_len % (len(poses)-1)
    new_poses = []
    # poses[i] -> poses[i+1]
    for i in range(len(poses)-1):
        # i, i+1
        # interpolate views
        if i < rest:
            add_poses = interpolate_views(add+1, poses[i], poses[i+1])
        else:
            add_poses = interpolate_views(add, poses[i], poses[i+1])
        
        new_poses.append(poses[i])
        new_poses += add_poses

    new_poses.append(poses[-1])
    new_poses = np.stack(new_poses, axis=0)
    return new_poses
    
def interpolate_render_poses_m9d(poses):
    # import pdb;pdb.set_trace()
    pose_torch = torch.from_numpy(poses)
    m6d = matrix_to_rotation_6d(pose_torch[..., :3, :3])
    m9d = torch.cat([m6d, pose_torch[..., :3, 3]], dim=-1)
    view_num_orig = len(poses)
    view_num_interp = view_num_orig * 5
    m9d_interp = interpolate_render_poses(m9d, view_num=view_num_interp)    
    m6d_interp = m9d_interp[..., :6]
    rotation_matrix = rotation_6d_to_matrix(torch.from_numpy(m6d_interp))
    rotation_matrix = rotation_matrix.data.cpu().numpy()
    # import pdb;pdb.set_trace()
    trans_interp = m9d_interp[..., 6:]
    pose_interp = np.concatenate([rotation_matrix, trans_interp[..., np.newaxis]], axis=-1)
    # bottom = repeat(torch.tensor([[0, 0, 0, 1.0]]), "() 4 -> n () 4", n=view_num_orig)
    # return torch.cat([pose_torch_interp, bottom], dim=1)
    return pose_interp


def path_to_poses(shortest_path):
    c2ws = []
    for path_i in shortest_path:
        rot = Rotation.from_quat(path_i.rotation)
        trans = path_i.position
        trans = np.array(trans)
        trans = trans[..., np.newaxis]
        c2w = np.concatenate([rot.as_matrix(), trans], axis=-1)
        c2ws.append(c2w)
    c2ws = np.stack(c2ws, axis=0)
    return c2ws # n 3 4
