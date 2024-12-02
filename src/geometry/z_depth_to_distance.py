import torch
import torch.nn.functional as F

def depth_to_distance_map_batch(depth_maps, fxfycxcy):
    """
    将深度图批量转换为距离图
    参数:
    depth_maps (torch.Tensor): 深度图批量，形状为 (batch_size, height, width)
    fx, fy (float): x轴和y轴上的焦距
    cx, cy (float): 图像的主点
    
    返回:
    distance_maps (torch.Tensor): 距离图批量，形状为 (batch_size, height, width)
    """
    batch_size, height, width = depth_maps.shape
    
    # 创建网格
    u, v = torch.meshgrid(torch.arange(width, device=depth_maps.device), 
                          torch.arange(height, device=depth_maps.device))
    u = u.unsqueeze(0).expand(batch_size, -1, -1).float()
    v = v.unsqueeze(0).expand(batch_size, -1, -1).float()

    fx = fxfycxcy[:, 0] # b h w 
    fy = fxfycxcy[:, 1] # b h w
    cx = fxfycxcy[:, 2] # b h w
    cy = fxfycxcy[:, 3] # b h w

    # 计算X和Y
    X = (u - cx) * depth_maps / fx
    Y = (v - cy) * depth_maps / fy

    # 计算距离
    distance_maps = torch.sqrt(X**2 + Y**2 + depth_maps**2)
    return distance_maps

