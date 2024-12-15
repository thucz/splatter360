from dataset_generation.configs import my_helpers
import numpy as np
class Utils360:
  def __init__(self, full_height, full_width):
    width = full_width
    height = full_height
    theta, phi = np.meshgrid((np.arange(width) + 0.5) * (2 * np.pi / width),
                             (np.arange(height) + 0.5) * (np.pi / height))
    uvs, uv_sides = my_helpers.spherical_to_cubemap(theta.reshape(-1), phi.reshape(-1))
    self.width = width
    self.height = height
    self.uvs = uvs.reshape(height, width, 2)
    self.uv_sides = uv_sides.reshape(height, width)
    self.depth_to_dist_cache = {}
    # self.m3d_dist = m3d_dist
    self.config_hfov = 90

    

  def stitch_cubemap(self, cubemap, clip=True):
    """Stitches a single cubemap into an equirectangular image.
    Args:
      cubemap: Cubemap images as 6xHxWx3 arrays.
      clip: Clip values to [0, 1].
    Returns:
      Single equirectangular image as HxWx3 image.
    """
    cube_height, cube_width = cubemap.shape[1:3]

    uvs = self.uvs
    uv_sides = self.uv_sides
    height = self.height
    width = self.width

    skybox_uvs = np.stack(
      (uvs[:, :, 0] * (cube_width - 1), uvs[:, :, 1] * (cube_height - 1)),
      axis=-1)
    final_image = np.zeros((height, width, 3), dtype=np.float32)
    for i in range(0, 6):
      # Grabs a transformed side of the cubemap.
      my_side_indices = np.equal(uv_sides, i)
      final_image[my_side_indices] = my_helpers.bilinear_interpolate(
        cubemap[i, :, :, :], skybox_uvs[my_side_indices, 0],
        skybox_uvs[my_side_indices, 1])
    if clip:
      final_image = np.clip(final_image, 0, 1)
    return final_image

  def zdepth_to_distance(self, depth_image):
    """Converts a depth (z-depth) image to a euclidean distance image.

    Args:
      depth_image: Equirectangular depth image as BxHxWx1 array.

    Returns: Equirectangular distance image.

    """
    batch_size, height, width, channels = depth_image.shape
    
    # print("height, width:", height, width)
    # import ipdb;ipdb.set_trace()
    cache_key = "_".join((str(height), str(width)))
    self.cache_depth_to_dist(height, width)
    ratio = self.depth_to_dist_cache[cache_key]
    new_depth_image = depth_image * ratio[np.newaxis, :, :, np.newaxis]
    return new_depth_image

  def cache_depth_to_dist(self, height, width):
    """Caches a depth to dist ratio"""
    cache_key = "_".join((str(height), str(width)))
    if cache_key not in self.depth_to_dist_cache:
      # import ipdb;ipdb.set_trace()
      cubemap_height = height//2
      cubemap_width = height//2
      # cubemap_height = 256
      # cubemap_width = 256

      # Distance to image plane
      theta, phi = np.meshgrid(
        (np.arange(width) + 0.5) * (2 * np.pi / width),
        (np.arange(height) + 0.5) * (np.pi / height))
      uvs, uv_sides = my_helpers.spherical_to_cubemap(theta.reshape(-1), phi.reshape(-1))
      cubemap_uvs = uvs.reshape(height, width, 2)
      uv_int = np.stack(
        (cubemap_uvs[:, :, 0] * (cubemap_width - 1),
         cubemap_uvs[:, :, 1] * (cubemap_height - 1)), axis=-1)
      
      width_center = cubemap_width / 2 - 0.5
      height_center = cubemap_height / 2 - 0.5

      focal_len = (cubemap_height / 2) / np.tan(self.config_hfov * np.pi / 180.0/2)

      diag_dist = np.sqrt((uv_int[:, :, 0] - width_center) ** 2 +
                          (uv_int[:, :, 1] - height_center) ** 2 + focal_len ** 2)
      self.depth_to_dist_cache[cache_key] = diag_dist / focal_len
