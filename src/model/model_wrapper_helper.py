
import torch
import torch.nn.functional as F
def dilate(bin_img, ksize=5):
    pad = (ksize - 1) // 2
    bin_img = F.pad(bin_img, pad=[pad, pad, pad, pad], mode='reflect')
    out = F.max_pool2d(bin_img, kernel_size=ksize, stride=1, padding=0)
    return out

# def dilate(bin_img, ksize=5):
#     src_size = bin_img.shape
#     pad = (ksize - 1) // 2
    
#     out = F.max_pool2d(bin_img, kernel_size=ksize, stride=1, padding=0)
#     out = F.interpolate(out,
#                         size=src_size[2:],
#                         mode="bilinear")
#     return out


def erode(bin_img, ksize=5):
    # 腐蚀
    out = 1 - dilate(1 - bin_img, ksize)
    return out

def clamp_away_from(tensor, val=0, eps=1e-10):
    """Clamps all elements in tensor away from val.

    All values which are epsilon away stay the same.
    All values epsilon close are clamped to the nearest acceptable value.

    Args:
        tensor: Input tensor.
        val: Value you do not want in the tensor.
        eps: Distance away from the value which is acceptable.

    Returns:
        Input tensor where no elements are within eps of val.

    """
    if not torch.is_tensor(val):
        val = torch.tensor(val, device=tensor.device, dtype=tensor.dtype)
    tensor = torch.where(torch.ge(tensor, val), torch.max(tensor, val + eps),
                        torch.min(tensor, val - eps))
    return tensor

def safe_divide(num, den, eps=1e-10):
    """Performs a safe divide. Do not use this function.

    Args:
        num: Numerator.
        den: Denominator.
        eps: Epsilon.

    Returns:
        Quotient tensor.

    """
    new_den = clamp_away_from(den, eps=eps)
    return num / new_den


def compute_l1_sphere_loss(y_pred, y_true, mask=None, keep_batch=False):
    """
    Computes the l1 loss between the rendered image and the ground truth
    with a sin factor to account for the size of each pixel.
    Args:
        y_pred: Predicted image as a (B, V, H, W) tensor.
        y_true: Ground truth image as a (B, V, H, W) tensor.
        mask: Mask for valid GT values.
    Returns:
        Loss tensor.
    """
    
    batch_size, views, height, width = y_pred.shape
    sin_phi = torch.arange(0, height, dtype=y_pred.dtype, device=y_pred.device)
    sin_phi = torch.sin((sin_phi + 0.5) * torch.pi / height)

    sum_axis = (0, 1, 2, 3)
    if keep_batch:
        sum_axis = (1, 2, 3)

    if mask is not None:
        sin_phi = sin_phi.view(1, 1, height, 1).expand(batch_size, views, height, width,)
        sin_phi = sin_phi * mask
        loss = torch.abs(y_true - y_pred) * sin_phi
        loss = safe_divide(torch.sum(loss, dim=sum_axis), torch.sum(sin_phi, dim=sum_axis))        
    else:
        raise NotImplementedError
    return loss