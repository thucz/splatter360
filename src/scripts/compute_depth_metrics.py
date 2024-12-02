import json
import numpy as np
import torch

# Refer to SimpleRecon: https://github.com/nianticlabs/simplerecon/blob/main/utils/metrics_utils.py
def compute_depth_metrics(gt, pred, mult_a=False):
    """
    Computes error metrics between predicted and ground truth depths
    """
    thresh = torch.max((gt / pred), (pred / gt))
    a_dict = {}
    a_dict["a5"] = (thresh < 1.05     ).float().mean()
    a_dict["a10"] = (thresh < 1.10     ).float().mean()
    a_dict["a25"] = (thresh < 1.25     ).float().mean()

    a_dict["a0"] = (thresh < 1.10     ).float().mean()
    a_dict["a1"] = (thresh < 1.25     ).float().mean()
    a_dict["a2"] = (thresh < 1.25 ** 2).float().mean()
    a_dict["a3"] = (thresh < 1.25 ** 3).float().mean()
    if mult_a:
        for key in a_dict:
            a_dict[key] = a_dict[key]*100

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    sq_rel = torch.mean((gt - pred) ** 2 / gt)

    abs_diff = torch.mean(torch.abs(gt - pred))

    metrics_dict = {
                    "abs_diff": abs_diff,
                    "abs_rel": abs_rel,
                    "sq_rel": sq_rel,
                    "rmse": rmse,
                    "rmse_log": rmse_log,
                }
    metrics_dict.update(a_dict)

    return metrics_dict

def compute_depth_metrics_batched(gt_bN, pred_bN, valid_masks_bN, mult_a=False):
    """
    Computes error metrics between predicted and ground truth depths, 
    batched. Abuses nan behavior in torch.
    """

    gt_bN = gt_bN.clone()
    pred_bN = pred_bN.clone()

    gt_bN[~valid_masks_bN] = torch.nan
    pred_bN[~valid_masks_bN] = torch.nan

    thresh_bN = torch.max(torch.stack([(gt_bN / pred_bN), (pred_bN / gt_bN)], 
                                                            dim=2), dim=2)[0]
    a_dict = {}
    
    a_val = (thresh_bN < (1.0+0.05)     ).float()
    a_val[~valid_masks_bN] = torch.nan
    a_dict[f"a5"] = torch.nanmean(a_val, dim=1)

    a_val = (thresh_bN < (1.0+0.10)     ).float()
    a_val[~valid_masks_bN] = torch.nan
    a_dict[f"a10"] = torch.nanmean(a_val, dim=1) 

    a_val = (thresh_bN < (1.0+0.25)     ).float()
    a_val[~valid_masks_bN] = torch.nan
    a_dict[f"a25"] = torch.nanmean(a_val, dim=1)

    a_val = (thresh_bN < (1.0+0.10)     ).float()
    a_val[~valid_masks_bN] = torch.nan
    a_dict[f"a0"] = torch.nanmean(a_val, dim=1)

    a_val = (thresh_bN < (1.0+0.25)     ).float()
    a_val[~valid_masks_bN] = torch.nan
    a_dict[f"a1"] = torch.nanmean(a_val, dim=1)

    a_val = (thresh_bN < (1.0+0.25) ** 2).float()
    a_val[~valid_masks_bN] = torch.nan
    a_dict[f"a2"] = torch.nanmean(a_val, dim=1)

    a_val = (thresh_bN < (1.0+0.25) ** 3).float()
    a_val[~valid_masks_bN] = torch.nan
    a_dict[f"a3"] = torch.nanmean(a_val, dim=1)

    if mult_a:
        for key in a_dict:
            a_dict[key] = a_dict[key]*100

    rmse_bN = (gt_bN - pred_bN) ** 2
    rmse_b = torch.sqrt(torch.nanmean(rmse_bN, dim=1))

    rmse_log_bN = (torch.log(gt_bN) - torch.log(pred_bN)) ** 2
    rmse_log_b = torch.sqrt(torch.nanmean(rmse_log_bN, dim=1))

    abs_rel_b = torch.nanmean(torch.abs(gt_bN - pred_bN) / gt_bN, dim=1)

    sq_rel_b = torch.nanmean((gt_bN - pred_bN) ** 2 / gt_bN, dim=1)

    abs_diff_b = torch.nanmean(torch.abs(gt_bN - pred_bN), dim=1)

    metrics_dict = {
                    "abs_diff": abs_diff_b,
                    "abs_rel": abs_rel_b,
                    "sq_rel": sq_rel_b,
                    "rmse": rmse_b,
                    "rmse_log": rmse_log_b,
                }
    metrics_dict.update(a_dict)

    return metrics_dict