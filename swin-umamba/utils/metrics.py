import torch
import numpy as np
from medpy import metric


def get_accuracy(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)
    corr = torch.sum(SR == GT)
    tensor_size = SR.size(0) * SR.size(1) * SR.size(2) * SR.size(3)
    acc = float(corr) / float(tensor_size)
    return acc


def get_sensitivity(SR, GT, threshold=0.5):
    # Sensitivity == Recall
    SE = 0
    SR = SR > threshold
    GT = GT == torch.max(GT)
    TP = ((SR == 1).byte() + (GT == 1).byte()) == 2
    FN = ((SR == 0).byte() + (GT == 1).byte()) == 2
    SE = float(torch.sum(TP)) / (float(torch.sum(TP + FN)) + 1e-6)
    return SE


def get_specificity(SR, GT, threshold=0.5):
    SP = 0
    SR = SR > threshold
    GT = GT == torch.max(GT)
    TN = ((SR == 0).byte() + (GT == 0).byte()) == 2
    FP = ((SR == 1).byte() + (GT == 0).byte()) == 2
    SP = float(torch.sum(TN)) / (float(torch.sum(TN + FP)) + 1e-6)
    return SP


def get_precision(SR, GT, threshold=0.5):
    PC = 0
    SR = SR > threshold
    GT = GT == torch.max(GT)
    TP = ((SR == 1).byte() + (GT == 1).byte()) == 2
    FP = ((SR == 1).byte() + (GT == 0).byte()) == 2
    PC = float(torch.sum(TP)) / (float(torch.sum(TP + FP)) + 1e-6)
    return PC


def iou_score(output, target):
    smooth = 1e-5
    if torch.is_tensor(output):
        output = torch.sigmoid(output).detach().cpu().numpy()
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5

    output[output > 0] = 1
    target[target > 0] = 1
    HD95 = 0.0  # 默认值，避免条件分支未定义
    if output.sum() > 0 and target.sum()>0:
        HD95 = metric.binary.hd95(output_, target_)

    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou = (intersection + smooth) / (union + smooth)
    dice = (2 * iou) / (iou + 1)
    output_ = torch.tensor(output_)
    target_ = torch.tensor(target_)

    SE = get_sensitivity(output_, target_, threshold=0.5)
    PC = get_precision(output_, target_, threshold=0.5)
    SP = get_specificity(output_, target_, threshold=0.5)
    ACC = get_accuracy(output_, target_, threshold=0.5)
    # F1 = 2 * SE * PC / (SE + PC + 1e-6)
    return iou, dice, SE, PC, SP, ACC, HD95


def iou_score_per_sample(output, target):
    """
    逐样本计算分割指标，返回每个样本的指标字典。
    用于需要计算样本间均值±标准差的场景。

    Args:
        output: torch.Tensor, shape (B, C, H, W), 模型原始输出（未sigmoid）
        target: torch.Tensor, shape (B, H, W) or (B, 1, H, W), 二值标签

    Returns:
        dict: 每个指标对应一个 list，长度为 B
    """
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).detach().cpu().numpy()
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()

    # 确保 target 是 4D
    if target.ndim == 3:
        target = target[:, np.newaxis, :, :]

    B = output.shape[0]
    metrics = {
        'iou': [], 'dice': [], 'se': [], 'pc': [],
        'sp': [], 'acc': [], 'hd95': []
    }

    for b in range(B):
        pred = output[b:b+1]  # (1, C, H, W)
        gt = target[b:b+1]    # (1, 1, H, W)

        pred_bin = (pred > 0.5).astype(np.float64)
        gt_bin = (gt > 0.5).astype(np.float64)

        intersection = (pred_bin * gt_bin).sum()
        union = (pred_bin + gt_bin).sum() - intersection

        iou = (intersection + smooth) / (union + smooth)
        dice = (2 * iou) / (iou + 1)

        # 计算 SE, PC, SP, ACC
        pred_t = torch.tensor(pred_bin)
        gt_t = torch.tensor(gt_bin)

        SE = get_sensitivity(pred_t, gt_t, threshold=0.5)
        PC = get_precision(pred_t, gt_t, threshold=0.5)
        SP = get_specificity(pred_t, gt_t, threshold=0.5)
        ACC = get_accuracy(pred_t, gt_t, threshold=0.5)

        # HD95
        if pred_bin.sum() > 0 and gt_bin.sum() > 0:
            hd95 = float(metric.binary.hd95(pred_bin, gt_bin))
        else:
            hd95 = 0.0

        metrics['iou'].append(float(iou))
        metrics['dice'].append(float(dice))
        metrics['se'].append(float(SE))
        metrics['pc'].append(float(PC))
        metrics['sp'].append(float(SP))
        metrics['acc'].append(float(ACC))
        metrics['hd95'].append(hd95)

    return metrics


def dice_coef(output, target):
    smooth = 1e-5
    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0

from scipy.ndimage import distance_transform_edt, binary_erosion, generate_binary_structure

def hd95_distance(pred, gt, voxelspacing=None):
    """
    计算分割结果和 GT 之间的 HD95（对称 95% Hausdorff 距离）
    pred, gt: 二值 numpy 数组（形状相同），前景为 True/1
    voxelspacing: 体素间距 (dz, dy, dx)，2D 可以用 (dy, dx) 或默认 None
    """
    pred = np.asarray(pred).astype(bool)
    gt   = np.asarray(gt).astype(bool)

    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: {pred.shape} vs {gt.shape}")

    # 两个都没有前景 → 距离为 0（完全空分割的情况）
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    # 只有一个有前景 → HD 理论上无限大，这里用 np.inf 表示
    if pred.sum() == 0 or gt.sum() == 0:
        return np.inf

    # 生成结构元素（4/6 邻域），适用于 2D/3D
    footprint = generate_binary_structure(pred.ndim, 1)

    # 提取表面（边界）点：mask XOR erode(mask)
    pred_border = pred ^ binary_erosion(pred, footprint, border_value=0)
    gt_border   = gt   ^ binary_erosion(gt,   footprint, border_value=0)

    # 距离变换：每个点到“对方边界”的最近距离
    # gt 的距离图：每个位置到 gt 边界的最近距离
    dt_gt   = distance_transform_edt(~gt_border, sampling=voxelspacing)
    dt_pred = distance_transform_edt(~pred_border, sampling=voxelspacing)

    # pred 边界上，每个点到 gt 边界的距离
    distances_pred_to_gt = dt_gt[pred_border]
    # gt 边界上，每个点到 pred 边界的距离
    distances_gt_to_pred = dt_pred[gt_border]

    all_distances = np.hstack([distances_pred_to_gt, distances_gt_to_pred])

    hd95 = np.percentile(all_distances, 95)
    return float(hd95)
