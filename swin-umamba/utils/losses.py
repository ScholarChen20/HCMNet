import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

deep_supervision_scales = [1.0, 0.5, 0.25, 0.125]  # 示例比例
weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
weights[-1] = 0  # 忽略最低分辨率
weights = weights / weights.sum()       #归一化


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        smooth = 1e-5
        bce = F.binary_cross_entropy_with_logits(input, target)
        # ce = CrossEntropyLoss()
        # bce = ce(input, target)

        input = torch.sigmoid(input)
        num = target.size(0)               #return batch_size
        # input = input.view(num, -1)
        # target = target.view(num, -1)
        input = input.reshape(num, -1)
        target = target.reshape(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num

        return 0.5 * bce + dice

class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss()

    def forward(self, pred, target):
        size = pred.size(0)
        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)

        return self.bceloss(pred_, target_)


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1
        size = pred.size(0)

        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)
        intersection = pred_ * target_
        dice_score = (2 * intersection.sum(1) + smooth)/(pred_.sum(1) + target_.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum()/size

        return dice_loss


class BceDiceLoss(nn.Module):
    def __init__(self, wb=1, wd=1):
        super(BceDiceLoss, self).__init__()
        self.bce = BCELoss()
        self.dice = DiceLoss()
        self.wb = wb
        self.wd = wd

    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        diceloss = self.dice(pred, target)

        loss = self.wd * diceloss + self.wb * bceloss
        return loss

from scipy.ndimage import distance_transform_edt


def generate_distance_map(mask: torch.Tensor, normalize: bool = True, use_signed: bool = False) -> torch.Tensor:
    """
    生成二值掩码的距离变换图（Distance Transform Map）
    :param mask: 二值掩码（0为背景，1为前景），形状 [B, H, W]
    :param normalize: 是否归一化到[0,1]
    :return: 距离变换图，形状同mask
    """
    mask_np = mask.cpu().numpy()
    dist_maps = np.zeros_like(mask_np)

    for b in range(mask_np.shape[0]):
        foreground = mask_np[b] > 0.5

        if use_signed:
            # 有符号距离变换
            dist_foreground = distance_transform_edt(foreground)
            dist_background = distance_transform_edt(~foreground)
            dist = dist_foreground - dist_background
        else:
            # 无符号距离变换
            dist = distance_transform_edt(foreground)

        if normalize:
            dist_max = np.max(np.abs(dist)) if normalize else np.max(dist)
            dist = dist / (dist_max + 1e-8)

        dist_maps[b] = dist

    return torch.from_numpy(dist_maps).float().to(mask.device)


class DynamicWeightScheduler:
    def __init__(self, total_steps: int, mode: str = 'cosine', gamma_min: float = 0.1, gamma_max: float = 0.5):
        """
        :param total_steps: 总训练步数（总epoch * 每epoch步数）
        :param mode: 动态模式（'linear', 'cosine', 'exponential'）
        :param gamma_min: 初始权重
        :param gamma_max: 最大权重
        """
        self.total_steps = total_steps
        self.mode = mode
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

    def get_weight(self, current_step: int) -> float:
        """根据当前步数计算动态权重"""
        if self.mode == 'linear':
            return self.gamma_min + (self.gamma_max - self.gamma_min) * (current_step / self.total_steps)
        elif self.mode == 'cosine':
            return self.gamma_min + 0.5 * (self.gamma_max - self.gamma_min) * (
                        1 - np.cos(np.pi * current_step / self.total_steps))
        elif self.mode == 'exponential':
            lambda_ = 3.0  # 指数增长速率
            return self.gamma_min * np.exp(lambda_ * current_step / self.total_steps)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

class HybridLossWithDynamicBoundary:
    def __init__(self, total_steps: int = 150, alpha: float = 1.0, beta: float = 1.0,
                 gamma_min: float = 0.1, gamma_max: float = 0.5, mode: str = 'cosine'):
        """
        :param alpha: BCE权重
        :param beta: Dice权重
        :param gamma_min/gamma_max: 边界损失的动态权重范围
        :param mode: 动态权重模式
        """
        self.alpha = alpha
        self.beta = beta
        self.weight_scheduler = DynamicWeightScheduler(total_steps, mode, gamma_min, gamma_max)
        self.seg_loss = BCEDiceLoss()

    def _generate_boundary_weights(self, target: torch.Tensor, use_signed_distance: bool = False) -> torch.Tensor:
        """生成边界权重图"""
        signed_distance_map = generate_distance_map(
            target, use_signed=True)

        # 基于距离的权重
        if use_signed_distance:
            boundary_weights = torch.exp(-torch.abs(signed_distance_map))
        else:
            boundary_weights = 1.0 - torch.clamp(signed_distance_map, 0, 1)

        # 调整权重强度
        boundary_weights = torch.pow(boundary_weights, 1.0)
        boundary_weights = torch.clamp(boundary_weights, 0.1, 1.0)

        return boundary_weights


    def _boundary_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """基于距离变换的边界损失"""
        boundary_weights = self._generate_boundary_weights(target)
        # 焦点边界损失
        pred_sigmoid = torch.sigmoid(pred)
        bce_loss = F.binary_cross_entropy(pred_sigmoid, target, reduction='none')

        # 焦点权重
        focal_factor = torch.pow(torch.abs(pred_sigmoid - target), 2.0)

        # 组合损失
        boundary_loss = bce_loss * focal_factor * boundary_weights

        return boundary_loss.mean()

    def __call__(self, pred: torch.Tensor, target: torch.Tensor, current_step: int) -> torch.Tensor:
        # 计算各分量损失
        seg_loss = self.seg_loss(pred, target)
        boundary_loss = self._boundary_loss(pred, target)

        # 动态权重
        gamma_t = self.weight_scheduler.get_weight(current_step)

        # 混合损失
        total_loss = (self.alpha  + self.beta)/2 * seg_loss + gamma_t * boundary_loss
        return total_loss