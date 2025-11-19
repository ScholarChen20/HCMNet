import torch
import torch.nn.functional as F
import numpy as np
import cv2
from matplotlib import cm


class FeatureCAM:
    """
    对任意 feature map 进行 Grad-CAM 可视化
    target_layer: 你想 hook 的层，可为 nn.Module 或一个 Tensor（如 res_out[i]）
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.feature = None
        self.gradient = None

        # 注册前向钩子
        target_layer.register_forward_hook(self._forward_hook)
        # 注册反向钩子
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.feature = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradient = grad_output[0].detach()

    def generate_cam(self, input_image, target_index=None):
        """
        input_image: shape = [1,3,H,W]
        target_index: 针对输出通道监督（默认最大激活）
        """

        output = self.model(input_image)
        if isinstance(output, (list, tuple)):
            output = output[0]  # deep supervised, 取最终输出

        # 如果没有分类输出，选最大 activation
        if target_index is None:
            target_index = output.argmax()

        loss = output[:, target_index].sum()
        self.model.zero_grad()
        loss.backward()

        # feature: [C,H,W], gradient: [C,H,W]
        weight = self.gradient.mean(dim=(1, 2), keepdim=True)  # GAP
        cam = (weight * self.feature).sum(0)  # [H,W]

        cam = cam.clamp(min=0).cpu().numpy()
        cam = cam / cam.max()

        # 变成彩色 heatmap
        heatmap = (cm.jet(cam)[..., :3] * 255).astype(np.uint8)
        return heatmap, cam