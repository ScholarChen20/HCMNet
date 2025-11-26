import os
import numpy as np
import cv2
from matplotlib import cm
import cv2
from nets.BCMamba import BCMamba, load_pretrained_ckpt,MedFormer, load_from
import torch
import torch.nn.functional as F

class FeatureCAM:
    """
    对任意 feature map 进行 Grad-CAM 可视化
    支持分类模型和分割模型
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
        # 前向传播
        output = self.model(input_image)[0]

        # 处理分割模型输出
        if len(output.shape) == 4:  # [B, C, H, W]
            if target_index is None:
                target_index = 0
            # 选择特定的输出通道
            channel_output = output[:, target_index]  # [B, H, W]
            # 创建损失函数
            loss = channel_output.mean()
        else:  # 分类模型
            if isinstance(output, (list, tuple)):
                output = output[0]
            if target_index is None:
                target_index = output.argmax()
            loss = output[:, target_index].sum()

        # 反向传播
        self.model.zero_grad()
        loss.backward()

        # 检查特征和梯度
        if self.feature is None:
            raise RuntimeError("Feature is None. Check forward hook.")
        if self.gradient is None:
            raise RuntimeError("Gradient is None. Check backward hook.")

        print(f"Feature shape: {self.feature.shape}")
        print(f"Gradient shape: {self.gradient.shape}")

        # 修正：在空间维度上平均 (dim=2和3)
        weight = self.gradient.mean(dim=(2, 3), keepdim=True)  # [1, 384, 1, 1]
        print(f"Weight shape: {weight.shape}")

        # 计算CAM
        cam = (weight * self.feature).sum(dim=1, keepdim=False)  # [1, 16, 16]
        cam = cam.squeeze(0)  # 移除batch维度 [16, 16]
        print(f"CAM shape: {cam.shape}")

        # 处理CAM
        cam = cam.clamp(min=0).cpu().numpy()
        cam_max = cam.max()
        if cam_max > 0:
            cam = cam / cam_max
        else:
            print("Warning: CAM max is 0, normalization skipped")
            # 创建一个空热力图
            heatmap = np.zeros((16, 16, 3), dtype=np.uint8)
            return heatmap, cam

        # 生成热力图
        heatmap = (cm.jet(cam)[..., :3] * 255).astype(np.uint8)
        print(f"Heatmap shape: {heatmap.shape}")
        return heatmap, cam

def main(file_name, type):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MedFormer().to(device)
    load_from(model)
    # model.load_state_dict(torch.load("./output/BCMamba/BUSI/BUSI_pretrained_150_1.pth"))
    model.eval()

    # 1. 输入图像
    image_path = "./data/BUSI/train/images/" + file_name
    img = cv2.imread(image_path)
    img = cv2.resize(img, (256, 256))
    img_t = torch.tensor(img / 255, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
    img_t.requires_grad_(True)  # 设置需要梯度，以便反向传播

    # 2. 先跑一次得到中间特征（不需要梯度，因为我们只是要获取目标层的位置，但这次前向传播并不用于GradCAM）
    save_path = os.path.join("./hotmap/", file_name.split(".")[0])
    os.makedirs(save_path, exist_ok=True)  # 创建保存热力图的文件夹,路径为./hotmap/file_name

    # 3. Hook res_out[2]
    target_layer = model.cnnnet.stages[3]
    cam = FeatureCAM(model, target_layer)
    heatmap, cam_raw = cam.generate_cam(img_t, target_index=0)  # 在generate_cam内部会进行前向和反向传播
    heatmap_resized = cv2.resize(heatmap, (256, 256))
    overlay = cv2.addWeighted(img, 0.5, heatmap_resized, 0.5, 0)
    cv2.imwrite(os.path.join(save_path, "feature_map_cnn.png"), overlay)
    print("Saved heatmap to", save_path + "/feature_map_cnn.png")

    target_layer = model.Fuse[2]
    cam = FeatureCAM(model, target_layer)
    heatmap, cam_raw = cam.generate_cam(img_t, target_index=0)  # 在generate_cam内部会进行前向和反向传播
    heatmap_resized = cv2.resize(heatmap, (256, 256))
    overlay = cv2.addWeighted(img, 0.5, heatmap_resized, 0.5, 0)
    cv2.imwrite(os.path.join(save_path, "feature_map_fuse.png"), overlay)
    print("Saved heatmap to", save_path + "/feature_map_fuse.png")


    heatmap_tensor = spatial_attention_heatmap(model, img_t, target_layer_idx=3)
    heatmap_resized = process_heatmap_for_visualization(heatmap_tensor, (256, 256))

    overlay = create_overlay(img, heatmap_resized)
    cv2.imwrite(os.path.join(save_path, "feature_map_mamba.png"), overlay)
    print("Saved heatmap to", save_path + "/feature_map_mamba.png")


def spatial_attention_heatmap(model, x, target_layer_idx=3):
    """基于空间注意力的热力图"""
    # features = model.vssm_encoder(x)[target_layer_idx]  # [1, 192, 32, 32]
    features = model.vmunet(x)[target_layer_idx]  # [1, 192, 32, 32]

    # 计算空间注意力（每个位置在所有通道上的响应）
    spatial_attention = features.abs().mean(dim=1)  # [1, 32, 32]

    # 增强对比度
    heatmap = F.relu(spatial_attention)  # 去除负激活
    heatmap = heatmap ** 2  # 增强高激活区域

    return heatmap.squeeze(0)


def process_heatmap_for_visualization(heatmap_tensor, target_size=(256, 256)):
    """将热力图张量转换为适合可视化的numpy数组"""
    # 确保是CPU和numpy
    if heatmap_tensor.is_cuda:
        heatmap_tensor = heatmap_tensor.cpu()

    # 移除批次维度并转换为numpy
    if heatmap_tensor.dim() == 3:
        heatmap_np = heatmap_tensor.squeeze(0).detach().numpy()  # [H, W]
    else:
        heatmap_np = heatmap_tensor.detach().numpy()  # 已经是2D

    print(f"Heatmap numpy shape: {heatmap_np.shape}")

    # 归一化到0-1
    heatmap_np = (heatmap_np - heatmap_np.min()) / (heatmap_np.max() - heatmap_np.min() + 1e-8)

    # 调整大小
    heatmap_resized = cv2.resize(heatmap_np, target_size, interpolation=cv2.INTER_CUBIC)

    return heatmap_resized


def gradient_based_heatmap(model, x, target_layer_idx=3):
    """基于梯度的热力图（需要模型在训练模式）"""
    model.train()
    x.requires_grad_(True)

    features = model.vssm_encoder(x)[target_layer_idx]

    # 选择激活最强的通道
    channel_activations = features.mean(dim=[2, 3])  # [1, 192]
    max_channel = channel_activations.argmax(dim=1)

    # 计算梯度
    model.zero_grad()
    features[0, max_channel].mean().backward()

    # 使用梯度作为热力图
    grad_heatmap = x.grad[0].mean(dim=0).abs()  # [H, W]

    return grad_heatmap

def create_overlay(img, heatmap, alpha=0.7):
    """创建热力图叠加"""
    # 确保img和heatmap的尺寸匹配
    print(f"Image shape: {img.shape}")
    print(f"Heatmap shape: {heatmap.shape}")

    # 如果img是彩色(RGB)，需要将热力图转换为3通道
    if len(img.shape) == 3 and img.shape[2] == 3:
        # 将热力图转换为彩色(Jet colormap)
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # 确保img是float类型
        img_float = img.astype(np.float32) / 255.0
        heatmap_float = heatmap_colored.astype(np.float32) / 255.0

        # 叠加
        overlay = cv2.addWeighted(img_float, 1 - alpha, heatmap_float, alpha, 0)
        overlay = np.uint8(overlay * 255)
    else:
        # 灰度图像
        img_float = img.astype(np.float32) / 255.0
        heatmap_float = heatmap.astype(np.float32)

        # 叠加
        overlay = cv2.addWeighted(img_float, 1 - alpha, heatmap_float, alpha, 0)
        overlay = np.uint8(overlay * 255)

    return overlay

if __name__ == '__main__':
    # main("benign (13).png", "fuse")
    # main("benign (26).png", "cnn")
    # main("benign (195).png", "mamba")
    main("benign (326).png", "mamba")
    # main("malignant (185).png", "mamba")

