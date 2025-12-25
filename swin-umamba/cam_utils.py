import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from nets.BCMamba import MedFormer,BCMamba  # 或 BCMamba，看你实际用哪个

# ===============================
# 1. Grad-CAM 模块（CNN / Fuse 用）
# ===============================
class FeatureCAM:
    """
    对任意 feature map 进行 Grad-CAM 可视化
    支持分割模型（B,1,H,W）或分类模型
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.feature = None
        self.gradient = None

        # 前向 hook：拿到特征
        target_layer.register_forward_hook(self._forward_hook)
        # 反向 hook：拿到梯度
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        # 保存一份特征，detach 不影响反向传播
        self.feature = out.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        # grad_output[0] 就是对该层输出的梯度
        self.gradient = grad_output[0].detach()


    def generate_cam(self, input_image, target_index=None):
        """
        input_image: [1,3,H,W]
        target_index: 对输出通道的索引（分割模型默认 0）
        return: cam_heat (H',W'，归一化 0~1)
        """
        self.model.zero_grad()
        self.feature = None
        self.gradient = None

        # ---- 前向 ----
        output = self.model(input_image)

        # 分割模型：输出通常为 [B,1,H,W] 或 List[tensor]
        if isinstance(output, (list, tuple)):
            output = output[0]

        if output.dim() == 4:
            # segmentation: [B,C,H,W]
            if target_index is None:
                target_index = 0
            channel_output = output[:, target_index]  # [B,H,W]
            loss = channel_output.mean()
        else:
            # classification: [B,num_classes]
            if target_index is None:
                target_index = output.argmax()
            loss = output[:, target_index].sum()

        # ---- 反向 ----
        loss.backward()

        if self.feature is None:
            raise RuntimeError("Feature is None. Check forward hook.")
        if self.gradient is None:
            raise RuntimeError("Gradient is None. Check backward hook.")

        # self.feature: [B,C,h,w]
        feat = self.feature
        grad = self.gradient

        # GAP 得到每个通道的权重
        # [B,C,1,1]
        weight = grad.mean(dim=(2, 3), keepdim=True)

        # [B,1,h,w] -> 取 batch 0
        cam = (weight * feat).sum(dim=1, keepdim=False)[0]  # [h,w]

        cam = cam.clamp(min=0)
        cam = cam.cpu().numpy()
        cam_max = cam.max()
        if cam_max > 0:
            cam = cam / cam_max
        else:
            cam = np.zeros_like(cam)

        return cam  # 0~1，shape [h,w]


def paper_overlay(img_rgb, heatmap_01, alpha=0.65):
    """
    img_rgb: [H,W,3] RGB
    heatmap_01: [h,w] 0~1
    """
    H, W = img_rgb.shape[:2]
    heatmap = cv2.resize(heatmap_01, (W, H), interpolation=cv2.INTER_CUBIC)

    # jet colormap
    heatmap_color = cv2.applyColorMap(
        np.uint8(255 * heatmap),
        cv2.COLORMAP_JET
    )
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # 压暗原图，突出蓝底
    img = img_rgb.astype(np.float32) / 255.0
    img = img * 0.6

    heatmap_color = heatmap_color.astype(np.float32) / 255.0

    overlay = img * (1 - alpha) + heatmap_color * alpha
    return np.uint8(overlay * 255)



def feature_to_heatmap(feature, topk_ratio=0.2, power=1.5):
    """
    feature: torch.Tensor [C,H,W]
    topk_ratio: 只使用响应最强的 top-k 通道（论文常用 10%~30%）
    power: >1 聚焦高响应区域
    return: heatmap [H,W] in 0~1
    """
    feature = feature.detach().cpu()  # [C,H,W]

    C, H, W = feature.shape

    # ---------- 1. 通道能量 ----------
    channel_energy = feature.abs().mean(dim=(1, 2))  # [C]

    # ---------- 2. 选 Top-K 通道 ----------
    k = max(1, int(C * topk_ratio))
    topk_idx = torch.topk(channel_energy, k=k, largest=True).indices

    # ---------- 3. 加权融合 ----------
    heat = torch.zeros((H, W))
    for idx in topk_idx:
        heat += feature[idx].abs()

    heat /= k

    # ---------- 4. 聚焦高响应 ----------
    heat = torch.relu(heat)
    heat = heat ** power

    # ---------- 5. normalize ----------
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)

    return heat.numpy()  # [H,W] 0~1


def visualize_attention(model, file_name):
    model.eval()

    # ---------- load image ----------
    img_path = f"./data/BUS/train/images/{file_name}"
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (256, 256))

    img_t = torch.tensor(
        img_rgb / 255.0,
        dtype=torch.float32
    ).permute(2, 0, 1).unsqueeze(0).cuda()

    # ---------- forward ----------
    with torch.no_grad():
        _, f_cnn, f_mamba, f_fuse = model(img_t)

    # ---------- feature → heatmap ----------
    h_cnn  = feature_to_heatmap(f_cnn.squeeze(0),  topk_ratio=0.1, power=1.0)
    h_mam  = feature_to_heatmap(f_mamba.squeeze(0),topk_ratio=0.3, power=1.6)
    h_fuse = feature_to_heatmap(f_fuse.squeeze(0) + f_mamba.squeeze(0)/2 + f_cnn.squeeze(0)/2, topk_ratio=0.25, power=2.0)

    # ---------- overlay ----------
    o_cnn  = paper_overlay(img_rgb, h_cnn)
    o_mam  = paper_overlay(img_rgb, h_mam)
    o_fuse = paper_overlay(img_rgb, h_fuse)

    # ---------- save ----------
    save_dir = f"./hotmap/{file_name.split('.')[0]}"
    os.makedirs(save_dir, exist_ok=True)

    cv2.imwrite(f"{save_dir}/cnn.png",   cv2.cvtColor(o_cnn,  cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"{save_dir}/mamba.png", cv2.cvtColor(o_mam,  cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"{save_dir}/fuse.png",  cv2.cvtColor(o_fuse, cv2.COLOR_RGB2BGR))

    print("Saved feature heatmaps to", save_dir)



def plot_panel(file_names):
    fig, ax = plt.subplots(len(file_names), 5, figsize=(8, 6.5))

    for i, f in enumerate(file_names):
        folder = f"./hotmap/{f.split('.')[0]}"
        img = cv2.imread(f"./data/BUS/train/images/{f}")
        img = cv2.resize(img, (256, 256))
        mask = cv2.imread(f"./data/BUS/train/masks/{f}")
        mask = cv2.resize(mask, (256, 256))

        cnn = cv2.imread(f"{folder}/cnn.png")
        mam = cv2.imread(f"{folder}/mamba.png")
        fuse = cv2.imread(f"{folder}/fuse.png")

        # (a) Input
        ax[i][0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if i == 0:
            ax[i][0].set_title("(a)")

        # (b) GT
        ax[i][1].imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
        if i == 0:
            ax[i][1].set_title("(b)")
        # (c) CNN
        ax[i][2].imshow(cv2.cvtColor(cnn, cv2.COLOR_BGR2RGB))
        if i == 0:
            ax[i][2].set_title("(c)")

        # (d) Mamba
        ax[i][3].imshow(cv2.cvtColor(mam, cv2.COLOR_BGR2RGB))
        if i == 0:
            ax[i][3].set_title("(d)")

        # (e) Fuse
        ax[i][4].imshow(cv2.cvtColor(fuse, cv2.COLOR_BGR2RGB))
        if i == 0:
            ax[i][4].set_title("(e)")

        # 关闭坐标轴
        for j in range(5):
            ax[i][j].axis('off')

    plt.tight_layout()
    plt.savefig("./hotmap/BUS/summary_3.png", dpi=500)
    plt.show()



# ===============================
# 6. 测试入口
# ===============================
if __name__ == "__main__":
    # 示例：单张图片
    # run_cam_for_image("benign (13).png")

    list = ["000021.png", "000041.png", "000050.png", "000093.png"]
    # list = ["benign (13).png", "benign (26).png", "benign (174).png", "benign (300).png", "benign (326).png"]
    # model = MedFormer(lora_rank=16, deep_supervised=True).cuda()
    # 按需加载权重：
    # state = torch.load("./output/MedMamba/BUS/BUS_pretrained_150_4.pth")
    # state = torch.load("./output/MedMamba/Ablation/BUSI/BUSI_pretrained_150_rank16_4.pth")
    # state = torch.load("./output/MedMamba/BUSI/BUSI_pretrained_150_4.pth")
    # model.load_state_dict(state)
    # model.eval()
    # for fn in list:
        # visualize_attention(model, fn)

    plot_panel(list)

    # 若要多张并排作图：
    # names = ["benign (13).png", "benign (26).png", ...]
    # for fn in names:
    #     run_cam_for_image(fn)
    # plot_heatmap_grid()
