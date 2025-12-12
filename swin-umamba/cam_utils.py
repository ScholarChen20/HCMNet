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


# ===============================
# 2. 统一的蓝底红区颜色映射 & 叠加
# ===============================
def to_jet_colormap(cam_01):
    """
    cam_01: numpy [h,w], 0~1
    return: heatmap_color: [h,w,3], uint8, Jet 蓝->红
    """
    cam_uint8 = np.uint8(cam_01 * 255)
    heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    return heatmap_rgb


def create_blue_overlay(img_bgr, cam_01, alpha=0.7):
    """
    img_bgr: 原图 BGR [H,W,3]
    cam_01: CAM 0~1 [h,w]（会 resize 到 img 大小）
    统一风格：蓝底 + 红色高响应
    """

    H, W = img_bgr.shape[:2]
    cam_resized = cv2.resize(cam_01, (W, H), interpolation=cv2.INTER_CUBIC)
    heatmap_rgb = to_jet_colormap(cam_resized)

    # 适当压暗热力图，让蓝底更明显
    heatmap_float = (heatmap_rgb.astype(np.float32) / 255.0) ** 0.9

    # 把原图也稍微压暗一些，避免盖住蓝底
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_float = img_rgb.astype(np.float32) / 255.0
    img_float = img_float * 0.6

    overlay = cv2.addWeighted(img_float, 1 - alpha, heatmap_float, alpha, 0)
    overlay = np.uint8(overlay * 255)
    # 最后再转回 BGR 方便 cv2.imwrite
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    return overlay_bgr


# ===============================
# 3. Mamba 分支：梯度注意力热力图
# ===============================
def gradient_based_heatmap_mamba(model, x, target_layer_idx=3):
    """
    Mamba / vmunet 分支的空间梯度热力图
    返回: numpy [H,W], 0~1
    """
    model.zero_grad()
    model.train()     # 要有梯度
    x = x.clone().detach().requires_grad_(True)

    # vmunet 返回的是多层特征列表，这里沿用你之前的写法
    features = model.vmunet(x)[target_layer_idx + 1]  # [B,C,h,w]

    # 选激活最强的通道
    channel_activations = features.mean(dim=[2, 3])  # [B,C]
    max_channel = channel_activations.argmax(dim=1)  # [B]

    # 以该通道的 mean 作为“目标”
    target_feat = features[0, max_channel].mean()
    target_feat.backward()

    # 使用输入梯度的平均作为空间热力图
    grad_heatmap = x.grad[0].mean(dim=0).abs()  # [H,W]
    grad_heatmap = grad_heatmap.cpu().numpy()
    grad_heatmap -= grad_heatmap.min()
    if grad_heatmap.max() > 0:
        grad_heatmap /= grad_heatmap.max()

    return grad_heatmap  # [H,W], 0~1

def enable_cnn_grad(model):
    # 只在可视化时解冻 cnnnet 参数，训练代码仍然可以保持 freeze
    if hasattr(model, "cnnnet"):
        for p in model.cnnnet.parameters():
            p.requires_grad_(True)

# ===============================
# 4. 单张图完整可视化 pipeline
# ===============================
def run_cam_for_image(file_name, save_root="./hotmap"):
    """
    对单张图生成:
    - feature_map_cnn.png
    - feature_map_fuse.png
    - feature_map_mamba.png
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------- 加载模型 -------
    model = MedFormer().to(device)
    # 按需加载权重：
    # state = torch.load("./output/MedMamba/BUSI/BUSI_pretrained_150_3.pth")
    # model.load_state_dict(state)
    model.eval()

    # ------- 读入图像 -------
    image_path = os.path.join("./data/BUSI/train/images", file_name)
    img = cv2.imread(image_path)  # BGR
    if img is None:
        raise FileNotFoundError(image_path)
    img = cv2.resize(img, (256, 256))
    img_t = torch.tensor(img / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

    # ------- 输出目录 -------
    save_path = os.path.join(save_root, os.path.splitext(file_name)[0])
    os.makedirs(save_path, exist_ok=True)

    # =======================
    # 4.1 CNN 分支 Grad-CAM
    # =======================
    # Hook ConvNeXt stage4 最后一个 Block 的 pwconv2
    enable_cnn_grad(model)
    target_layer_cnn = model.cnnnet.stages[3][-1].pwconv2
    cam_cnn = FeatureCAM(model, target_layer_cnn)
    cam_map_cnn = cam_cnn.generate_cam(img_t)          # [h,w],0~1
    overlay_cnn = create_blue_overlay(img, cam_map_cnn)
    cv2.imwrite(os.path.join(save_path, "feature_map_cnn.png"), overlay_cnn)
    print("Saved:", os.path.join(save_path, "feature_map_cnn.png"))

    # =======================
    # 4.2 Fuse(TIF) 分支 Grad-CAM
    # =======================
    # Hook TIF 第4层 (索引3) 中 conv_l 的卷积
    target_layer_fuse = model.Fuse[3].conv_l[0]        # 5x5 Conv2d
    cam_fuse = FeatureCAM(model, target_layer_fuse)
    cam_map_fuse = cam_fuse.generate_cam(img_t)
    overlay_fuse = create_blue_overlay(img, cam_map_fuse)
    cv2.imwrite(os.path.join(save_path, "feature_map_fuse.png"), overlay_fuse)
    print("Saved:", os.path.join(save_path, "feature_map_fuse.png"))

    # =======================
    # 4.3 Mamba 分支梯度热力图
    # =======================
    cam_map_mamba = gradient_based_heatmap_mamba(model, img_t, target_layer_idx=3)
    overlay_mamba = create_blue_overlay(img, cam_map_mamba)
    cv2.imwrite(os.path.join(save_path, "feature_map_mamba.png"), overlay_mamba)
    print("Saved:", os.path.join(save_path, "feature_map_mamba.png"))


# ===============================
# 5. 论文排版用的可视化（选用）
# ===============================
def plot_heatmap_grid(hotmap_dir="./hotmap"):
    """
    把 ./hotmap 下多个 case 画成 Figure 11 那种排版:
    Input | GT | cnn | fuse | mamba
    """
    name_map = {
        'feature_map_cnn.png': '(c)',
        'feature_map_fuse.png': '(d)',
        'feature_map_mamba.png': '(e)'
    }

    folders = [f for f in os.listdir(hotmap_dir)
               if os.path.isdir(os.path.join(hotmap_dir, f))
               and f not in ['images', 'masks']]

    image_names = ['feature_map_cnn.png', 'feature_map_fuse.png', 'feature_map_mamba.png']

    rows = len(folders)
    fig, axes = plt.subplots(rows, 5, figsize=(10, 2 * rows))

    if rows == 1:
        axes = axes.reshape(1, -1)

    for i, folder in enumerate(folders):
        # (a) Input
        img_path_in = os.path.join(hotmap_dir, "images", folder + ".png")
        img_in = cv2.imread(img_path_in)
        if img_in is not None:
            img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)
            img_in = cv2.resize(img_in, (224, 224))
            axes[i, 0].imshow(img_in)
            if i == 0:
                axes[i, 0].set_title("(a)")
        else:
            axes[i, 0].text(0.5, 0.5, 'Input not found', ha='center', va='center', color='red')
        axes[i, 0].axis('off')

        # (b) GT
        img_path_gt = os.path.join(hotmap_dir, "masks", folder + ".png")
        img_gt = cv2.imread(img_path_gt)
        if img_gt is not None:
            img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
            img_gt = cv2.resize(img_gt, (224, 224))
            axes[i, 1].imshow(img_gt)
            if i == 0:
                axes[i, 1].set_title("(b)")
        else:
            axes[i, 1].text(0.5, 0.5, 'Mask not found', ha='center', va='center', color='red')
        axes[i, 1].axis('off')

        # (c)(d)(e) 三个热力图
        folder_path = os.path.join(hotmap_dir, folder)
        for j, img_name in enumerate(image_names):
            p = os.path.join(folder_path, img_name)
            ax = axes[i, j + 2]
            if os.path.exists(p):
                img = cv2.imread(p)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224, 224))
                ax.imshow(img)
                if i == 0:
                    ax.set_title(name_map[img_name])
            else:
                ax.text(0.5, 0.5, 'Image not found', ha='center', va='center', color='red')
            ax.axis('off')

    plt.tight_layout()
    plt.savefig('heatmap_summary.png', dpi=600, bbox_inches='tight')
    print("Saved heatmap summary to heatmap_summary.png")
    plt.show()


def feature_to_heatmap(feature):
    """
    feature: [C,H,W]
    return: heatmap RGB 0~255
    """
    feature = feature.detach().cpu()

    # channel-wise mean activation
    heat = feature.abs().mean(dim=0)  # [H,W]

    # normalize to 0~1
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    heat_np = heat.numpy()

    # apply color map (jet)
    heat_color = cv2.applyColorMap(np.uint8(255 * heat_np), cv2.COLORMAP_JET)
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)

    return heat_color

def overlay(img, heatmap, alpha=0.6):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    heatmap = heatmap.astype(np.float32) / 255.0

    blended = img * (1 - alpha) + heatmap * alpha
    return np.uint8(blended * 255)

def visualize_attention(model, file_name):
    model.eval()

    img_path = f"./data/BUSI/train/images/{file_name}"
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))

    img_t = torch.tensor(img/255, dtype=torch.float32).permute(2,0,1).unsqueeze(0).cuda()

    # forward with features
    _, f_cnn, f_mamba, f_fuse = model(img_t)

    # convert to heatmaps
    h_cnn  = feature_to_heatmap(f_cnn.squeeze(0))
    h_mam  = feature_to_heatmap(f_mamba.squeeze(0))
    h_fuse = feature_to_heatmap(f_fuse.squeeze(0))

    # overlay
    o_cnn  = overlay(img, cv2.resize(h_cnn, (256,256)))
    o_mam  = overlay(img, cv2.resize(h_mam, (256,256)))
    o_fuse = overlay(img, cv2.resize(h_fuse, (256,256)))

    # save
    save_dir = f"./hotmap/{file_name.split('.')[0]}"
    os.makedirs(save_dir, exist_ok=True)

    cv2.imwrite(f"{save_dir}/cnn.png",  cv2.cvtColor(o_cnn, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"{save_dir}/mamba.png",cv2.cvtColor(o_mam, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"{save_dir}/fuse.png", cv2.cvtColor(o_fuse, cv2.COLOR_RGB2BGR))

    print("Saved heatmaps to", save_dir)

def plot_panel(file_names):
    fig, ax = plt.subplots(len(file_names), 5, figsize=(10, 12))

    for i, f in enumerate(file_names):
        folder = f"./hotmap/{f.split('.')[0]}"
        img = cv2.imread(f"./data/BUSI/train/images/{f}")
        img = cv2.resize(img, (256, 256))
        mask = cv2.imread(f"./data/BUSI/train/masks/{f}")
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
    plt.savefig("./hotmap/BUS/summary.png", dpi=500)
    plt.show()



# ===============================
# 6. 测试入口
# ===============================
if __name__ == "__main__":
    # 示例：单张图片
    # run_cam_for_image("benign (13).png")

    # list = ["000021.png", "000022.png", "000041.png", "000050.png", "000093.png"]
    list = ["benign (13).png", "benign (26).png", "benign (174).png", "benign (300).png", "benign (326).png"]
    # model = BCMamba().cuda()
    # 按需加载权重：
    # state = torch.load("./output/BCMamba/BUSI/BUSI_pretrained_150_1.pth")
    # model.load_state_dict(state)
    # model.eval()
    # for fn in list:
    #     visualize_attention(model, fn)

    plot_panel(list)

    # 若要多张并排作图：
    # names = ["benign (13).png", "benign (26).png", ...]
    # for fn in names:
    #     run_cam_for_image(fn)
    # plot_heatmap_grid()
