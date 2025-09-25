import matplotlib.pyplot as plt
import numpy as np
import cv2
import os



def load_images(input_dir, gt_dir, pred_dirs):
    """
    加载原始图像、Ground Truth 和多个分割结果。
    """
    input_images = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.png')])  #jpg
    gt_images = sorted([os.path.join(gt_dir, f) for f in os.listdir(gt_dir) if f.endswith('.png')])
    pred_images = {method: sorted([os.path.join(pred_dir, f) for f in os.listdir(pred_dir) if f.endswith('.png')])
                   for method, pred_dir in pred_dirs.items()}
    return input_images, gt_images, pred_images


def ablation_segmentation_results(input_images, gt_images, pred_images, methods):
    num_samples = len(input_images)
    num_methods = len(methods)
    fig, axes = plt.subplots(num_samples, num_methods + 1, figsize=(10, num_samples * 1.7))
    plt.subplots_adjust(wspace=0.05, hspace=0.05, left=0.05, right=0.95)

    for i in range(num_samples):
        # ======= 修复第一列：超声原图+真实边界 =======
        # 读取并处理输入图像
        input_img = cv2.cvtColor(cv2.imread(input_images[i]), cv2.COLOR_BGR2RGB)
        input_img = cv2.resize(input_img, (256, 256))

        # 读取并处理GT掩码
        gt_mask = cv2.imread(gt_images[i], cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.resize(gt_mask, (256, 256))

        # 修复轮廓检测（OpenCV 4.x返回两个值）
        contours, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 创建轮廓叠加层（使用透明背景）
        overlay = np.zeros_like(input_img)
        cv2.drawContours(overlay, contours, -1, (255, 0, 0), 2)  # 红色轮廓 (RGB顺序)

        # 正确的图像混合（原图+轮廓）
        blended = cv2.addWeighted(input_img, 1, overlay, 0.8, 0)  # 调整透明度

        axes[i, 0].imshow(blended)
        if i == 0:
            axes[i, 0].set_title("Input with GT", fontsize=12)
        axes[i, 0].axis('off')
        # ======= 预测结果列 =======
        for col_idx, method in enumerate(methods, start=1):
            # 调试输出路径
            # print(f"Processing: Sample {i + 1}, Method {method}")
            # print(f"Pred mask path: {pred_images[method][i]}")

            pred_mask_path = pred_images[method][i]
            pred_mask = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)

            # 检查图像是否加载成功
            if pred_mask is None:
                # print(f"!!! ERROR: Failed to load image: {pred_mask_path}")
                # 创建空图像占位符
                canvas = np.zeros((256, 256, 3), dtype=np.uint8)
                axes[i, col_idx].imshow(canvas)
                axes[i, col_idx].set_title("Load Error", fontsize=10, color='red')
                axes[i, col_idx].axis('off')
                continue

            # 检查图像内容
            unique_vals = np.unique(pred_mask)
            # print(f"Unique values in mask: {unique_vals}")
            # print(f"Min value: {np.min(pred_mask)}, Max value: {np.max(pred_mask)}")

            pred_mask = cv2.resize(pred_mask, (256, 256))
            canvas = np.zeros((256, 256, 3), dtype=np.uint8)

            # 确保GT掩码已定义
            gt_mask = cv2.resize(cv2.imread(gt_images[i], cv2.IMREAD_GRAYSCALE), (256, 256))

            # 计算不同分割区域
            over_segmented = np.logical_and(pred_mask > 0, gt_mask == 0)
            under_segmented = np.logical_and(pred_mask == 0, gt_mask > 0)
            correctly_segmented = np.logical_and(pred_mask > 0, gt_mask > 0)

            # 调试输出区域大小
            # print(f"Over-segmented pixels: {np.sum(over_segmented)}")
            # print(f"Under-segmented pixels: {np.sum(under_segmented)}")
            # print(f"Correctly-segmented pixels: {np.sum(correctly_segmented)}")
            # print(
                # f"Total marked pixels: {np.sum(over_segmented) + np.sum(under_segmented) + np.sum(correctly_segmented)}")

            # 着色 - 使用更鲜艳的颜色
            canvas[over_segmented] = [255, 0, 0]  # 红色：过分割
            canvas[under_segmented] = [0, 255, 0]  # 绿色：欠分割
            canvas[correctly_segmented] = [255, 255, 255]  # 白色：正确分割

            # 暂时不使用透明度，直接显示
            axes[i, col_idx].imshow(canvas)

            if i == 0:
                axes[i, col_idx].set_title(method, fontsize=12)
            axes[i, col_idx].axis('off')

    plt.tight_layout(pad=1.0)
    plt.show()



def plot_segmentation_results(input_images, gt_images, pred_images, methods, dataset_name, save_path=None):
    """
    七列可视化布局：
    列1：超声原图+真实边界(红色)
    列2-7：各方法预测(绿色)与真实(红色)掩码叠加
    """
    num_samples = len(input_images)
    num_methods = len(methods)
    assert num_methods == 10, "需要10种分割方法"

    fig, axes = plt.subplots(num_samples, 11, figsize=(15, num_samples * 1.7))  #``figsize``控制图的大小,行距离、列距离
    plt.subplots_adjust(wspace=0.05, hspace=0.05,left=0.05, right=0.95,top=0.95, bottom=0.05) #left=0.05,减少左边距,上边距

    for i in range(num_samples):
        print(f"\n=== 样本 {i + 1} ===")
        print(f"输入图像: {os.path.basename(input_images[i])}")
        print(f"GT图像: {os.path.basename(gt_images[i])}")

        # ==================== 第一列：超声原图+真实边界 ====================
        # 读取并调整原图尺寸
        input_img = cv2.cvtColor(cv2.imread(input_images[i]), cv2.COLOR_BGR2RGB)
        input_img = cv2.resize(input_img, (256, 256))

        # 提取真实掩码边界
        gt_mask = cv2.imread(gt_images[i], cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.resize(gt_mask, (256, 256))
        contours, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 在原图上绘制红色边界
        overlay = input_img.copy()
        cv2.drawContours(overlay, contours, -1, (255, 0, 0), 2)
        alpha = 0.6  # 透明度
        blended = cv2.addWeighted(overlay, alpha, input_img, 1 - alpha, 0)

        axes[i, 0].imshow(blended)
        if i == 0:
            axes[i, 0].set_title("Input with GT", fontsize=12)
        axes[i, 0].axis('off')

        # # 在每行第一个子图的左侧添加文本标注（域名称），缩短左边距
        # axes[i, 0].text(-0.15, 0.5, dataset_name[i], transform=axes[i, 0].transAxes,
        #                 rotation='vertical', verticalalignment='center', fontsize=12)

        # ==================== 第二到第七列：方法预测结果 ====================
        for col_idx, method in enumerate(methods, start=1):
            # 读取预测掩码
            pred_mask = cv2.imread(pred_images[method][i], cv2.IMREAD_GRAYSCALE)
            pred_mask = cv2.resize(pred_mask, (256, 256))

            # 创建RGB画布
            canvas = np.zeros((256, 256, 3), dtype=np.uint8)

            # # 绘制真实掩码（红色半透明）
            # canvas[gt_mask > 0] = [255, 0, 0]  # BGR->RGB转换
            # # 绘制预测掩码（绿色半透明）
            # canvas[pred_mask > 0] = [0, 255, 0]

            # (修改后的)过度细分（预测为1，真实为0） -> 红色
            over_segmented = np.logical_and(pred_mask > 0, gt_mask == 0)
            canvas[over_segmented] = [255, 0, 0]

            # 细分不足（预测为0，真实为1） -> 绿色
            under_segmented = np.logical_and(pred_mask == 0, gt_mask > 0)
            canvas[under_segmented] = [0, 255, 0]

            # 正确细分（预测为1，真实为1） -> 白色
            correctly_segmented = np.logical_and(pred_mask > 0, gt_mask > 0)
            canvas[correctly_segmented] = [255, 255, 255]
            # 设置透明度
            alpha_mask = np.zeros_like(pred_mask)
            alpha_mask[over_segmented | under_segmented | correctly_segmented] = 50

            # 设置透明度
            alpha_mask = np.zeros_like(canvas)
            alpha_mask[gt_mask > 0] = 50  # 真实掩码透明度
            alpha_mask[pred_mask > 0] = 50  # 预测掩码透明度
            alpha_mask = cv2.cvtColor(alpha_mask, cv2.COLOR_BGR2GRAY)

            # 显示叠加结果
            axes[i, col_idx].imshow(canvas, alpha=alpha_mask / 255.0)
            if i == 0:
                axes[i, col_idx].set_title(method, fontsize=12)
            axes[i, col_idx].axis('off')


    # 保存或显示
    plt.tight_layout(pad=1.0)#避免图片被压缩
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()


def origin_segmentation_results(input_images, gt_images, pred_images, methods, save_path=None):
    """
    绘制分割结果。
    """
    num_samples = len(input_images)               #5张图像
    num_methods = len(methods)                #5种方法

    fig, axes = plt.subplots(num_samples, num_methods + 2, figsize=(num_methods * 2, num_samples*2-3))
    plt.subplots_adjust(wspace=0.01, hspace=0.01)

    for i in range(num_samples):
        # 原始输入图像
        input_img = cv2.imread(input_images[i])
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        input_img = cv2.resize(input_img, (224, 224))
        axes[i, 0].imshow(input_img)
        if i==0:
            axes[i, 0].set_title("Input",fontsize=10)
        axes[i, 0].axis('off')

        # Ground Truth
        gt_img = cv2.imread(gt_images[i], cv2.IMREAD_GRAYSCALE)
        gt_img = cv2.resize(gt_img, (256,256))
        axes[i, 1].imshow(gt_img, cmap='gray')
        if i == 0:
            axes[i, 1].set_title("GT",fontsize=10)
        axes[i, 1].axis('off')

        # 分割结果
        for j, method in enumerate(methods):
            pred_img = cv2.imread(pred_images[method][i], cv2.IMREAD_GRAYSCALE)
            pred_img = cv2.resize(pred_img, (256,256))
            axes[i, j + 2].imshow(pred_img, cmap='gray')
            if i == 0:
                axes[i, j + 2].set_title(method,fontsize=10)
            axes[i, j + 2].axis('off')

    plt.tight_layout(pad=0.5)
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def save_imgs_from_path(img_path, msk_path, msk_pred_path, i, save_path, datasets, threshold=0.5, test_data_name=None):
    # Load the image
    img = Image.open(img_path)
    img = np.array(img) / 255.0

    # Load the mask and prediction mask
    msk = Image.open(msk_path)
    msk = np.array(msk)

    msk_pred = Image.open(msk_pred_path)
    msk_pred = np.array(msk_pred)

    # Process masks based on dataset type
    if datasets == 'retinal':
        pass  # Retinal data assumed to be correctly loaded as 2D masks
    else:
        msk = np.where(msk > 0.5, 1, 0)
        msk_pred = np.where(msk_pred > threshold, 1, 0)

    # Plot and save the images
    fig, axes = plt.subplots(1, 3, figsize=(8, 4))

    # Plot the original image
    axes[0].imshow(img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Plot the ground truth mask
    axes[1].imshow(msk, cmap='gray')
    axes[1].set_title('Ground Truth Mask')
    axes[1].axis('off')

    # Plot the predicted mask
    axes[2].imshow(msk_pred, cmap='gray')
    axes[2].set_title('Predicted Mask')
    axes[2].axis('off')

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

def show():
    """
    单张图片掩码和模型预测图像展示
    """
    input_images = '/home/cwq/MedicalDP/SwinUmamba/data/nnUNet_raw/Dataset705_Thyroid/test/images/L4-0013-5.jpg'
    gt_images = '/home/cwq/MedicalDP/SwinUmamba/data/nnUNet_raw/Dataset705_Thyroid/test/masks/L4-0013-5.png'
    msk_pred='./output/VMUNetv2/ph1/L4-0013-5.png'
    # input_images = cv2.imread(input_images, cv2.COLOR_BGRA2BGR)
    # gt_images = cv2.imread(gt_images,cv2.IMREAD_GRAYSCALE)
    # msk_pred=cv2.imread(msk_pred,cv2.IMREAD_GRAYSCALE)
    save_imgs_from_path(img_path=input_images, msk_path=gt_images,msk_pred_path=msk_pred,i=0, datasets="None", threshold=0.5,
              test_data_name='test', save_path=None)


if __name__ == '__main__':
    dataset_name = ['Site1','Site2','Site3','Site4']
    # methods = ['DeepAll', 'FedDG', 'DoFe', 'RAM-DSIR', 'TriD', 'DDG(Ours)']
    methods = ['UNet', 'UNet++', 'AAU-net', 'Swin-UNet', 'HiFormer-L', 'H2Former', 'BEFUNet', 'U-Mamba', 'Swin-UMamba','Ours']  # 对比方法
    ablation_methods = ['baseline', 'w/o CNN', 'w/o TiFusion', 'w/o MWFFD', 'w/o Loss', 'BC-Mamba']

    type = "Comparison"

    if type == "Comparison":
        # 对比实验图像路径
        input_dir = './visualize/VisualizeResult/images'
        gt_dir = './visualize/VisualizeResult/masks'
        pred_dirs = {
            'UNet': './visualize/VisualizeResult/UNet',
            'UNet++': './visualize/VisualizeResult/UNet++',
            'AAU-net': './visualize/VisualizeResult/AAU-net',
            'Swin-UNet': './visualize/VisualizeResult/SwinUNet',
            'HiFormer-L': './visualize/VisualizeResult/HiFormer-L',
            'H2Former': './visualize/VisualizeResult/H2Former',
            'BEFUNet': './visualize/VisualizeResult/BEFUNet',
            'U-Mamba': './visualize/VisualizeResult/UMamba',
            'Swin-UMamba': './visualize/VisualizeResult/SwinUMamba',
            'Ours': './visualize/VisualizeResult/Ours'
        }
        # --------------------------对比实验结果可视化--------------------------
        input_images, gt_images, pred_images = load_images(input_dir, gt_dir, pred_dirs)          # 加载图像路径
        plot_segmentation_results(input_images, gt_images, pred_images, methods, dataset_name, save_path=None)
    elif type == "Transfer":
        input_dir = './visualize/Transfer-Visualize/images'
        gt_dir = './visualize/Transfer-Visualize/masks'
        pred_dirs = {
            'UNet': './visualize/Transfer-Visualize/UNet',
            'UNet++': './visualize/Transfer-Visualize/UNet++',
            'AAU-net': './visualize/Transfer-Visualize/AAU-net',
            'Swin-UNet': './visualize/Transfer-Visualize/SwinUNet',
            'HiFormer-L': './visualize/Transfer-Visualize/HiFormer-L',
            'H2Former': './visualize/Transfer-Visualize/H2Former',
            'BEFUNet': './visualize/Transfer-Visualize/BEFUNet',
            'U-Mamba': './visualize/Transfer-Visualize/UMamba',
            'Swin-UMamba': './visualize/Transfer-Visualize/SwinUMamba',
            'Ours': './visualize/Transfer-Visualize/Ours'
        }
        # --------------------------域转移实验结果可视化--------------------------
        input_images, gt_images, pred_images = load_images(input_dir, gt_dir, pred_dirs)          # 加载图像路径
        plot_segmentation_results(input_images, gt_images, pred_images, methods,dataset_name, save_path=None)
    elif type == "Ablation":
        # 消融实验图像路径
        ablation_input_dir = './visualize/Ablation/images'
        ablation_gt_dir = './visualize/Ablation/masks'
        ablation_pred_dirs = {
            'baseline': './visualize/Ablation/baseline',
            'w/o CNN': './visualize/Ablation/CNN',
            'w/o TiFusion': './visualize/Ablation/TiFusion',
            'w/o MWFFD': './visualize/Ablation/MWFFD',
            'w/o Loss': './visualize/Ablation/Hybrid',
            'BC-Mamba': './visualize/Ablation/Ours'
        }
        #--------------------------消融实验结果可视化-----------------------
        abla_input_images, abla_gt_images, abla_pred_images = load_images(ablation_input_dir, ablation_gt_dir, ablation_pred_dirs)
        ablation_segmentation_results(abla_input_images, abla_gt_images, abla_pred_images, ablation_methods)
