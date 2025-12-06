import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
np.random.seed(42)
import numpy as np

def generate_dice(epochs, model_name, dataset_name, total_epochs, type):
    """
    从Excel文件读取Dice曲线；若文件不存在，则生成模拟数据。
    每5个epoch取平均值，再对这些平均点进行线性插值，得到完整epoch的Dice值。
    """
    file_path = os.path.join('./log_dir', dataset_name, f"{model_name}_{total_epochs}.xlsx")
    try:
        df = pd.read_excel(file_path)
        if 'Epoch' in df.columns and type in df.columns:
            df = df.sort_values('Epoch').reset_index(drop=True)
            if type == 'TrainLoss':
                dice_data = df['TrainLoss'].values[:epochs]
            else:
                dice_data = df['Dice'].values[:epochs]
        else:
            print(f"Warning: 文件 {file_path} 缺少 'Epoch' 或 {type} 列，使用模拟数据。")
            dice_data = None
    except FileNotFoundError:
        print(f"Warning: 文件 {file_path} 未找到，使用模拟数据。")
        dice_data = None
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        dice_data = None

    # === 模拟数据 ===
    if dice_data is None or len(dice_data) < epochs:
        dice_data = np.linspace(0.4, 0.80, epochs)

    # === 每5个epoch取平均 ===
    step = 10
    smoothed_dice = []
    x_smooth = []

    # 第1个epoch使用原始值
    if len(dice_data) > 0:
        smoothed_dice.append(dice_data[0])
        x_smooth.append(1)

    # 后续每5个epoch取平均值，并设置对应epoch为第5、10、15...
    for i in range(step, len(dice_data), step):
        window = dice_data[i - step:i]
        smoothed_dice.append(np.mean(window))
        x_smooth.append(i)

    smoothed_dice = np.array(smoothed_dice)
    x_smooth = np.array(x_smooth)

    # === 插值到完整的 epoch 1~epochs ===
    x_full = np.arange(1, epochs + 1)
    interpolated_dice = np.interp(x_full, x_smooth, smoothed_dice)

    return interpolated_dice


def plot_dice_figure(models, dataset_name, epochs, type):
    # 生成数据
    data = {'Epoch': np.arange(1, epochs + 1)}
    for model in models:
        data[model] = generate_dice(epochs, model, dataset_name, epochs, type)
    df = pd.DataFrame(data)
    # 动态生成颜色（使用 matplotlib 的 tab10 调色板）
    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
    colors[2] = (1, 0.2, 0.2, 1)      #medmamba 红色
    colors[13] = (0.2, 0.8, 0.2, 1)      #medmamba 深绿色
    color_map = {model: colors[i] for i, model in enumerate(models)}

    fig, ax = plt.subplots(figsize=(10, 6))
    for model in models:
        ax.plot(
            df['Epoch'],
            df[model],
            color=color_map[model],
            linewidth=2.0,    # 线条宽度
            alpha = 0.9,
            label=model
        )

    # 添加坐标轴标签
    ax.set_xlabel('Epoch', fontsize=12, fontweight='medium')
    ax.set_ylabel(type, fontsize=12, fontweight='medium')
    # 添加标题
    ax.set_title(f'Benign Tumour of {dataset_name}', fontsize=14, fontweight='bold', pad=20)
    # 设置坐标轴范围
    ax.set_xlim(0, epochs)
    ax.set_ylim(0.1, 0.85)
    # 设置刻度
    ax.set_xticks(np.arange(0, epochs+1, 10))
    ax.set_yticks(np.arange(0.1, 0.81, 0.1))
    ax.tick_params(axis='both', labelsize=10)
    # 添加网格线（虚线）
    ax.grid(True, linestyle='--', alpha=0.7)
    # 添加图例（右下角，避免遮挡）
    ax.legend(loc='lower right', fontsize=10, frameon=True, fancybox=True, shadow=False, edgecolor='black')
    # 调整布局
    plt.tight_layout()
    # 保存图像
    plt.savefig(f'./visualize/fps_vs_dice/{dataset_name}_epochs_vs_{type}.png', dpi=600, bbox_inches='tight')
    plt.show()

def plot_loss_figure(models, dataset_name, epochs, type='TrainLoss'):
    # 生成数据
    data = {'Epoch': np.arange(1, epochs + 1)}
    for model in models:
        data[model] = generate_dice(epochs, model, dataset_name, epochs, type)
    df = pd.DataFrame(data)
    # 动态生成颜色（使用 matplotlib 的 tab10 调色板）
    colors =plt.cm.Set3(np.linspace(0, 1, len(models)))
    color_map = {model: colors[i] for i, model in enumerate(models)}

    fig, ax = plt.subplots(figsize=(10, 6))
    for model in models:
        ax.plot(
            df['Epoch'],
            df[model],
            color=color_map[model],
            linewidth=1.5,    # 线条宽度
            label=model
        )

    # 添加坐标轴标签
    ax.set_xlabel('Epoch', fontsize=12, fontweight='medium')
    ax.set_ylabel('Train Loss', fontsize=12, fontweight='medium')
    # 添加标题
    ax.set_title(f'Benign Tumour of {dataset_name}', fontsize=14, fontweight='bold', pad=20)
    # 设置坐标轴范围
    ax.set_xlim(0, epochs)
    ax.set_ylim(0.15, 1.1)
    # 设置刻度
    ax.set_xticks(np.arange(0, epochs+1, 10))
    ax.set_yticks(np.arange(0.15, 1.1, 0.15))
    ax.tick_params(axis='both', labelsize=10)
    # 添加网格线（虚线）
    ax.grid(True, linestyle='--', alpha=0.7)
    # 添加图例（右下角，避免遮挡）
    ax.legend(loc='lower right', fontsize=10, frameon=True, fancybox=True, shadow=False)
    # 调整布局
    plt.tight_layout()
    # 保存图像
    plt.savefig(f'./visualize/fps_vs_dice/{dataset_name}_epochs_vs_loss.png', dpi=600, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    # 模型列表
    models = ['BCMamba', 'H2Former', 'HiFormer-L', 'TransUNet', 'SwinUNet',
              'BEFUNet', 'UNet', 'UNet++', 'AAUNet', 'ATTU-Net',
              'UMamba', 'VMUNetv2', 'SwinUMamba', 'MedMamba']

    plot_dice_figure(models, "BUSI", 150, "Dice")
    # plot_loss_figure(models, "BUSI", 150, "TrainLoss")


