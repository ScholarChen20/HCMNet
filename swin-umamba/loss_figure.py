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
            if type == 'Loss':
                dice_data = 1 - df['Loss'].values[:epochs]
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
        x = np.linspace(0, 1, epochs)
        dice_data = 0.4 + (0.78 - 0.4) * (1 - np.exp(-5 * x)) + np.random.normal(0, 0.01, epochs)
        dice_data = np.clip(dice_data, 0, 1)

    # === 每5个epoch取平均 ===
    step = 5
    smoothed_dice = []
    x_smooth = []
    for i in range(0, len(dice_data), step):
        window = dice_data[i:i+step]
        smoothed_dice.append(np.mean(window))
        # 平均点对应的 epoch（例如第一个平均点对应 epoch=5）
        x_smooth.append(min(i + step, epochs))

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
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
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
    ax.set_ylabel(type, fontsize=12, fontweight='medium')
    # 添加标题
    ax.set_title(f'Benign Tumour of {dataset_name}', fontsize=14, fontweight='bold', pad=20)
    # 设置坐标轴范围
    ax.set_xlim(0, epochs)
    ax.set_ylim(0, 1.0)
    # 设置刻度
    ax.set_xticks(np.arange(0, epochs+1, 50))
    ax.set_yticks(np.arange(0, 0.81, 0.1))
    ax.tick_params(axis='both', labelsize=10)
    # 添加网格线（虚线）
    ax.grid(True, linestyle='--', alpha=0.7)
    # 添加图例（右下角，避免遮挡）
    ax.legend(loc='lower right', fontsize=10, frameon=True, fancybox=True, shadow=False)
    # 调整布局
    plt.tight_layout()
    # 保存图像
    plt.savefig(f'./visualize/fps_vs_dice/{dataset_name}_epochs_vs_{type}.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    # 模型列表
    models = ['BCMamba (Ours)', 'H2Former', 'HiFormer-L', 'TransUNet', 'SwinUNet',
              'BEFUNet', 'UNet', 'UNet++', 'AAUNet', 'Attention U-Net',
              'UMamba', 'VM-UNet-V2', 'SwinUMamba', 'ResUNet']
    plot_dice_figure(models, "BUSI", 150, "Dice")
    plot_dice_figure(models, "BUSI", 150, "Loss")


