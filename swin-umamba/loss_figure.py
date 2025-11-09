import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# 示例数据（模拟收敛趋势）
np.random.seed(42)
import numpy as np


def generate_dice(epochs, model_name, dataset_name, total_epochs):
    """
    从Excel文件中读取真实的Dice序列，若文件不存在则使用模拟数据。
    每5个epoch取一次平均值，并用线性插值填充到完整epoch数。
    参数:
        epochs: int, 要获取的epoch数量
        model_name: str, 模型名称
        dataset_name: str, 数据集名称
        total_epochs: int, 训练总周期数

    返回:
        np.ndarray: 长度为 epochs 的Dice序列（已平滑并插值）
    """
    # 构造文件路径（处理空格和括号）
    safe_model_name = model_name.replace(' ', '_').replace('(', '').replace(')', '')
    file_path = os.path.join('./log_dir', dataset_name, f"{safe_model_name}_{total_epochs}.xlsx")

    # 尝试读取Excel文件
    try:
        df = pd.read_excel(file_path)
        if 'Epoch' in df.columns and 'Dice' in df.columns:
            # 确保按Epoch排序
            df = df.sort_values('Epoch').reset_index(drop=True)
            # 截取前epochs个数据点
            dice_data = df['Dice'].values[:epochs]

            # 每5个epoch取平均值
            step = 5
            smoothed_dice = []
            for i in range(0, len(dice_data), step):
                window = dice_data[i:i+step]
                smoothed_dice.append(np.mean(window))

            # 插值到完整epoch数
            x_smooth = np.arange(1, len(smoothed_dice) + 1)  # [1, 2, ..., 30]
            x_full = np.arange(1, epochs + 1)                 # [1, 2, ..., 150]
            interpolated_dice = np.interp(x_full, x_smooth, smoothed_dice)

            return interpolated_dice
        else:
            print(f"Warning: 文件 {file_path} 缺少 'Epoch' 或 'Dice' 列，使用模拟数据。")
    except FileNotFoundError:
        print(f"Warning: 文件 {file_path} 未找到，使用模拟数据。")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        # 使用模拟数据作为回退
        pass

    # 回退到模拟数据（并做相同处理）
    x = np.linspace(0, 1, epochs)
    dice = 0.4 + (0.78 - 0.4) * (1 - np.exp(-5 * x))  # 默认收敛曲线
    dice += np.random.normal(0, 0.01, epochs)
    dice = np.clip(dice, 0, 1)

    # 每5个epoch取平均值
    step = 5
    smoothed_dice = []
    for i in range(0, len(dice), step):
        window = dice[i:i+step]
        smoothed_dice.append(np.mean(window))

    # 插值到完整epoch数
    x_smooth = np.arange(1, len(smoothed_dice) + 1)
    x_full = np.arange(1, epochs + 1)
    interpolated_dice = np.interp(x_full, x_smooth, smoothed_dice)

    return interpolated_dice





def plot_loss_figure(models, dataset_name, epochs):

    # 生成数据
    data = {'Epoch': np.arange(1, epochs + 1)}
    for model in models:
        data[model] = generate_dice(epochs, model, dataset_name, epochs)

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
    ax.set_ylabel('Dice', fontsize=12, fontweight='medium')

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
    plt.savefig(f'./visualize/fps_vs_dice/{dataset_name}_epochs_vs_dice.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    # 模型列表
    models = ['BCMamba (Ours)', 'H2Former', 'HiFormer-L', 'TransUNet', 'SwinUNet',
              'BEFUNet', 'UNet', 'UNet++', 'AAUNet', 'Attention U-Net',
              'UMamba', 'VM-UNet-V2', 'SwinUMamba', 'ResUNet']
    plot_loss_figure(models, "BUSI", 150)
