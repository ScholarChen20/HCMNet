import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 示例数据（模拟收敛趋势）
np.random.seed(42)

def generate_dice(epochs, start, end, noise=0.01):
    """生成模拟的Dice序列：从start上升到end，添加轻微噪声"""
    x = np.linspace(0, 1, epochs)
    dice = start + (end - start) * (1 - np.exp(-5 * x))  # sigmoid函数模拟收敛
    dice += np.random.normal(0, noise, epochs)
    dice = np.clip(dice, 0, 1)
    return dice

# 模型列表
models = [
    'U-Net', 'U-Net++', 'Attention U-Net', 'TransUNet',
    'CENet', 'UNext', 'AAU-Net', 'MEF-UNet'
]

# 设置训练轮数
epochs = 300

# 生成数据
data = {'Epoch': np.arange(1, epochs+1)}
for model in models:
    if model == 'U-Net':
        data[model] = generate_dice(epochs, 0.4, 0.72)
    elif model == 'U-Net++':
        data[model] = generate_dice(epochs, 0.42, 0.75)
    elif model == 'Attention U-Net':
        data[model] = generate_dice(epochs, 0.15, 0.70)
    elif model == 'TransUNet':
        data[model] = generate_dice(epochs, 0.45, 0.73)
    elif model == 'CENet':
        data[model] = generate_dice(epochs, 0.43, 0.71)
    elif model == 'UNext':
        data[model] = generate_dice(epochs, 0.35, 0.74)
    elif model == 'AAU-Net':
        data[model] = generate_dice(epochs, 0.41, 0.76)
    else:  # MEF-UNet
        data[model] = generate_dice(epochs, 0.48, 0.78)

df = pd.DataFrame(data)

# 颜色映射（参考示例图颜色）
color_map = {
    'U-Net': '#212121',
    'U-Net++': '#FFEB3B',
    'Attention U-Net': '#BDBDBD',
    'TransUNet': '#FF9800',
    'CENet': '#2196F3',
    'UNext': '#4CAF50',
    'AAU-Net': '#9C27B0',
    'MEF-UNet': '#F44336'
}

# 线条宽度
linewidth = 1.5

def plot_loss_figure():
    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制每条折线
    for model in models:
        ax.plot(
            df['Epoch'],
            df[model],
            color=color_map[model],
            linewidth=linewidth,
            label=model
        )

    # 添加坐标轴标签
    ax.set_xlabel('Epoch', fontsize=12, fontweight='medium')
    ax.set_ylabel('Dice', fontsize=12, fontweight='medium')

    # 添加标题
    ax.set_title('Benign Tumour of BUSI', fontsize=14, fontweight='bold', pad=20)

    # 设置坐标轴范围
    ax.set_xlim(0, epochs)
    ax.set_ylim(0, 0.8)

    # 设置刻度
    ax.set_xticks(np.arange(0, epochs+1, 50))
    ax.set_yticks(np.arange(0, 0.81, 0.1))
    ax.tick_params(axis='both', labelsize=10)

    # 添加网格线（虚线）
    ax.grid(True, linestyle='--', alpha=0.7)

    # 添加图例（右上角，避免遮挡）
    ax.legend(loc='upper right', fontsize=10, frameon=True, fancybox=True, shadow=False)

    # 调整布局
    plt.tight_layout()

    # 保存图像
    plt.savefig('./visualize/fps_vs_dice/epochs_vs_dice.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    plot_loss_figure()
