import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 设置中文字体（可选）
plt.rcParams['axes.unicode_minus'] = False

# 46.50 49.67   ----  48.96 39.47
# 示例数据（与目标图一致）
data = {
    'Model': ['H2Former', 'HiFormer-L', 'TransUNet', 'SwinUNet','BEFUNet',
              'UNet', 'UNet++', 'AAU-net', 'Attention U-Net',
              'UMamba', 'VM-UNet-V2', 'SwinUMamba', 'MedMamba (Ours)'],
    'Params': [33.68, 31.50, 93.23, 41.38, 42.61,
               31.04, 36.63, 53.22, 34.88,
               76.38, 22.77, 59.88, 46.50],
    # 'FPS': [175, 150, 160, 155, 375, 75, 180, 325, 170, 190, 300, 180, 150, 175],
    # 'FLOPs': [8.62, 23.72, 12.23, 105.28, 8.7, 8.5, 36.99, 105.87, 11.76, 51.07, 147.94, 4.48, 43.91, 43.73],   # 1
    'FLOPs': [98.65, 64.22, 98.68, 34.75, 31.82,
              167.63, 212.34, 179.66, 204.06,
              295.87, 17.59, 175.73, 22.62],
    'Dice': [81.54, 82.06, 78.37, 75.72, 81.36,
             77.31, 77.90, 76.59, 77.68,
             80.32, 82.48, 83.53, 85.65]
}

df = pd.DataFrame(data)

# 颜色映射（参考原图颜色）

colors = plt.cm.Set3(np.linspace(0, 1, len(df)))
color_map = {model: colors[i] for i, model in enumerate(df['Model'])}
# 点大小（统一为固定大小）
point_size = 90

def plot_complexity():

    fig, ax = plt.subplots(figsize=(10, 7))
    # 定义参考参数量（百万级）
    ref_params = [25, 50, 75]
    ref_x = [235, 255, 280]  # 右上角位置
    ref_y = [84, 84, 84]

    # 绘制参考圆圈（灰色）
    for i, (p, x, y) in enumerate(zip(ref_params, ref_x, ref_y)):
        size = p * 40  # 缩放因子
        ax.scatter(x, y, s=size, color='gray', alpha=0.4, edgecolors='black', linewidth=0.8)
        ax.annotate(f'{p}M', xy=(x, y), xytext=(x , y - 0.5),
                   fontsize=12, ha='left', va='top', color='black')
        ax.scatter(x, y, color='gray', s=8, linewidth=2, marker='o', zorder=5)

    # 绘制每个点并标注模型名称
    for idx, row in df.iterrows():
        model_name = row['Model']
        x, y = row['FLOPs'], row['Dice']
        params = row['Params']  # 参数量（单位：百万）
        color = color_map[model_name]
        # 点大小与参数量成正比
        size = params * 40  # 缩放因子
        # 绘制散点
        # 绘制散点：内部浅色 + 边界深色
        ax.scatter(x, y, color=color, s=size, alpha=0.75, edgecolors=color, linewidth=1.5)

        # 在中心添加加粗点（颜色与边界一致）
        ax.scatter(x, y, color=color, s=8, linewidth=2, marker='o', zorder=5)
        # 添加文本标注（无箭头）
        # if model_name == 'BCMamba (Ours)':
        #     ax.annotate(model_name, xy=(x, y), xytext=(x + 10, y + 0.36),
        #                fontsize=10, fontweight='bold', ha='left', va='bottom')
        if model_name == 'SwinUMamba':
            ax.annotate(model_name, xy=(x, y), xytext=(x + 6, y + 0.65),
                       fontsize=10, ha='left', va='bottom')
        elif model_name == 'UNet':
            ax.annotate(model_name, xy=(x, y), xytext=(x - 0.5, y - 1.1),
                       fontsize=10, ha='left', va='bottom')
        elif model_name == 'H2Former':
            ax.annotate(model_name, xy=(x, y), xytext=(x + 5, y + 0.55),
                       fontsize=10, ha='left', va='bottom')
        elif model_name == 'HiFormer-L':
            ax.annotate(model_name, xy=(x, y), xytext=(x + 5, y + 0.54),
                       fontsize=10, ha='left', va='bottom')
        elif model_name == 'SwinUNet':
            ax.annotate(model_name, xy=(x, y), xytext=(x, y - 0.80),
                       fontsize=10, ha='left', va='bottom')
        elif model_name == 'TransUNet':
            ax.annotate(model_name, xy=(x, y), xytext=(x + 8, y - 1),
                       fontsize=10, ha='left', va='bottom')
        elif model_name == 'AAUNet':
            ax.annotate(model_name, xy=(x, y), xytext=(x + 5, y + 0.8),
                       fontsize=10, ha='left', va='bottom')
        elif model_name == 'Attention U-Net':
            ax.annotate(model_name, xy=(x, y), xytext=(x + 6, y - 0.65),
                       fontsize=10, ha='left', va='bottom')
        elif model_name == 'ResUNet':
            ax.annotate(model_name, xy=(x, y), xytext=(x + 6, y + 0.55),
                       fontsize=10, ha='left', va='bottom')
        elif model_name == 'UMamba':
            ax.annotate(model_name, xy=(x, y), xytext=(x - 8, y + 0.45),
                       fontsize=10, ha='right', va='bottom')
        else:
            ax.annotate(model_name, xy=(x, y), xytext=(x + 5, y + 0.55),
                       fontsize=10, ha='left', va='bottom')

    # 设置坐标轴标签
    ax.set_xlabel('FLOPs', fontsize=12, fontweight='medium')
    ax.set_ylabel('Average Dice Score (%)', fontsize=12, fontweight='medium')

    # 设置标题
    ax.set_title('FLOPs vs. Average Dice Score across 4 datasets',
                 fontsize=14, fontweight='bold', pad=20)

    # 设置坐标轴范围
    ax.set_xlim(15, 310)
    ax.set_ylim(73, 88)

    # 刻度设置
    ax.set_xticks(range(10, 310, 50))
    ax.set_yticks(range(76, 86, 2))
    ax.tick_params(axis='both', labelsize=10)

    # 添加网格（虚线）
    ax.grid(True, linestyle='--', alpha=0.7)

    # 调整布局
    plt.tight_layout()

    # 保存图像
    plt.savefig('./visualize/fps_vs_dice/fps_vs_dice_clean.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    plot_complexity()
