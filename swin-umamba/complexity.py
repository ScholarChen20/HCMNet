import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 设置中文字体（可选）
plt.rcParams['axes.unicode_minus'] = False

# 46.50 49.67   ----  48.96 39.47
# UMamba 76.38M-295.87G
# 示例数据（与目标图一致）
data = {
    'Model': ['H2Former', 'HiFormer-L', 'TransUNet', 'SwinUNet','BEFUNet',
              'UNet', 'UNet++', 'AAU-net', 'Attention U-Net',
              'MLAgg-UNet', 'VM-UNet-V2', 'SwinUMamba', 'LoMamba (Ours)'],
    'Params': [33.68, 31.50, 105.28, 41.38, 42.61,
               31.04, 36.63, 53.22, 34.88,
               3.21, 22.77, 59.88, 34.92],
    # 'FPS': [175, 150, 160, 155, 375, 75, 180, 325, 170, 190, 300, 180, 150, 175],
    # 'FLOPs': [8.62, 23.72, 12.23, 105.28, 8.7, 8.5, 36.99, 105.87, 11.76, 51.07, 147.94, 4.48, 43.91, 43.73],   # 1
    'FLOPs': [23.72, 12.23, 25.35, 8.70, 8.50,
              48.32, 138.28, 57.12, 66.7,
              0.962, 4.25, 30.49, 8.44],
    'Dice': [81.12, 82.21, 77.43, 74.24, 79.04,
             77.31, 77.23, 74.52, 76.31,
             76.03, 81.61, 81.05, 84.37]
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
            ax.annotate(model_name, xy=(x, y), xytext=(x + 5, y + 0.55),
                       fontsize=10, ha='left', va='bottom')
        elif model_name == 'H2Former':
            ax.annotate(model_name, xy=(x, y), xytext=(x + 5, y + 0.55),
                       fontsize=10, ha='left', va='bottom')
        elif model_name == 'HiFormer-L':
            ax.annotate(model_name, xy=(x, y), xytext=(x + 5, y + 0.54),
                       fontsize=10, ha='left', va='bottom')
        elif model_name == 'SwinUNet':
            ax.annotate(model_name, xy=(x, y), xytext=(x, y + 0.85),
                       fontsize=10, ha='left', va='bottom')
        elif model_name == 'TransUNet':
            ax.annotate(model_name, xy=(x, y), xytext=(x + 8, y + 1.1),
                       fontsize=10, ha='left', va='bottom')
        elif model_name == 'AAU-net':
            ax.annotate(model_name, xy=(x, y), xytext=(x + 3, y - 1.2),
                       fontsize=10, ha='left', va='bottom')
        elif model_name == 'Attention U-Net':
            ax.annotate(model_name, xy=(x, y), xytext=(x + 3, y - 1.2),
                       fontsize=10, ha='left', va='bottom')
        elif model_name == 'ResUNet':
            ax.annotate(model_name, xy=(x, y), xytext=(x + 6, y + 0.55),
                       fontsize=10, ha='left', va='bottom')
        elif model_name == 'UMamba':
            ax.annotate(model_name, xy=(x, y), xytext=(x - 8, y + 0.55),
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
    ax.set_yticks(range(76, 88, 2))
    ax.tick_params(axis='both', labelsize=10)

    # 添加网格（虚线）
    ax.grid(True, linestyle='--', alpha=0.7)

    # 调整布局
    plt.tight_layout()

    # 保存图像
    plt.savefig('./visualize/fps_vs_dice/fps_vs_dice_clean.png', dpi=1000, bbox_inches='tight')
    plt.show()

# 定义标记样式（右图用不同形状）
    marker_map = {
        'H2Former': '*',
        'HiFormer-L': '^',
        'TransUNet': 's',
        'SwinUNet': 'D',
        'BEFUNet': 'p',
        'UNet': 'o',
        'UNet++': 'v',
        'AAU-net': '<',
        'Attention U-Net': '>',
        'MLAgg-UNet': 'h',
        'VM-UNet-V2': '+',
        'SwinUMamba': 'x',
        'LoMamba (Ours)': 'X'
    }

def plot_dice_vs_params_and_flops():
    """
    同时绘制 Dice vs Params 和 Dice vs FLOPs 两个散点图。
    - 左图：统一形状，不同颜色
    - 右图：统一颜色，不同形状
    - 上方显示颜色图例
    - 子图间距紧凑
    """
    # 使用原始数据
    df = pd.DataFrame({
        'Model': ['H2Former', 'HiFormer-L', 'TransUNet', 'SwinUNet','BEFUNet',
                  'UNet', 'UNet++', 'AAU-net', 'Attention U-Net',
                  'MLAgg-UNet', 'VM-UNet-V2', 'SwinUMamba', 'LoMamba (Ours)'],
        'Params': [34.68, 31.50, 105.28, 41.38, 42.61,
                   31.04, 36.63, 53.22, 34.88,
                   3.21, 22.77, 59.88, 34.92],
        'FLOPs': [23.72, 14.53, 25.35, 8.70, 8.50,
                  48.32, 138.00, 57.12, 66.7,
                  1.96, 4.25, 30.49, 8.44],
        'Dice': [81.10, 82.41, 77.43, 74.24, 79.04,
                 77.31, 77.23, 74.52, 76.31,
                 76.03, 81.61, 80.85, 84.37]
    })

    # 定义颜色映射（参考你提供的图）
    color_map = {
        'H2Former': 'red',
        'HiFormer-L': 'blue',
        'TransUNet': 'cyan',
        'SwinUNet': 'green',
        'BEFUNet': 'magenta',
        'UNet': 'orange',
        'UNet++': 'yellow',
        'AAU-net': 'brown',
        'Attention U-Net': 'purple',
        'MLAgg-UNet': 'gray',
        'VM-UNet-V2': 'darkgreen',
        'SwinUMamba': 'pink',
        'LoMamba (Ours)': 'lightblue'
    }



    # 创建图形和子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Performance vs Complexity Analysis', fontsize=14, fontweight='bold')

    # ==================== 左图：Dice vs Params（统一形状，不同颜色）====================
    for idx, row in df.iterrows():
        model_name = row['Model']
        x, y = row['Params'], row['Dice']
        color = color_map[model_name]
        size = 110
        marker = 'o'  # 统一使用圆形

        ax1.scatter(x, y, color=color, s=size, marker=marker, edgecolors='black', linewidth=0.8)
        if model_name == 'VM-UNet-V2':
            ax1.annotate(model_name, xy=(x, y), xytext=(x, y + 0.2),
                         fontsize=10, ha='right', va='bottom', color='black')
        elif model_name == 'UNet' or model_name == 'SwinUNet' or model_name == 'TransUNet':
            ax1.annotate(model_name, xy=(x, y), xytext=(x, y + 0.2),
                         fontsize=10, ha='right', va='bottom', color='black')
        elif model_name == 'LoMamba (Ours)':  # 加粗
            ax1.annotate(model_name, xy=(x, y), xytext=(x + 1, y + 0.2), fontsize=10, ha='left', va='bottom', color='black', weight='bold')
        else:
            ax1.annotate(model_name, xy=(x, y), xytext=(x + 1, y + 0.2),
                         fontsize=10, ha='left', va='bottom', color='black')

    ax1.set_xlabel('Params (M)', fontsize=11)
    ax1.set_ylabel('Average Dice Score (%)', fontsize=11)
    ax1.set_xlim(0, 110)
    ax1.set_ylim(73, 86)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # ==================== 右图：Dice vs FLOPs（统一形状，不同颜色）====================
    for idx, row in df.iterrows():
        model_name = row['Model']
        x, y = row['FLOPs'], row['Dice']
        color = color_map[model_name]
        size = 115
        marker = '*'

        ax2.scatter(x, y, color=color, s=size, marker=marker, edgecolors='black', linewidth=0.5)
        if model_name == 'SwinUMamba':
            ax2.annotate(model_name, xy=(x, y), xytext=(x, y - 0.7),
                         fontsize=10, ha='left', va='bottom', color='black')
        elif model_name == 'UNet++' :
            ax2.annotate(model_name, xy=(x, y), xytext=(x + 1, y + 0.2),
                         fontsize=10, ha='right', va='bottom', color='black')
        elif model_name == 'VM-UNet-V2':
            ax2.annotate(model_name, xy=(x, y), xytext=(x + 0.25, y + 0.1),
                         fontsize=10, ha='left', va='bottom', color='black')
        elif model_name == 'LoMamba (Ours)':  # 加粗
            ax2.annotate(model_name, xy=(x, y), xytext=(x + 1, y + 0.2),fontsize=10, ha='left', va='bottom', color='black', weight='bold')
        else:
            ax2.annotate(model_name, xy=(x, y), xytext=(x + 1, y + 0.2),
                         fontsize=10, ha='left', va='bottom', color='black')

    ax2.set_xlabel('GFLOPs', fontsize=11)
    ax2.set_ylabel('Average Dice Score (%)', fontsize=11)
    ax2.set_xlim(0, 140)
    ax2.set_ylim(73, 86)
    ax2.grid(True, linestyle='--', alpha=0.6)

    # # ==================== 添加顶部图例（颜色-方法对应）====================
    # legend_elements = []
    # for model, color in color_map.items():
    #     legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8, label=model))
    #
    # # 将图例放在两个子图上方，居中
    # ax1.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=7, fontsize=9)

    # 调整布局，减小子图间距
    # plt.tight_layout(pad=3.0)
    plt.savefig('./visualize/fps_vs_dice/dice_vs_params_and_flops.png', dpi=600, bbox_inches='tight')
    plt.show()



if __name__ == '__main__':
    # plot_complexity()
    plot_dice_vs_params_and_flops()

