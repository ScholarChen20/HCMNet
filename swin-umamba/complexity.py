import matplotlib.pyplot as plt
import pandas as pd

# 设置中文字体（可选）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 示例数据（与目标图一致）
data = {
    'Model': ['BCMamba (Ours)', 'H2Former', 'HiFormer-L', 'TransUNet', 'SwinUNet',
              'BEFUNet', 'UNet', 'UNet++', 'AAUNet', 'Attention U-Net',
              'UMamba', 'VM-UNet-V2', 'SwinUMamba', 'ResUNet'],
    'FPS': [175, 150, 160, 155, 375, 75, 180, 325, 170, 190, 300, 180, 150, 175],
    'Dice': [90.0, 88.5, 88.8, 88.2, 88.5, 86.0, 86.0, 86.5, 84.0, 85.0, 85.5, 83.0, 82.0, 80.0]
}

df = pd.DataFrame(data)

# 颜色映射（参考原图颜色）
color_map = {
    'BCMamba (Ours)': '#FFD700',     # 黄色
    'H2Former': '#8BC34A',           # 浅绿
    'HiFormer-L': '#4CAF50',      # 绿色
    'TransUNet': '#4CAF50',          # 绿色
    'SwinUNet': '#B3E5FC',             # 蓝灰
    'BEFUNet': '#4DB6AC',      # 青色
    'UNet': '#4DB6AC',         # 青色
    'UNet++': '#7986CB',             # 淡蓝
    'AAUNet': '#9C27B0',            # 紫色
    'Attention U-Net': '#9C27B0',    # 紫色
    'UMamba': '#9C27B0',              # 紫色
    'VM-UNet-V2': '#4DB6AC',     # 青色
    'SwinUMamba': '#7986CB',           # 淡蓝
    'ResUNet': '#7986CB'            # 淡蓝
}

# 点大小（统一为固定大小）
point_size = 90

def plot_complexity():
    fig, ax = plt.subplots(figsize=(10, 7))

    # 绘制每个点并标注模型名称
    for idx, row in df.iterrows():
        model_name = row['Model']
        x, y = row['FPS'], row['Dice']

        # 绘制散点
        ax.scatter(x, y, color=color_map[model_name], s=point_size, alpha=0.8)

        # 添加文本标注（无箭头）
        if model_name == 'BCMamba (Ours)':
            ax.annotate(model_name, xy=(x, y), xytext=(x + 5, y + 0.2),
                       fontsize=10, fontweight='bold', ha='left', va='bottom')
        elif model_name == 'M2Snet':
            ax.annotate(model_name, xy=(x, y), xytext=(x - 5, y + 0.2),
                       fontsize=10, ha='right', va='bottom')
        elif model_name == 'I2U-Net Large':
            ax.annotate(model_name, xy=(x, y), xytext=(x + 5, y + 0.2),
                       fontsize=10, ha='left', va='bottom')
        elif model_name == 'H2Former':
            ax.annotate(model_name, xy=(x, y), xytext=(x - 5, y + 0.2),
                       fontsize=10, ha='right', va='bottom')
        elif model_name == 'TransUNet':
            ax.annotate(model_name, xy=(x, y), xytext=(x + 5, y - 0.2),
                       fontsize=10, ha='left', va='bottom')
        else:
            ax.annotate(model_name, xy=(x, y), xytext=(x + 5, y + 0.2),
                       fontsize=10, ha='left', va='bottom')

    # 设置坐标轴标签
    ax.set_xlabel('Frame Per Second (FPS)', fontsize=12, fontweight='medium')
    ax.set_ylabel('Average Dice Score (%)', fontsize=12, fontweight='medium')

    # 设置标题
    ax.set_title('Frame Per Second (FPS) vs. Average Dice Score across 4 datasets',
                 fontsize=14, fontweight='bold', pad=20)

    # 设置坐标轴范围
    ax.set_xlim(40, 410)
    ax.set_ylim(78, 92)

    # 刻度设置
    ax.set_xticks(range(50, 401, 50))
    ax.set_yticks(range(80, 92, 2))
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
