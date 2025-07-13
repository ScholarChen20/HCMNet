import os

from PIL import Image, ImageDraw, ImageFont

# 1. 打开原始图片（替换为你的图片路径）
img = Image.open("benign (330).png")
width, height = img.size

# 2. 设定网格参数（4行4列，对应16块）
rows = 4
cols = 4
cell_width = width // cols  # 每个子图的宽度
cell_height = height // rows  # 每个子图的高度

# 3. 选择字体（需本地有字体文件，如系统自带的Arial、思源黑体等，替换路径）
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)  # 字号可调整，字体路径需正确

# 4. 循环分割并添加编号
for i in range(rows):  # 遍历行（0~3）
    for j in range(cols):  # 遍历列（0~3）
        # 计算当前子图的裁剪范围：左、上、右、下
        left = j * cell_width
        top = i * cell_height
        right = (j + 1) * cell_width
        bottom = (i + 1) * cell_height

        # 裁剪子图
        sub_img = img.crop((left, top, right, bottom))

        # 在子图上绘制编号（如1~16）
        draw = ImageDraw.Draw(sub_img)
        number = i * cols + j + 1  # 计算当前编号（1~16）
        # 计算文字居中位置（也可改到角落，如(10,10)）
        # text_w, text_h = draw.textsize(str(number), font=font)
        text = str(number)
        # 1. 用 getlength() 获取文本宽度
        text_w = draw.textlength(text, font=font)

        # 2. 用 font.getmetrics() 获取字体的“上升高度”和“下降高度”，总和为文本高度
        ascent, descent = font.getmetrics()
        text_h = ascent + descent

        text_x = (cell_width - text_w) // 2
        text_y = (cell_height - text_h) // 2
        draw.text((text_x, text_y), str(number), fill="white", font=font)  # 文字颜色可改
        os.makedirs("output", exist_ok=True)  # 创建输出目录（如不存在则创建）

        # 保存子图（命名为 output_1.jpg ~ output_16.jpg）
        sub_img.save(os.path.join("output", f"output_{number}.png"))