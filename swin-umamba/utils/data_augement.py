# coding=utf-8
import argparse

import matplotlib as mpl
import cv2
import random
import os
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates

mpl.use('TkAgg')

img_w = 384
img_h = 384



image_sets = []


def read_directory(directory_name):
    for filename in os.listdir(directory_name):
        # image
        image_sets.append(filename)
    print(len(image_sets))

# alpha sigma 10,2  -> 5,3
def elastic_transform(image, label, alpha=10, sigma=2, alpha_affine=2, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]
    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    # pts1: 仿射变换前的点(3个点)
    pts1 = np.float32([center_square + square_size,
                       [center_square[0] + square_size,
                        center_square[1] - square_size],
                       center_square - square_size])
    # pts2: 仿射变换后的点
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine,
                                       size=pts1.shape).astype(np.float32)
    # 仿射变换矩阵
    M = cv2.getAffineTransform(pts1, pts2)
    # 对image进行仿射变换.
    imageB = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
    imageA = cv2.warpAffine(label, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    # generate random displacement fields
    # random_state.rand(*shape)会产生一个和shape一样打的服从[0,1]均匀分布的矩阵
    # *2-1是为了将分布平移到[-1, 1]的区间, alpha是控制变形强度的变形因子
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    # generate meshgrid
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    # x+dx,y+dy
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    # bilinear interpolation
    xb = map_coordinates(imageB, indices, order=1, mode='constant').reshape(shape)
    yb = map_coordinates(imageA, indices, order=1, mode='constant').reshape(shape)
    return xb, yb

#伽马校正，调整对比度
def gamma_transform(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)


def random_gamma_transform(img, gamma_vari):
    log_gamma_vari = np.log(gamma_vari)
    alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
    gamma = np.exp(alpha)
    return gamma_transform(img, gamma)


def rotate(xb, yb, angle):
    angle = random.uniform(-angle, angle)
    M_rotate = cv2.getRotationMatrix2D((img_w / 2, img_h / 2), angle, 1)
    xb = cv2.warpAffine(xb, M_rotate, (img_w, img_h))
    yb = cv2.warpAffine(yb, M_rotate, (img_w, img_h))
    return xb, yb


def blur(img):
    img = cv2.blur(img, (3, 3))
    return img

#加高斯噪声5，10 -> 1,3
def add_noise(xb):
    sigma = random.uniform(5, 10)
    h, w = xb.shape
    gauss = np.random.normal(0, sigma, (h, w))
    gauss = gauss.reshape(h, w)
    noisy = xb + gauss
    noisy = np.clip(noisy, 0, 255).astype('uint8')
    return noisy


def data_augment(xb, yb):
    if np.random.random() < 0.20:
        xb, yb = rotate(xb, yb, random.uniform(0, 1) * 20)
    # if np.random.random() < 0.20:
    #     xb, yb = rotate(xb, yb, 180)
    if np.random.random() < 0.20:
        xb, yb = rotate(xb, yb, random.uniform(1, 1.05) * 30)
    if np.random.random() < 0.30:
        xb = cv2.flip(xb, 1)  # flipcode > 0：沿y轴翻转
        yb = cv2.flip(yb, 1)

    if np.random.random() < 0.30:
        xb = random_gamma_transform(xb, 1.0)

    if np.random.random() < 0.30:
        xb = blur(xb)

    if np.random.random() < 0.25:
        xb = add_noise(xb)

    if np.random.random() < 0.35:
        xb, yb = elastic_transform(xb, yb)

    return xb, yb

# #
# # def creat_dataset(args, mode='augment'):
# #     global image_sets  # 使用全局变量
# #
# #     if not image_sets:
# #         raise ValueError("image_sets为空! 请先调用read_directory()")
# #     random.shuffle(image_sets)
# #     print('creating dataset...')
# #     image_each = 5  # image_num / len(image_sets)
# #     g_count = 0
# #
# #     for i in tqdm(range(len(image_sets))):
# #         count = 0
# #         print(image_sets[i])
# #         src_img = cv2.imread(args['Imagepath'] + image_sets[i], cv2.IMREAD_GRAYSCALE)  # 3 channels
# #         label_img = cv2.imread(args['Labelpath'] + image_sets[i], cv2.IMREAD_GRAYSCALE)  # single channel
# #
# #         src_img = cv2.resize(src_img, (512, 512), interpolation=cv2.INTER_NEAREST)
# #         label_img = cv2.resize(label_img, (224, 224), interpolation=cv2.INTER_NEAREST)
# #
# #         # print(src_img.shape)
# #         X_height, X_width = src_img.shape
# #         random_height = random.randint(0, X_height - img_h)
# #         random_width = random.randint(0, X_width - img_w)
# #         while count < image_each:
# #             src_roi = src_img[random_height: random_height +
# #                                              img_h, random_width: random_width + img_w]
# #             label_roi = label_img[random_height: random_height +
# #                                                  img_h, random_width: random_width + img_w]
# #
# #             image = Image.fromarray(np.uint8(src_roi))
# #             extrema = image.convert("L").getextrema()
# #             if extrema != (0, 0):
# #                 if mode == 'augment':
# #                     src_roi, label_roi = data_augment(src_roi, label_roi)
# #
# #             cv2.imwrite((args['Saveimagepath'] + '/%d.png' % g_count), src_roi)
# #             cv2.imwrite((args['Savelabelpath'] + '/%d.png' % g_count), label_roi)
# #             count += 1
# #             g_count += 1

def creat_dataset(args, mode='augment'):
    global image_sets  # 使用全局变量

    if not image_sets:
        raise ValueError("image_sets为空! 请先调用read_directory()")

    random.shuffle(image_sets)
    print('Creating ultrasound dataset...')
    image_each = 5  # 每张原图生成5个子图
    g_count = 0

    # 创建保存目录
    os.makedirs(args['Saveimagepath'], exist_ok=True)
    os.makedirs(args['Savelabelpath'], exist_ok=True)

    for i in tqdm(range(len(image_sets))):
        filename = image_sets[i]
        count = 0

        # 读取图像（添加路径分隔符）
        img_path = os.path.join(args['Imagepath'], filename)
        label_path = os.path.join(args['Labelpath'], filename)

        # 检查文件存在性
        if not os.path.exists(img_path):
            print(f"警告：图像文件 {img_path} 不存在")
            continue
        if not os.path.exists(label_path):
            print(f"警告：标签文件 {label_path} 不存在")
            continue

        # 统一调整为512x512
        src_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        src_img = cv2.resize(src_img, (512, 512), interpolation=cv2.INTER_LANCZOS4)

        label_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        label_img = cv2.resize(label_img, (512, 512), interpolation=cv2.INTER_NEAREST)

        # 生成随机裁剪位置
        max_height = src_img.shape[0] - img_h
        max_width = src_img.shape[1] - img_w

        while count < image_each:
            if max_height <= 0 or max_width <= 0:
                print(f"图像 {filename} 尺寸不足，无法裁剪")
                break

            random_height = random.randint(0, max_height)
            random_width = random.randint(0, max_width)

            # 提取ROI区域
            src_roi = src_img[random_height:random_height + img_h,
                      random_width:random_width + img_w]
            label_roi = label_img[random_height:random_height + img_h,
                        random_width:random_width + img_w]

            # 有效性检查
            if np.all(src_roi == 0):  # 排除全黑图像
                continue

            # 数据增强
            if mode == 'augment':
                src_roi, label_roi = data_augment(src_roi, label_roi)

            # 保存文件（使用零填充编号）
            cv2.imwrite(os.path.join(args['Saveimagepath'], f"1_{g_count:04d}.png"), src_roi)
            cv2.imwrite(os.path.join(args['Savelabelpath'], f"1_{g_count:04d}.png"), label_roi)

            count += 1
            g_count += 1


def args_parse():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--Namepath",default="./data/BUSI/", required=False, help="Name of your original image")
    ap.add_argument("--Imagepath",default="./data/test/origin/images", required=False, help="path of your original train image")
    ap.add_argument("--Labelpath",default="./data/test/origin/masks",  required=False, help="path of your original train label")
    ap.add_argument("--Saveimagepath",default="./data/test/aug/images",  required=False, help="savepath of your train imag")
    ap.add_argument("--Savelabelpath", default="./data/test/aug/masks", required=False, help="savepath of your train label")
    args = vars(ap.parse_args())
    return args


import os
import shutil


def copy_images(source_folder, target_folder, allowed_extensions=('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
    # 如果目标文件夹不存在则创建
    os.makedirs(target_folder, exist_ok=True)

    # 遍历源文件夹中的所有文件
    for filename in os.listdir(source_folder):
        # 检查文件扩展名是否为图片格式
        if filename.lower().endswith(allowed_extensions):
            # 构建完整路径
            source_path = os.path.join(source_folder, filename)
            target_path = os.path.join(target_folder, filename)

            # 执行复制操作
            shutil.copy2(source_path, target_path)
            print(f"已复制: {filename}")


##
def rename(source_folder,target_folder):
    os.makedirs(target_folder, exist_ok=True)
    for filename in os.listdir(source_folder):
        new_filename  =  "Test"+filename[10:]
        source_path = os.path.join(source_folder, filename)
        target_path = os.path.join(target_folder, new_filename)
        os.rename(source_path, target_path)
        print(f"已修改,{filename}")


if __name__ == '__main__':
    # copy_images(
    #     source_folder="./data/BUSI/test/images",
    #     target_folder="./data/BUSI/BUS_Aug/test/images"
    # )
    rename(
        source_folder="C:\\Users\\chenwq\\Downloads\\STU-Hospital-master\\STU_Test\\images",
        target_folder="C:\\Users\\chenwq\\Downloads\\STU-Hospital-master\\STU_Test\\images",
    )
    # args = args_parse()
    # read_directory("./data/test/origin/images")
    # creat_dataset(args, mode='augment')
