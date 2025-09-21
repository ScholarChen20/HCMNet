import os
import random
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F
from utils.transforms import get_transform,load_transform
from usaugment.albumentations import DepthAttenuation, GaussianShadow, HazeArtifact, SpeckleReduction

# augmentation = A.Compose([
#     A.HorizontalFlip(p=0.5),  # 随机水平翻转
#     A.VerticalFlip(p=0.5),    # 随机垂直翻转
#     A.RandomRotate90(p=0.5),  # 随机旋转 90 度
#     A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),  # 平移、缩放和旋转
#     A.RandomBrightnessContrast(p=0.5),  # 随机亮度对比度调整
#     A.GaussianBlur(p=0.2),              # 高斯模糊
#     A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化
#     ToTensorV2()  # 转换为 PyTorch 张量
# ])
class RandomHorizontalFlip(object):
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class RandomVerticalFlip(object):
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.vflip(image)
            target = F.vflip(target)
        return image, target

# transformer 等比例缩放图片
class ResizeKeepAspectRatio(object):
    def __init__(self, target_size=224):
        # target_size是最大边的大小，传入整数
        self.target_size = target_size

    def __call__(self, image, target):
        width, height = image.size
        aspect_ratio = width / height
        if aspect_ratio > 1:
            new_width = self.target_size
            new_height = int(self.target_size / aspect_ratio)
        else:
            new_height = self.target_size
            new_width = int(self.target_size * aspect_ratio)

        return image.resize((new_width, new_height), Image.LANCZOS),target.resize((new_width, new_height),Image.LANCZOS)

class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target


class Normalize(object):
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target

# # 在transform中使用自定义的resize转换器
# transform_val = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
# transform_train = transforms.Compose([
#     transforms.Resize((224,224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

class MedicineDataset(Dataset):
    def __init__(self, rootdir,mode):
        self.mode = mode
        self.rootdir = rootdir
        self.names = [i for i in os.listdir(os.path.join(self.rootdir,'images'))]
        self.image_list = [os.path.join(self.rootdir, 'images', i) for i in self.names]
        self.mask_list = [os.path.join(self.rootdir, 'masks', i.replace(".jpg",".png")) for i in self.names]
        # check files
        for i in self.image_list:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")
        for i in self.mask_list:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")
        self.val_transform = A.Compose([
            A.Resize(256, 256),
            A.Normalize(
                mean=[0.45],  # 根据数据集统计调整
                std=[0.25],  # 更适应超声低对比度特性
                max_pixel_value=255.0
            ),
            ToTensorV2()],is_check_shapes= False)  #,is_check_shapes=False
        self.train_transform =  A.Compose([
            A.RandomResizedCrop((256,256), scale=(0.8, 1.0), p=1.0),
            A.Resize(256, 256),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, border_mode=cv2.BORDER_CONSTANT,
                               value=0, p=0.5),
            A.HorizontalFlip(p=0.5),  # 垂直翻转
            A.VerticalFlip(p=0.5),  # 水平翻转
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.4),   #颜色抖动
            A.OneOf([
                A.ToGray(p=1),
                A.GridDistortion(border_mode=cv2.BORDER_CONSTANT, value=0,p=1),
                A.ElasticTransform(50,5,border_mode=cv2.BORDER_CONSTANT, value=0,p=0.5),
                A.CLAHE(p=1),
                A.HueSaturationValue(p=1),
                # A.ChannelShuffle(p=1),
                A.GridDropout(p=1),
                A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1),
                A.GaussianBlur(p=1),
                # 添加高斯噪声（模拟超声斑点噪声）
                A.GaussNoise(var_limit=(10.0, 50.0), p=1),
                # 局部遮挡（模拟探头接触不良）
                A.CoarseDropout(max_holes=8, max_height=20, max_width=20, p=1),
                # 自适应直方图均衡化（增强低对比度区域）
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1),
                     ], p=0.4),  # 垂直翻转
            #A.GaussianBlur(p=0.2),
            A.Normalize(    # 0.45 0.25
                mean=[0.45],  # 单通道均值（若图像是灰度）
                std=[0.25],  # 单通道标准差
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ])

    def __len__(self):
        # return self.path_df.shape[0]
        return len(self.image_list)  # df结构的shape[0]就是样本数，shape[1]可以理解为特征数
    def __getitem__(self, idx):
        # image = Image.open(self.image_list[idx]).convert('RGB')
        # mask = Image.open(self.mask_list[idx]).convert('L')
        image=cv2.imread(self.image_list[idx],cv2.IMREAD_COLOR)
        mask=cv2.imread(self.mask_list[idx],cv2.IMREAD_GRAYSCALE)
        _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
        # mask = torch.where((mask/ 255.0)> 0.5, 1, 0)
        if self.mode == 'train':
            augmentation = self.train_transform(image=image,mask=mask)               # 修改数据增强方式
            image = augmentation['image']
            mask = augmentation['mask']
            # return image, mask

        if self.mode == 'val':
            augmentation = self.val_transform(image=image, mask=mask)
            image = augmentation['image']
            mask = augmentation['mask']
            # return image, mask
        assert torch.all(image >= -3.0) and torch.all(image <= 3.0), f"图像数据异常: {image.min()}, {image.max()}"
        # assert torch.all(mask == 0) or torch.all(mask == 1), f"掩码数据异常: {mask.unique()}"
        return image, mask.float()

class Dataset(Dataset):
    def __init__(self, root_dir, transform=None, image_size=(224, 224)):
        self.image_size = image_size
        self.root_dir = root_dir
        self.transforms = transform

        self.names = [i for i in os.listdir(os.path.join(self.root_dir, 'images'))]
        # self.image_list = [os.path.join(self.root_dir, 'images', i) for i in self.names]
        self.image_list = [os.path.join(self.root_dir, 'images', i) for i in self.names]
        self.mask_list = [os.path.join(self.root_dir, 'masks', i.replace(".jpg",".png")) for i in self.names]

        # check files
        for i in self.image_list:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")
        for i in self.mask_list:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")

    def __getitem__(self, idx):
        image = Image.open(self.image_list[idx]).convert('RGB')
        # image = transform(image)
        mask = Image.open(self.mask_list[idx]).convert('L')
        # mask = transform(mask)
        if self.transforms is not None:
            image, mask = self.transforms(image, mask)
        return image, mask / 255.0

    def __len__(self):
        return len(self.image_list)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets

def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs

class ThyroidDataset(Dataset):
    def __init__(self, root_dir, transform=None,image_size=(224,224)):
        self.image_size = image_size
        self.root_dir = root_dir
        self.transforms = transform

        self.names = [i for i in os.listdir(os.path.join(self.root_dir, 'images'))]
        #self.image_list = [os.path.join(self.root_dir, 'images', i) for i in self.names]
        self.image_list = [os.path.join(self.root_dir, 'images', i) for i in self.names]
        self.mask_list = [os.path.join(self.root_dir, 'masks', i.replace('.jpg','.png')) for i in self.names]

        # check files
        for i in self.image_list:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")
        for i in self.mask_list:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")

    def __getitem__(self, idx):
        image = Image.open(self.image_list[idx]).convert('RGB')
        # image = transform(image)
        mask = Image.open(self.mask_list[idx]).convert('L')
        # mask = transform(mask)
        if self.transforms is not None:
            image, mask = self.transforms(image, mask)
        return image, mask / 255.0

    def __len__(self):
        return len(self.image_list)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)          #图像填充，填充值为0
        batched_targets = cat_list(targets, fill_value=255)     #mask填充值为255
        return batched_imgs, batched_targets

class PolypDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transforms = transform

        self.names = [i for i in os.listdir(os.path.join(self.root_dir, 'images'))]
        #self.image_list = [os.path.join(self.root_dir, 'images', i) for i in self.names]
        self.image_list = [os.path.join(self.root_dir, 'images', i) for i in self.names]
        self.mask_list = [os.path.join(self.root_dir, 'masks', i) for i in self.names]

        # check files
        for i in self.image_list:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")
        for i in self.mask_list:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")

    def __getitem__(self, idx):
        image = Image.open(self.image_list[idx]).convert('RGB')
        # image = transform(image)
        mask = Image.open(self.mask_list[idx]).convert('L')
        # image = np.array(Image.open(self.image_list[idx]).convert('RGB'))
        # # isic 数据集未做二值化处理
        # mask = np.expand_dims(np.array(Image.open(self.mask_list[idx]).convert('L')), axis=2) / 255
        if self.transforms is not None:
            image, mask = self.transforms(image, mask)
        return image, mask/255.0
    def __len__(self):
        return len(self.image_list)

    # @staticmethod
    # def collate_fn(batch):
    #     images, targets = list(zip(*batch))
    #     batched_imgs = cat_list(images, fill_value=0)          #图像填充，填充值为0
    #     batched_targets = cat_list(targets, fill_value=255)     #mask填充值为255
    #     return batched_imgs, batched_targets

def get_loader(image_path, batch_size, shuffle=True,train=True):
    if train:
        dataset = PolypDataset(image_path, transform=load_transform(train=True))
        data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  )
    else:
        dataset = PolypDataset(image_path, transform=load_transform(train=False))
        data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  )

    return data_loader

def get_loader2(image_path, batch_size, shuffle=True, train=True):
    if train:
        dataset = ThyroidDataset(image_path, get_transform(train=True))
    else:
        dataset = ThyroidDataset(image_path, get_transform(train=False))
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  collate_fn=ThyroidDataset.collate_fn)
    return data_loader