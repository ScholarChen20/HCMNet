import heapq
import os

import numpy as np
import pandas as pd

import cv2
import torch
from future.backports.datetime import datetime
import datetime
from matplotlib import pyplot as plt
from pydantic.v1.utils import get_model

from tqdm import tqdm
from utils.config import  parse_args
from nets import get_dataset,net
from nets.vision_transformer import SwinUnet, Swin_model
from dataset import Dataset, ThyroidDataset, PolypDataset,MedicineDataset
from thop import profile
from utils.metrics import iou_score
from utils.utils import AverageMeter
from utils.SWconfig import get_config,swin_config
from ptflops import get_model_complexity_info
current_date = datetime.date.today()

def compute_complexity(config):
    model = net(config['model'])
    # input = torch.randn(1, 3, 256, 256).cuda()  # 确保输入在 GPU 上
    # flops, params = profile(model, inputs=(input,))
    # print('flops:{}G'.format(flops/1e9)) #转为G
    # print('params:{}M'.format(params/1e6)) #转为M

    flops, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', flops))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

def Mamba_main():
    config = vars(parse_args())
    model = net(config['model'])
    #需要改
    train_epochs = config['epochs']
    model_path = os.path.join(
        config['output'],
        config['model'],
        # 'BUS',
        config['dataset'],
        f"{config['model_pth']}_{train_epochs}_{config['iteration']}.pth")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    #需要改
    val_dataset = MedicineDataset(os.path.join(get_dataset(config["dataset"]), "test"), mode="val", img_size=config['img_size'])
    # val_dataset = ThyroidDataset(os.path.join(get_dataset(config['dataset']),"test"), get_transform(train=False))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = 16, shuffle=False)
    # val_dataset = MedicineDataset(os.path.join(get_dataset(config['dataset']), "test"), mode="val")
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)
    # val_dataset = PolypDataset(os.path.join(get_dataset(config['dataset']),"val-seg"),load_transform(train=False))
    # val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
    #                                           batch_size=24,
    #                                           shuffle=False,
    #                                           collate_fn=PolypDataset.collate_fn)
    val_names = val_dataset.names
    count = 0
    top_dice_list = []
    top_k = 5

    #需要改
    mask_pred = os.path.join(config['output'], config['model'],config['dataset'])
    #pred生成路径
    file_dir = os.path.join(mask_pred, config['iteration'] + '_pred_BEFUNet_' + str(current_date.strftime("%Y-%m-%d")))
    os.makedirs(file_dir, exist_ok=True)
    file_path = file_dir + "/Metric.xlsx"

    avg_meters = {'test_iou': AverageMeter(), 'test_dice': AverageMeter(),
                  'test_acc': AverageMeter(), 'test_pc': AverageMeter(), 'test_se': AverageMeter(),
                  'test_sp': AverageMeter()}
    with torch.no_grad():
        for input, target in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()
            output  = model(input)
            mask = output.clone()
            mask = torch.sigmoid(mask).cpu().numpy() > 0.5
            #需要改
            for i in range(len(mask)):
                cv2.imwrite(
                    os.path.join(file_dir, val_names[count].split('.')[0] + '.png'),
                    (mask[i, 0] * 255).astype('uint8'))
                count = count + 1
                val_names_batch = val_names[count:count + len(mask)]  # 更新 val_names_batch
            target = torch.unsqueeze(target,dim=1)
            iou, dice, SE, PC, SP, ACC = iou_score(output, target)
            avg_meters['test_iou'].update(iou, input.size(0))
            avg_meters['test_dice'].update(dice, input.size(0))
            avg_meters['test_acc'].update(ACC, input.size(0))
            avg_meters['test_pc'].update(PC, input.size(0))
            avg_meters['test_se'].update(SE, input.size(0))
            avg_meters['test_sp'].update(SP, input.size(0))
            for name in val_names_batch:
                top_dice_list.append((dice, name))  # 注意：这里用的是 batch-level 的 dice

            # 维护 top-k 的最大堆
            if len(top_dice_list) < top_k:
                heapq.heappush(top_dice_list, (dice, val_names[count - len(mask)]))
            else:
                heapq.heappushpop(top_dice_list, (dice, val_names[count - len(mask)]))

    print(f'*************{config["model"]}模型的在{config["dataset"]}_测试指标结果:********')
    print("IoU:", avg_meters['test_iou'].avg)
    print("Dice:", avg_meters['test_dice'].avg)
    print("ACC:", avg_meters['test_acc'].avg)
    print("PC:", avg_meters['test_pc'].avg)
    print("SE:", avg_meters['test_se'].avg)
    print("SP:", avg_meters['test_sp'].avg)

    # for dice_val, name in sorted(top_dice_list, reverse=True):
    #     print(f"File: {name}, Dice: {dice_val:.4f}")
    #  将 top-k 文件名添加到 metrics
    top_dice_sorted = sorted(top_dice_list, key=lambda x: x[0], reverse=True)[0:top_k]
    metrics = {
        'Metric': ['IOU', 'DICE', 'ACC', 'PC', 'SE', 'SP'],
        'Value': [avg_meters['test_iou'].avg, avg_meters['test_dice'].avg, avg_meters['test_acc'].avg,
                  avg_meters['test_pc'].avg, avg_meters['test_se'].avg, avg_meters['test_sp'].avg]
    }
    # 添加 Top-k 文件名
    metrics['Metric'].extend([f'Top-{i + 1} Dice' for i in range(len(top_dice_sorted))])
    metrics['Value'].extend([f"{name} (Dice: {dice:.4f})" for dice, name in top_dice_sorted])
    # 将数据转换为 pandas DataFrame
    df = pd.DataFrame(metrics)

    # 检查文件是否已存在
    if not os.path.exists(file_path):
        # 如果文件不存在，写入新的文件
        df.to_excel(file_path, index=False)
    else:
        # 如果文件已存在，读取现有文件并追加新数据
        existing_df = pd.read_excel(file_path)
        new_df = pd.concat([existing_df, df], ignore_index=True)
        new_df.to_excel(file_path, index=False)
    torch.cuda.empty_cache()

def deep_main():
    # print("=> creating model")
    config = vars(parse_args())
    model = net(config['model'])
    # flops,params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=True)
    # print(f"模型 Params and FLOPs:{params}, {flops}")
    train_epochs = config['epochs']
    model_path = os.path.join(
        config['output'],
        config['model'],
        config['dataset'],
        f"{config['model_pth']}_{train_epochs}.pth")
    model.load_state_dict(torch.load(model_path),False)
    model.eval()
    val_dataset = MedicineDataset(os.path.join(get_dataset(config["dataset"]), "test"), mode="val")  # 99
    # val_dataset = ThyroidDataset(os.path.join(get_dataset(config['dataset']),"test"), get_transform(train=False))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = 4, shuffle=False,)
    # val_dataset = MedicineDataset(os.path.join(get_dataset(config['dataset']), "test"), mode="val")
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)
    # val_dataset = PolypDataset(os.path.join(get_dataset(config['dataset']),"val-seg"),load_transform(train=False))
    # val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=24, shuffle=False, collate_fn=PolypDataset.collate_fn)
    val_names = val_dataset.names
    count = 0
    top_dice_list = []  # 保存 top-k 的最大堆
    top_k = 5  # 保存 top-k 的最大堆
    mask_pred = os.path.join(config['output'], config['model'], config['dataset'] + "_seg")

    # 文件路径
    file_dir = os.path.join(mask_pred, config['iteration'] + '_pred_' + str(current_date.strftime("%Y-%m-%d")))
    os.makedirs(file_dir, exist_ok=True)
    file_path = file_dir + "/Metric.xlsx"

    avg_meters = {'test_iou': AverageMeter(), 'test_dice': AverageMeter(),
                  'test_acc': AverageMeter(), 'test_pc': AverageMeter(),
                  'test_se': AverageMeter(), 'test_sp': AverageMeter()}
    with torch.no_grad():
        for input, target in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()
            # print("输入数据范围:", torch.min(target), torch.max(target))
            output  = model(input)[0]
            mask = output.clone()
            mask = torch.sigmoid(mask).cpu().numpy() > 0.5
            #需要改
            for i in range(len(mask)):
                cv2.imwrite(   # 保存为png格式
                    os.path.join(file_dir, val_names[count].split('.')[0] + '.png'),
                    (mask[i, 0] * 255).astype('uint8'))
                count = count + 1
                val_names_batch = val_names[count:count + len(mask)] # 更新 val_names_batch
            target = torch.unsqueeze(target,dim=1)
            iou, dice, SE, PC, SP, ACC = iou_score(output, target)
            avg_meters['test_iou'].update(iou, input.size(0))
            avg_meters['test_dice'].update(dice, input.size(0))
            avg_meters['test_acc'].update(ACC, input.size(0))
            avg_meters['test_pc'].update(PC, input.size(0))
            avg_meters['test_se'].update(SE, input.size(0))
            avg_meters['test_sp'].update(SP, input.size(0))

            for name in val_names_batch:
                top_dice_list.append((dice, name))  # 注意：这里用的是 batch-level 的 dice
            # 维护 top-k 的最大堆
            if len(top_dice_list) < top_k:
                heapq.heappush(top_dice_list, (dice, val_names[count - len(mask)]))
            else:
                heapq.heappushpop(top_dice_list, (dice, val_names[count - len(mask)]))


    print(f'*************{config["model"]}模型的在{config["dataset"]}_测试指标结果:********')
    print("IoU:", avg_meters['test_iou'].avg)
    print("Dice:", avg_meters['test_dice'].avg)
    print("ACC:", avg_meters['test_acc'].avg)
    print("PC:", avg_meters['test_pc'].avg)
    print("SE:", avg_meters['test_se'].avg)
    print("SP:", avg_meters['test_sp'].avg)
    #  将 top-k 文件名添加到 metrics
    top_dice_sorted = sorted(top_dice_list, key=lambda x: x[0], reverse=True)[0:top_k]
    for dice, name in top_dice_sorted:
        print(f"Top-{top_dice_sorted.index((dice, name)) + 1} Dice: {dice:.4f}, File: {name}")
    metrics = {
        'Metric': ['IOU', 'DICE', 'ACC', 'PC', 'SE', 'SP'],
        'Value': [avg_meters['test_iou'].avg, avg_meters['test_dice'].avg, avg_meters['test_acc'].avg,
                  avg_meters['test_pc'].avg, avg_meters['test_se'].avg, avg_meters['test_sp'].avg]
    }
    # 添加 Top-k 文件名
    metrics['Metric'].extend([f'Top-{i + 1} Dice' for i in range(len(top_dice_sorted))])
    metrics['Value'].extend([f"{name} (Dice: {dice:.4f})" for dice, name in top_dice_sorted])
    # 将数据转换为 pandas DataFrame
    df = pd.DataFrame(metrics)

    # 检查文件是否已存在
    if not os.path.exists(file_path):
        # 如果文件不存在，写入新的文件
        df.to_excel(file_path, index=False)
    else:
        # 如果文件已存在，读取现有文件并追加新数据,新增一行空行
        df = pd.DataFrame(columns=df.columns)  # 新增一行空行
        existing_df = pd.read_excel(file_path)
        new_df = pd.concat([existing_df, df], ignore_index=True)
        new_df.to_excel(file_path, index=False)
    torch.cuda.empty_cache()

if __name__ == '__main__':
    config = vars(parse_args())
    if config['deepSupervisor']:
        deep_main()
    else:
        Mamba_main()

    # compute_complexity(config)  # todo 测试模型参数