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
from dataset import MedicineDataset
from thop import profile
from utils.metrics import iou_score
from utils.utils import AverageMeter
from ptflops import get_model_complexity_info
from nets.BCMamba import count_parameters,convnext_tiny,freeze_pretrained_weights_only_lora
current_date = datetime.date.today()

def compute_complexity(config):
    model = net('SwinUMamba', config['rank'], config['deep_supervision'])
    # input = torch.randn(1, 3, 256, 256).cuda()  # 确保输入在 GPU 上
    # flops, params = profile(model, inputs=(input,))
    # print('flops:{}G'.format(flops/1e9)) #转为G
    # print('params:{}M'.format(params/1e6)) #转为M

    flops, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', flops))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

def count_lora_parameters():
    m = convnext_tiny(pretrained=False, lora_rank=16, lora_alpha=32.0)
    print("Model built. Counting params before freeze:")
    count_parameters(m)  # initially all params trainable

    # Load pretrained separately if you have checkpoint, then freeze only LoRA:
    # m = convnext_tiny(pretrained=True, lora_rank=8, lora_alpha=32.0)
    freeze_pretrained_weights_only_lora(m)
    print("After freezing (only LoRA trainable):")
    count_parameters(m)

    # Example forward
    x = torch.randn(1, 3, 224, 224)
    logits, skips = m(x)
    print("Forward OK. logits shape:", logits.shape)


def count_flops_and_params(model, input_shape=( 3, 256, 256)):
    model.eval()

    with torch.no_grad():
        macs, params = get_model_complexity_info(
            model,
            input_res=input_shape,
            as_strings=True,
            print_per_layer_stat=False,
            verbose=False
        )

    print("============== Model Complexity ==============")
    print(f"Input shape: {input_shape}")
    print(f"FLOPs / MACs: {macs}")
    print(f"Params:      {params}")

    # also compute trainable & LoRA params
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # lora = sum(p.numel() for n, p in model.named_parameters() if "lora" in n.lower())
    lora = sum(p.numel() for n,p in model.named_parameters() if ("lora_A" in n) or ("lora_B" in n))

    print("----------------------------------------------")
    print(f"Total Params:     {total / 1e6:.9f} M")
    print(f"Trainable Params: {trainable / 1e9:.9f} M")
    print(f"LoRA Params:      {lora / 1e6:.9f} M")
    print("================================================")

    return macs, params


def main():
    config = vars(parse_args())
    # model = net('MedMamba', config['rank'], config['deep_supervision'])
    model = net(config['model'], config['rank'], config['deep_supervision'])
    train_epochs = config['epochs']
    model_path = os.path.join(
        config['output'],
        config['model'],
        # "MedMamba",
        # config['Ablation'],
        "BUSI",
        "BUSI_pretrained_150_3.pth")
        # f"{config['model_pth']}_{train_epochs}_{config['iteration']}.pth")
    model.load_state_dict(torch.load(model_path))
    model.eval()

    val_dataset = MedicineDataset(os.path.join(get_dataset(config['val_dataset']), "test"), mode="val", img_size=256)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = 1, shuffle=False)
    val_names = val_dataset.names
    count = 0
    top_dice_list = []
    top_k = 5

    mask_pred = os.path.join(config['output'], config['model'], config['val_dataset'])
    #pred生成路径
    file_dir = os.path.join(mask_pred, 'BUSI_3_pred_' + str(current_date.strftime("%Y-%m-%d")))
    os.makedirs(file_dir, exist_ok=True)
    file_path = file_dir + "/Metric.xlsx"

    avg_meters = {'test_iou': AverageMeter(), 'test_dice': AverageMeter(),
                  'test_acc': AverageMeter(), 'test_pc': AverageMeter(), 'test_se': AverageMeter(),
                  'test_sp': AverageMeter()}
    with torch.no_grad():
        for input, target in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()
            if config['deep_supervision']:
                output  = model(input)[0]
            else:
                output = model(input)
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
            iou, dice, SE, PC, SP, ACC, hd95 = iou_score(output, target)
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

    print(f'*************MLAggUNet模型的在 STU 测试指标结果:**************')
    print("IoU:", avg_meters['test_iou'].avg)
    print("Dice:", avg_meters['test_dice'].avg)
    print("ACC:", avg_meters['test_acc'].avg)
    print("PC:", avg_meters['test_pc'].avg)
    print("SP:", avg_meters['test_sp'].avg)
    print("SE:", avg_meters['test_se'].avg)

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

if __name__ == '__main__':
    # main()

    # count_lora_parameters()

    config = vars(parse_args())
    compute_complexity(config)
    # model = net(config['model'], config['rank'], config['deep_supervision'])
    # count_flops_and_params(model)

    # input = torch.randn(1, 3, 256, 256).cuda()  # 确保输入在 GPU 上
    # macs, params = get_model_complexity_info(model, (3,256,256), as_strings=True, print_per_layer_stat=True)
    # print(f"模型 Params and FLOPs:{params}, {macs}")

