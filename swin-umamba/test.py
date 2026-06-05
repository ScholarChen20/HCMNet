import heapq
import os
import pandas as pd
import cv2
import torch
import datetime

from thop import profile
from tqdm import tqdm
from utils.config import  parse_args
from nets import get_dataset,net
from dataset import Dataset, ThyroidDataset, PolypDataset,MedicineDataset
from utils.metrics import iou_score_per_sample
from utils.utils import AverageMeter
from ptflops import get_model_complexity_info
current_date = datetime.date.today()

def compute_complexity(config):
    model = net(config['model'], config['rank'], config['deep_supervision'])
    # model = net("VMUNetv2", 4, False)
    input = torch.randn(1, 3, 256, 256).cuda()  # 确保输入在 GPU 上
    flops, params = profile(model, inputs=(input,))
    print('flops:{}G'.format(flops/1e9)) #转为G
    print('params:{}M'.format(params/1e6)) #转为M

    # flops, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=True)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', flops))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))

def main(config):
    model = net(config['model'], config['rank'], config['deep_supervision'])
    train_epochs = config['epochs']
    model_path = os.path.join(
        config['output'],
        config['model'],
        # config['Ablation'],   #  消融路劲配置
        config['dataset'],
        # f"{config['model_pth']}_{train_epochs}_{config['ablaType']}_{config['iteration']}.pth")
        f"{config['model_pth']}_{train_epochs}_{config['iteration']}.pth")
    model.load_state_dict(torch.load(model_path))
    model.eval()

    val_dataset = MedicineDataset(os.path.join(get_dataset(config["dataset"]), "train"), mode="val", img_size=config['img_size'])
    # val_dataset = ThyroidDataset(os.path.join(get_dataset(config['dataset']),"test"), get_transform(train=False))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = 16, shuffle=False)
    # val_dataset = PolypDataset(os.path.join(get_dataset(config['dataset']),"val-seg"),load_transform(train=False))
    # val_loader = torch.utils.data.DataLoader(dataset=val_dataset,batch_size=24,shuffle=False,collate_fn=PolypDataset.collate_fn)

    val_names = val_dataset.names
    count = 0
    top_dice_heap = []  # 最小堆，维护 top-k 最高 dice 的样本
    top_k = 5

    #掩码pred-生成路径  config['Ablation'] \  config['ablaType'] + "_" +
    mask_pred = os.path.join(config['output'], config['model'], config['dataset'])
    # mask_pred = os.path.join(config['output'], config['model'], config['Ablation'], config['dataset'])
    file_dir = os.path.join(mask_pred, "train_" + config['iteration'] + '_pred_' + str(current_date.strftime("%Y-%m-%d")))
    os.makedirs(file_dir, exist_ok=True)
    file_path = file_dir + "/Metric.xlsx"

    avg_meters = {'test_iou': AverageMeter(), 'test_dice': AverageMeter(),'test_acc': AverageMeter(), 'test_pc': AverageMeter(),
                  'test_se': AverageMeter(),'test_sp': AverageMeter(), 'test_hd95': AverageMeter()}
    with torch.no_grad():
        for input, target in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()
            if config['deep_supervision']:
                output = model(input)[0]
            else:
                output = model(input)
            mask = output.clone()
            mask = torch.sigmoid(mask).cpu().numpy() > 0.5

            # 逐样本计算指标
            sample_metrics = iou_score_per_sample(output, target)

            for i in range(len(mask)):
                # 保存预测掩码
                cv2.imwrite(os.path.join(file_dir, val_names[count].split('.')[0] + '.png'),
                           (mask[i, 0] * 255).astype('uint8'))

                # 更新逐样本指标到 AverageMeter（n=1，每个样本独立）
                sample_dice = sample_metrics['dice'][i]
                avg_meters['test_iou'].update(sample_metrics['iou'][i], 1)
                avg_meters['test_dice'].update(sample_dice, 1)
                avg_meters['test_acc'].update(sample_metrics['acc'][i], 1)
                avg_meters['test_pc'].update(sample_metrics['pc'][i], 1)
                avg_meters['test_se'].update(sample_metrics['se'][i], 1)
                avg_meters['test_sp'].update(sample_metrics['sp'][i], 1)
                avg_meters['test_hd95'].update(sample_metrics['hd95'][i], 1)

                # 维护 top-k 最小堆（堆顶是最小的 dice，新元素大于堆顶时替换）
                sample_name = val_names[count]
                if len(top_dice_heap) < top_k:
                    heapq.heappush(top_dice_heap, (sample_dice, sample_name))
                elif sample_dice > top_dice_heap[0][0]:
                    heapq.heappushpop(top_dice_heap, (sample_dice, sample_name))

                count = count + 1

    print(f'*************{config["model"]}模型的在{config["dataset"]}_测试指标结果:********')
    print(f"IoU: {avg_meters['test_iou'].avg*100:.2f}±{avg_meters['test_iou'].std()*100:.2f}")
    print(f"Dice: {avg_meters['test_dice'].avg*100:.2f}±{avg_meters['test_dice'].std()*100:.2f}")
    print(f"ACC: {avg_meters['test_acc'].avg*100:.2f}±{avg_meters['test_acc'].std()*100:.2f}")
    print(f"PC: {avg_meters['test_pc'].avg*100:.2f}±{avg_meters['test_pc'].std()*100:.2f}")
    print(f"SP: {avg_meters['test_sp'].avg*100:.2f}±{avg_meters['test_sp'].std()*100:.2f}")
    print(f"SE: {avg_meters['test_se'].avg*100:.2f}±{avg_meters['test_se'].std()*100:.2f}")
    print(f"HD95: {avg_meters['test_hd95'].avg:.2f}±{avg_meters['test_hd95'].std():.2f}")



    top_dice_sorted = sorted(top_dice_heap, key=lambda x: x[0], reverse=True)[:top_k]
    metrics = {
        'Metric': ['IOU', 'DICE', 'ACC', 'PC', 'SE', 'SP', 'HD95'],
        'Value': [f"{avg_meters['test_iou'].avg*100:.2f}±{avg_meters['test_iou'].std()*100:.2f}",
                  f"{avg_meters['test_dice'].avg*100:.2f}±{avg_meters['test_dice'].std()*100:.2f}",
                  f"{avg_meters['test_acc'].avg*100:.2f}±{avg_meters['test_acc'].std()*100:.2f}",
                  f"{avg_meters['test_pc'].avg*100:.2f}±{avg_meters['test_pc'].std()*100:.2f}",
                  f"{avg_meters['test_se'].avg*100:.2f}±{avg_meters['test_se'].std()*100:.2f}",
                  f"{avg_meters['test_sp'].avg*100:.2f}±{avg_meters['test_sp'].std()*100:.2f}",
                  f"{avg_meters['test_hd95'].avg:.2f}±{avg_meters['test_hd95'].std():.2f}"]
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
    config = vars(parse_args())
    main(config)
    #
    # compute_complexity(config)  # todo 测试模型参数