import os
import math
import logging
import pandas as pd
import random
from sched import scheduler
import datetime
import numpy as np
import torch
import torch.optim as optim
from accelerate import Accelerator
from tqdm import tqdm
from nets import net,get_dataset
from dataset import Dataset, ThyroidDataset,get_loader,MedicineDataset
from utils.config import parse_args
from utils.losses import BCEDiceLoss,HybridLossWithDynamicBoundary
from utils.metrics import iou_score
from utils.utils import AverageMeter,get_scheduler
current_date = datetime.date.today()
import logging

def set_random_seed(seed):
    """set random seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def cosine_annealing(step, alpha_max, T_max):
    return alpha_max * (1 + math.cos(math.pi * step / T_max)) / 2

def is_deep_supervision(accelerator, avg_meters, criterion, epoch, model, optimizer, scheduler, step, train_loader):
    """deep_supervision training strategy
        set L1:L2:L3 is 0.5:0.25:0.125"""
    for iter, data in enumerate(train_loader):
        step += iter
        current_step = epoch * len(train_loader) + iter
        image, mask = data
        out1, out2, out3, out4 = model(image)
        mask = torch.unsqueeze(mask, dim=1)
        '''HybridLoss'''
        loss1 = criterion(out1, mask, current_step)
        loss2 = criterion(out2, mask, current_step)
        loss3 = criterion(out3, mask, current_step)
        loss = loss1 * 0.5 + 0.25 * loss2 + 0.125 * loss3 # 0.25 0.125
        '''BCEDiceLoss'''
        # loss = criterion(out1, mask) + 0.25 * criterion(out2, mask) + 0.125 * criterion(out3, mask)  # 0.25 0.125
        avg_meters['train_loss'].update(loss.item(), image.size(0))
        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()

def no_deep_supervision(accelerator, avg_meters, criterion, model, optimizer, scheduler, train_loader):
    """no deep_supervision training strategy"""
    for image, mask in tqdm(train_loader):
        output = model(image)
        mask = torch.unsqueeze(mask, dim=1)
        loss = criterion(output, mask)
        avg_meters['train_loss'].update(loss.item(), image.size(0))

        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()

def main():
    config = vars(parse_args())
    model = net(config['model'], config['rank'], config['deep_supervision'])

    accelerator = Accelerator(mixed_precision='fp16', log_with='wandb')
    if accelerator.is_main_process:
        accelerator.init_trackers('ph2_val', config=config, init_kwargs={'wandb': {'name': 'swin-umamba'}})

    if config["dataset"] != "Poply":
        train_dataset = MedicineDataset(os.path.join(get_dataset(config["dataset"]),"train"), mode="train", img_size=config['img_size']) #785
        val_dataset = MedicineDataset(os.path.join(get_dataset(config["dataset"]),"val"), mode="val", img_size=config['img_size']) #99
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    else:
        train_loader = get_loader(os.path.join(get_dataset(config["dataset"]),"train"), batch_size=config['batch_size'], shuffle=True, train=True)  # Kvasir_dataset
        val_loader = get_loader(os.path.join(get_dataset(config["dataset"]),"train"), batch_size=config['batch_size'], shuffle=False, train=False)

    criterion = BCEDiceLoss()  #  HybridLossWithDynamicBoundary
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'])
    # model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)
    scheduler = get_scheduler(optimizer=optimizer)
    model, optimizer, scheduler, train_loader, val_loader = accelerator.prepare(model, optimizer, scheduler,train_loader, val_loader)
    best_iou = 0.
    deep_supervision  = config['deep_supervision']
    step = 0
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
    for epoch in range(config['epochs']):
        accelerator.print('Epoch [%d/%d]' % (epoch + 1, config['epochs']))
        avg_meters = {'train_loss': AverageMeter(), 'val_iou': AverageMeter(), 'val_dice': AverageMeter(),
                      'val_acc': AverageMeter(), 'val_pc': AverageMeter(), 'val_se': AverageMeter(),
                      'val_sp': AverageMeter()}
        try:
            model.train()
            if deep_supervision:
                is_deep_supervision(accelerator, avg_meters, criterion, epoch, model, optimizer, scheduler, step, train_loader)
            else:
                no_deep_supervision(accelerator, avg_meters, criterion, model, optimizer, scheduler, train_loader)
            accelerator.log({'train_loss': avg_meters['train_loss'].avg})
            print('Training Loss : {:.4f}'.format(avg_meters['train_loss'].avg))
            model.eval()

            for image, mask in tqdm(val_loader):
                with torch.no_grad():
                    if deep_supervision:
                        pred = model(image)[0]
                    else:
                        pred = model(image)   #torch.Size([24, 1, 256, 256]) torch.Size([24, 256, 256])
                pred, mask = accelerator.gather_for_metrics((pred, mask))
                mask = torch.unsqueeze(mask,dim=1)
                iou, dice, SE, PC, SP, ACC = iou_score(pred, mask)
                avg_meters['val_iou'].update(iou, image.size(0))
                avg_meters['val_dice'].update(dice, image.size(0))
                avg_meters['val_acc'].update(ACC, image.size(0))
                avg_meters['val_pc'].update(PC, image.size(0))
                avg_meters['val_se'].update(SE, image.size(0))
                avg_meters['val_sp'].update(SP, image.size(0))

            accelerator.log({'val_iou': avg_meters['val_iou'].avg, 'val_dice': avg_meters['val_dice'].avg,
                             'val_acc': avg_meters['val_acc'].avg, 'val_pc': avg_meters['val_pc'].avg,
                             'val_se': avg_meters['val_se'].avg, 'val_sp': avg_meters['val_sp'].avg})
            accelerator.print('val_iou %.4f - val_dice %.4f - val_acc %.4f -val_pc %.4f - val_se %.4f - val_sp %.4f'
                              % (avg_meters['val_iou'].avg, avg_meters['val_dice'].avg, avg_meters['val_acc'].avg,
                                 avg_meters['val_pc'].avg,avg_meters['val_se'].avg,avg_meters['val_sp'].avg))
            accelerator.wait_for_everyone()
            train_epochs = config['epochs']
            model_path = os.path.join(
                config['output'],
                config['model'], config['Ablation'],
                config['dataset'],
                f"{config['model_pth']}_{train_epochs}_{config['iteration']}.pth" )
            if avg_meters['val_iou'].avg > best_iou:
                best_iou = avg_meters['val_iou'].avg
                unwrapped_model = accelerator.unwrap_model(model)
                accelerator.save(unwrapped_model.state_dict(), model_path)
            accelerator.print('best_iou:{}'.format(best_iou))
            log_data.append({
                'Epoch': epoch + 1,
                'TrainLoss': avg_meters['train_loss'].avg,
                'Dice': avg_meters['val_dice'].avg
            })
            df = pd.DataFrame(log_data)
            df.to_excel(log_file, index=False)

        except RuntimeError as e:
            print(f"RuntimeError: {e}")
            exit(1)

if __name__ == '__main__':
    config = vars(parse_args())
    if config['seed'] >= 0:
        logging.info("Setting fixed seed: {}".format(config['seed']))
        # set_random_seed(config['seed'])
    os.makedirs(os.path.join(config['output'],config['model'],config['dataset']), exist_ok=True)  # 创建模型输出文件夹
    os.makedirs(os.path.join(config['log_dir'],config['dataset']), exist_ok=True) # 创建日志文件夹
    log_data = []
    log_file = os.path.join(config['log_dir'], config['dataset'], f"{config['model']}_{config['epochs']}.xlsx")

    main()
    # python - m  torch.distributed.run - -nproc_per_node =0,1 train.py
