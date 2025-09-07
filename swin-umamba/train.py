import os
import math
import logging
from sched import scheduler
import datetime
import numpy as np
import torch
import torch.optim as optim
from accelerate import Accelerator
from tqdm import tqdm
from nets import net,get_dataset
from nets.vision_transformer import Swin_model
from utils.SWconfig import swin_config, get_config
from dataset import Dataset, ThyroidDataset,get_loader,MedicineDataset
from utils.transforms import get_transform
from utils.config import parse_args
from utils.losses import BCEDiceLoss,HybridLossWithDynamicBoundary
from utils.metrics import iou_score
from utils.utils import AverageMeter,get_scheduler

current_date = datetime.date.today()

def cosine_annealing(step, alpha_max, T_max):

    return alpha_max * (1 + math.cos(math.pi * step / T_max)) / 2

def deep_main():
    os.environ["WANDB_API_KEY"] = 'cdc9d021d94adc1de2796c1c3be4f798060945cf'
    os.environ["WANDB_MODE"] = "offline"
    config = vars(parse_args())
    model = net(config['model'])
    # initialize accelerator
    accelerator = Accelerator(mixed_precision='fp16', log_with='wandb')
    if accelerator.is_main_process:
        accelerator.init_trackers('ph2_val', config=config, init_kwargs={'wandb': {'name': 'swin-umamba3'}})
    # train_root_path = os.path.join(config["root_dir"], 'train')
    # val_root_path = os.path.join(config["root_dir"], 'val')

    if config["dataset"] != "Polpy":
        train_dataset = MedicineDataset(os.path.join(get_dataset(config["dataset"]), "train"), mode="train")  # 785
        val_dataset = MedicineDataset(os.path.join(get_dataset(config["dataset"]), "val"), mode="val")  # 99
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=config['batch_size'], shuffle=True)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=config['batch_size'], shuffle=False)
        # train_dataset = ThyroidDataset(os.path.join(get_dataset(config["dataset"]),"train"), get_transform(train=True))
        # val_dataset = ThyroidDataset(os.path.join(get_dataset(config["dataset"]),"val"), get_transform(train=False))
        # train_loader = torch.utils.data.DataLoader(
        #     train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=train_dataset.collate_fn)
        # val_loader = torch.utils.data.DataLoader(
        #     val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=val_dataset.collate_fn)
    else:
        train_loader = get_loader(os.path.join(get_dataset(config["dataset"]),"train"),batch_size=config['batch_size'], shuffle=True, train=True)   #Kvasir_dataset
        val_loader = get_loader(os.path.join(get_dataset(config["dataset"]),"val"),batch_size=config['batch_size'],shuffle=False, train=False)

    criterion = BCEDiceLoss()   #BCEDiceLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'])
    # scheduler = get_scheduler(optimizer=optimizer)
    # model, optimizer, scheduler,train_loader, val_loader = accelerator.prepare(model, optimizer,scheduler, train_loader, val_loader)
    model, optimizer,train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)

    best_iou = 0.
    step = 0
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
    for epoch in range(config['epochs']):
        torch.cuda.empty_cache()
        accelerator.print('Epoch [%d/%d]' % (epoch + 1, config['epochs']))
        avg_meters = {'train_loss': AverageMeter(), 'val_iou': AverageMeter(), 'val_dice': AverageMeter(),
                      'val_acc': AverageMeter(), 'val_pc': AverageMeter(), 'val_se': AverageMeter(),
                      'val_sp': AverageMeter()}
        try:
            model.train()
            # alpha_max_2 = 0.75
            # alpha_max_3 = 0.50
            for iter, data in enumerate(train_loader):
                step+=iter
                current_step  = epoch * len(train_loader) + iter
                image ,mask = data
                # 检查数据范围 # print("输入数据范围:", torch.min(image), torch.max(image)) tensor(-2.1179, device='cuda:2') tensor(2.6400, device='cuda:2')
                out1,out2,out3,out4 = model(image)
                mask = torch.unsqueeze(mask,dim=1)
                '''Hybridloss'''
                # loss1 = criterion(out1, mask,current_step)
                # loss2 = criterion(out2, mask,current_step)
                # loss3 = criterion(out3, mask,current_step)
                '''BCEloss'''
                # alpha_2 = cosine_annealing(step, alpha_max_2, config["epochs"])
                # alpha_3 = cosine_annealing(step, alpha_max_3, config["epochs"])
                loss1 = criterion(out1, mask)
                loss2 = criterion(out2, mask) #* alpha_2
                loss3 = criterion(out3, mask) #* alpha_3

                loss = loss1 + 0.25*loss2 + 0.125*loss3  #0.25 0.125
                avg_meters['train_loss'].update(loss.item(), image.size(0))
                optimizer.zero_grad()
                accelerator.backward(loss)
                optimizer.step()
                # scheduler.step()
            accelerator.log({'train_loss': avg_meters['train_loss'].avg})
            print('Training Loss : {:.4f}'.format(avg_meters['train_loss'].avg))
            model.eval()

            for image, mask in tqdm(val_loader):
                with torch.no_grad():
                    pred = model(image)[0]   #vmunetv2
                pred, mask = accelerator.gather_for_metrics((pred, mask)) # torch.Size([72, 1, 256, 256]) torch.Size([72, 256, 256])
                # print(pred.shape,mask.shape)
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
                                 avg_meters['val_pc'].avg, avg_meters['val_se'].avg, avg_meters['val_sp'].avg))

            accelerator.wait_for_everyone()
            model_path = os.path.join("./output",config['model'],"Ablation_DDTI",config['model_pth']+"_HCMNet_150.pth")
            if avg_meters['val_iou'].avg > best_iou:
                best_iou = avg_meters['val_iou'].avg
                unwrapped_model = accelerator.unwrap_model(model)
                accelerator.save(unwrapped_model.state_dict(),
                                 model_path)
            accelerator.print('best_iou:{}'.format(best_iou))

        except RuntimeError as e:
            print(f"RuntimeError: {e}")
            exit(1)

def Mamba_main():
    os.environ["WANDB_API_KEY"] = 'cdc9d021d94adc1de2796c1c3be4f798060945cf'
    os.environ["WANDB_MODE"] = "offline"
    config = vars(parse_args())
    model = net(config['model'])
    # initialize accelerator
    accelerator = Accelerator(mixed_precision='fp16', log_with='wandb')
    if accelerator.is_main_process:
        accelerator.init_trackers('ph2_val', config=config, init_kwargs={'wandb': {'name': 'swin-umamba3'}})

    if config["dataset"] != "Poply":
        train_dataset = MedicineDataset(os.path.join(get_dataset(config["dataset"]),"train"), mode="train") #785
        val_dataset = MedicineDataset(os.path.join(get_dataset(config["dataset"]),"val"), mode="val") #99
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=config['batch_size'], shuffle=True)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=config['batch_size'], shuffle=False)
        # train_dataset = ThyroidDataset(os.path.join(get_dataset(config["dataset"]), "train"), get_transform(train=True))
        # val_dataset = ThyroidDataset(os.path.join(get_dataset(config["dataset"]), "val"), get_transform(train=False))
        # train_loader = torch.utils.data.DataLoader(
        #     train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=train_dataset.collate_fn)
        # val_loader = torch.utils.data.DataLoader(
        #     val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=val_dataset.collate_fn)
    else:
        train_loader = get_loader(os.path.join(get_dataset(config["dataset"]),"train"), batch_size=config['batch_size'], shuffle=True, train=True)  # Kvasir_dataset
        val_loader = get_loader(os.path.join(get_dataset(config["dataset"]),"train"), batch_size=config['batch_size'], shuffle=False, train=False)
    # print("Model train_dataset:", len(train_loader), "; val_dataset:", len(val_loader))

    criterion = BCEDiceLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'])
    # model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)
    scheduler = get_scheduler(optimizer=optimizer)
    model, optimizer, scheduler, train_loader, val_loader = accelerator.prepare(model, optimizer, scheduler,train_loader, val_loader)
    best_iou = 0.
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
    for epoch in range(config['epochs']):
        accelerator.print('Epoch [%d/%d]' % (epoch + 1, config['epochs']))
        avg_meters = {'train_loss': AverageMeter(), 'val_iou': AverageMeter(), 'val_dice': AverageMeter(),
                      'val_acc': AverageMeter(), 'val_pc': AverageMeter(), 'val_se': AverageMeter(),
                      'val_sp': AverageMeter()}
        try:
            model.train()
            # logging.getLogger("PIL").setLevel(logging.WARNING)  # 将 PIL 的日志级别设置为 WARNING
            for image, mask in tqdm(train_loader):
                # print("输入数据范围:", torch.min(image), torch.max(image))                  # 检查数据范围
                output = model(image)
                mask = torch.unsqueeze(mask,dim=1)
                loss = criterion(output, mask)
                avg_meters['train_loss'].update(loss.item(), image.size(0))

                optimizer.zero_grad()
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()

            accelerator.log({'train_loss': avg_meters['train_loss'].avg})
            print('Training Loss : {:.4f}'.format(avg_meters['train_loss'].avg))

            model.eval()

            for image, mask in tqdm(val_loader):
                with torch.no_grad():
                    pred = model(image)  #torch.Size([24, 1, 256, 256]) torch.Size([24, 256, 256])
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
            model_path = os.path.join("./output",config['model'],config['model_pth']+"_150.pth")
            if avg_meters['val_iou'].avg > best_iou:
                best_iou = avg_meters['val_iou'].avg
                unwrapped_model = accelerator.unwrap_model(model)
                accelerator.save(unwrapped_model.state_dict(),
                                 model_path)
            accelerator.print('best_iou:{}'.format(best_iou))

        except RuntimeError as e:
            print(f"RuntimeError: {e}")
            exit(1)

def main():
    os.environ["WANDB_API_KEY"] = 'cdc9d021d94adc1de2796c1c3be4f798060945cf'
    os.environ["WANDB_MODE"] = "offline"
    args = swin_config()
    config = vars(args) #Swin_UNet
    configs = get_config(args)
    # config = vars(parse_args())

    # initialize accelerator
    accelerator = Accelerator(mixed_precision='fp16', log_with='wandb')
    if accelerator.is_main_process:
        accelerator.init_trackers('ph2_val', config=config, init_kwargs={'wandb': {'name': 'swin-umamba3'}})


    train_dataset = MedicineDataset(os.path.join(get_dataset(config["dataset"]), "train"), mode="train")  # 785
    val_dataset = MedicineDataset(os.path.join(get_dataset(config["dataset"]), "val"), mode="val")  # 99
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config['batch_size'], shuffle=False)

    model = Swin_model(configs)

    criterion = BCEDiceLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'])
    model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer,
                                                                     train_loader, val_loader)

    best_iou = 0.

    for epoch in range(config['epochs']):
        accelerator.print('Epoch [%d/%d]' % (epoch + 1, config['epochs']))
        avg_meters = {'train_loss': AverageMeter(), 'val_iou': AverageMeter(), 'val_dice': AverageMeter(),
                      'val_acc': AverageMeter(), 'val_pc': AverageMeter(), 'val_se': AverageMeter(),
                      'val_sp': AverageMeter()}

        # train
        model.train()
        for image, mask in tqdm(train_loader):
        # for image ,mask in train_loader:
        #     output = model(image).squeeze()          #(12,2,224,224)
        #     mask = torch.unsqueeze(mask,dim=1)
        #     mask = torch.concat([mask, mask.clone()], dim=1)
        #     loss = criterion(output, mask[:])

            output = model(image)
            mask = torch.unsqueeze(mask, dim=1)
            loss = criterion(output, mask)
            avg_meters['train_loss'].update(loss.item(), image.size(0))

            optimizer.zero_grad()
            accelerator.backward(loss)

            optimizer.step()

        accelerator.log({'train_loss': avg_meters['train_loss'].avg})
        print('Training Loss : {:.4f}'.format(avg_meters['train_loss'].avg))

        model.eval()
        for image, mask in tqdm(val_loader):
        # for image ,mask in val_loader:
            with torch.no_grad():
                # pred = model(image).squeeze()
                pred = model(image)
            pred, mask = accelerator.gather_for_metrics((pred, mask))
            mask = torch.unsqueeze(mask, dim=1)
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
        accelerator.print('val_iou %.4f - val_dice %.4f' % (avg_meters['val_iou'].avg, avg_meters['val_dice'].avg))

        accelerator.wait_for_everyone()
        model_path = os.path.join("./output", config['model'], config['model_pth'] + "_150.pth")
        if avg_meters['val_iou'].avg > best_iou:
            best_iou = avg_meters['val_iou'].avg
            unwrapped_model = accelerator.unwrap_model(model)
            accelerator.save(unwrapped_model.state_dict(),
                             model_path)
        accelerator.print('best_iou:{}'.format(best_iou))


if __name__ == '__main__':
    config = vars(parse_args())
    if config['deepSupervisor']:
        deep_main()
    else:
        Mamba_main()
    # python - m  torch.distributed.run - -nproc_per_node =0,1 train.py
    # main()

