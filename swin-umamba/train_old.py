import os
import datetime
import torch
import torch.optim as optim
from accelerate import Accelerator
from tqdm import tqdm
from nets import net,get_dataset
from nets.vision_transformer import Swin_model
from utils.SWconfig import swin_config, get_config
from dataset import MedicineDataset
from utils.losses import BCEDiceLoss
from utils.metrics import iou_score
from utils.utils import AverageMeter

current_date = datetime.date.today()


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
        results_name = config['epochs']
        model_path = os.path.join(
            config['output'],
            config['model'],
            f"{config['model_pth']}_{results_name}.pth")
        if avg_meters['val_iou'].avg > best_iou:
            best_iou = avg_meters['val_iou'].avg
            unwrapped_model = accelerator.unwrap_model(model)
            accelerator.save(unwrapped_model.state_dict(),
                             model_path)
        accelerator.print('best_iou:{}'.format(best_iou))

if __name__ == '__main__':
    main()