import argparse

from sympy import false


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="BUSI",type=str,
                        help='dataset name')
    parser.add_argument('--val_dataset', default="STU",type=str,
                        help='validation dataset name')
    parser.add_argument('--root_dir', default="./data/STU/", type=str,
                        help='dataset root path')
    parser.add_argument('--model_pth', default="BUSI_pretrained", type=str,
                        help='model root path')
    parser.add_argument('--lr', default=0.001, type=float, metavar='N',
                        help='learning rate')   # 默认学习率为0.001
    parser.add_argument('--epochs', default= 150, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', default = 24, type=int, metavar='N',help='mini-batch size')
    parser.add_argument('--img_size', type=int,
                        default = 224, help='input patch size of network input')
    parser.add_argument('--model', default="HiFormer-L", help='training model')
    parser.add_argument('--output', default="./output", help='output dir')
    parser.add_argument('--iteration', default="2", help='the number of training model')
    parser.add_argument('--Ablation', default="Ablation", help='Ablation root dir')
    parser.add_argument('--deep_supervision', default= False, type=bool, help='deepSupervisor training method')
    parser.add_argument('--optimizer', type=str,
                        default='AdamW', help='choosing optimizer AdamW or SGD')
    parser.add_argument('--augmentation',
                        default=False, help='choose to do random flip rotation')
    parser.add_argument('--seed', type=int, default = 3, help='random seed')
    parser.add_argument('--n_skip', type=int, default= 3, help='using number of skip-connect, default is num')
    parser.add_argument('--rank', type=int, default = 4, help='lora rank')
    parser.add_argument('--log_dir', type=str, default='./log_dir', help='every training epoch of metric and loss')
    args = parser.parse_args()

    return args
