import argparse

from sympy import false


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="BUS",type=str,
                        help='dataset name')
    parser.add_argument('--root_dir', default="./data/BUS/", type=str,
                        help='dataset root path')
    parser.add_argument('--model_pth', default="BUS_pretrained", type=str,
                        help='model root path')
    parser.add_argument('--lr', default=0.001, type=float, metavar='N',
                        help='learning rate')   # 默认学习率为0.001
    parser.add_argument('--epochs', default= 150, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', default=8, type=int, metavar='N',help='mini-batch size')
    parser.add_argument('--img_size', type=int,
                        default=256, help='input patch size of network input')
    parser.add_argument('--model', default="SwinUMambaD", help='training model')
    parser.add_argument('--deepSupervisor', default=True,type=bool,help='deepSupervisor model')
    parser.add_argument('--optimizer', type=str,
                        default='AdamW', help='choosing optimizer AdamW or SGD')
    parser.add_argument('--augmentation',
                        default=False, help='choose to do random flip rotation')
    args = parser.parse_args()

    return args
