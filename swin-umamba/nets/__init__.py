from .vmunet import load_vm_model

# from .vmunet_v2 import load_vm2_model,load_vm3_model,VMC_UNet,CFFMNet
from .Swin_umamba import getmodel
from .SwinUMambaD import get_swin_umambaD

from .UMambaBot_2d import umamba_bot_model
from .model.UNetPlus import NestedUNet
from .model.AAUNet import AAUnet
from .module.Hiformer import get_hiformer
from .module.H2Former import res34_swin_MS
# from .model.Attention_UNet import  AttU_Net
from .BEFUnet.BEFUnet import get_BEFUNet
from .model.U_Lite import ULite
from .SM_Unet import get_sm_model
# from .RS3Mamba import RSMamba
from .model.UNet import UNet
from .vision_transformer import Swin_model
from .module.CDFormer import CDFormer

def net(model_name):
    if model_name == 'VMUNet':
        model = load_vm_model()
    # elif model_name == 'VMUNetv2':
    #     model = load_vm2_model()
    elif model_name == 'SwinUMamba':
        model = getmodel()
    elif model_name == 'SwinUMambaD':
        model = get_swin_umambaD()
    elif model_name == 'Former':
        # model = get_BEFUNet()
        # model = get_hiformer()
        model = res34_swin_MS(224,1)
    elif model_name == 'UMamba':
        model = umamba_bot_model()
    elif model_name == "NewSwinUM":
        model = ULite().cuda()
    elif model_name == "UNet":
        model = UNet(3,1).cuda()
    elif model_name == "CFFormer":
        model = CDFormer().cuda()
    # elif model_name == "MedFormer":
    #     model = get_med_model()
    elif model_name == "SM_UNet":
        model = get_sm_model()
    elif model_name == "AAUNet":
        model = AAUnet().cuda()
    elif model_name == "UNet++":
        model = NestedUNet().cuda()
    elif model_name == "SwinUNet":
        model = Swin_model()
    else:
        print("No model!")
        return RuntimeError
    return model

def get_dataset(datasets):
    ######## 肠息肉 ######
    if datasets == 'isic18':
        data_path = './data/isic2018/'
    elif datasets == 'isic17':
        data_path = './data/isic2017/'
    elif datasets == 'kvasir-ins':
        data_path = './data/Kvasir_Instrument/'
    elif datasets == 'kvasir':
        data_path = './data/Kvasir-Seg/'
    elif datasets == 'cvc-300':
        data_path = './data/CVC-300/'
    elif datasets == 'cvc-Colon':
        data_path = './data/CVC-ColonDB/'
    elif datasets == 'cvc-Clinic':
        data_path = './data/CVC-ClinicDB/'
    elif datasets == 'polpy':
        data_path = './data/Polpy/'
    ######## 甲状腺超声 ######
    elif datasets == 'TUS':
        data_path = '/home/cwq/MedicalDP/SwinUmamba/data/nnUNet_raw/Dataset705_Thyroid/'
    elif datasets == 'DDTI' :
        data_path = './data/DDTI/'
    ######## 乳腺超声 #########
    elif datasets == 'BUSI':
        data_path = './data/BUSI/'
    elif datasets == 'BUS':
        data_path = './data/BUS/'
    elif datasets == 'STU':
        data_path = './data/STU/'
    else:
        raise Exception('datasets name in not right!')
    return data_path


    # from ptflops import get_model_complexity_info
    # macs, params = get_model_complexity_info(model, (1,128, 128, 128), as_strings=True, print_per_layer_stat=True, verbose=True)
    # print(f'Computational complexity: {macs}')
    # print(f'Number of parameters: {params}')

import torch
import torch.nn as nn
from einops import rearrange


##  Top-K Sparse Attention (TKSA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads=4, bias=True):
        super(Attention, self).__init__()
        self.num_heads = num_heads

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.attn_drop = nn.Dropout(0.)

        self.attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        _, _, C, _ = q.shape

        mask1 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask2 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask3 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask4 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        index = torch.topk(attn, k=int(C/2), dim=-1, largest=True)[1]
        mask1.scatter_(-1, index, 1.)
        attn1 = torch.where(mask1 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C*2/3), dim=-1, largest=True)[1]
        mask2.scatter_(-1, index, 1.)
        attn2 = torch.where(mask2 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C*3/4), dim=-1, largest=True)[1]
        mask3.scatter_(-1, index, 1.)
        attn3 = torch.where(mask3 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C*4/5), dim=-1, largest=True)[1]
        mask4.scatter_(-1, index, 1.)
        attn4 = torch.where(mask4 > 0, attn, torch.full_like(attn, float('-inf')))

        attn1 = attn1.softmax(dim=-1)
        attn2 = attn2.softmax(dim=-1)
        attn3 = attn3.softmax(dim=-1)
        attn4 = attn4.softmax(dim=-1)

        out1 = (attn1 @ v)
        out2 = (attn2 @ v)
        out3 = (attn3 @ v)
        out4 = (attn4 @ v)

        out = out1 * self.attn1 + out2 * self.attn2 + out3 * self.attn3 + out4 * self.attn4

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out