# Mamba-SSM 安装报错

[rank2]:   File "/home/cwq/anaconda3/envs/vim/lib/python3.10/site-packages/mamba_ssm/ops/selective_scan_interface.py", line 358, in backward
[rank2]:     dx, dconv1d_weight, dconv1d_bias, *_ = causal_conv1d_cuda.causal_conv1d_bwd(
[rank2]: TypeError: causal_conv1d_bwd(): incompatible function arguments. The following argument types are supported:
[rank2]:     1. (arg0: torch.Tensor, arg1: torch.Tensor, arg2: Optional[torch.Tensor], arg3: torch.Tensor, arg4: Optional[torch.Tensor], arg5: bool) -> list[torch.Tensor]

[rank2]: Invoked with: tensor([[[ 8.0754e-01,  4.5564e-01,  3.5621e-01,  ...,  6

# 解决方案
1. mamba-ssm 2.2.4、 causal-conv1d 1.5.0 post8 (cu118 + torch2.3 + py3.10)
2. selective_scan_interface.py - 292行
```
    conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(
         x, conv1d_weight, conv1d_bias, True
       )
```

3. selective_scan_interface.py - 358行
```
    dx, dconv1d_weight, dconv1d_bias, *_ = causal_conv1d_cuda.causal_conv1d_bwd(
            x, conv1d_weight, conv1d_bias, dconv1d_out, dx, True
        )
```