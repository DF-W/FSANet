import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from einops import rearrange, repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from timm.models.layers import DropPath
from ptflops import get_model_complexity_info
from lib.spectformer import spectformer
from lib.spectformer_1 import spectformer_1


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class DepthWiseConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding, 
                      stride=stride, dilation=dilation, groups=32)
        self.norm_layer = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.norm_layer(self.conv(x)))
    
class UpsampleModule(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, padding=1, scale_factor=2):
        super(UpsampleModule, self).__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.conv = DepthWiseConv2d(in_planes=in_planes, out_planes=out_planes, kernel_size=kernel_size, padding=padding)
        
    def forward(self, x):
        x = self.up(x) 
        x = self.conv(x)
       
        return x

    
class Projector(nn.Module):
    def __init__(self):
        super(Projector, self).__init__()
        
        self.up_x_4_1 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up_x_3_1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up_x_2_1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.conv1_x = BasicConv2d(64+128+256+512, 256, 1)
        self.conv2_x = nn.Conv2d(256, 256, 1)

        self.up_lf_3_1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up_lf_2_1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.conv1_lf = BasicConv2d(64+128+256, 256, 1)
        self.conv2_lf = nn.Conv2d(256, 256, 1)
        
        self.up_hf_3_1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up_hf_2_1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.conv1_hf = BasicConv2d(64+128+256, 256, 1)
        self.conv2_hf = nn.Conv2d(256, 256, 1)

    def forward(self, x, lf, hf):
        x4 = self.up_x_4_1(x[3])
        x3 = self.up_x_3_1(x[2])
        x2 = self.up_x_2_1(x[1])
        x1 = x[0]
        x = self.conv2_x(self.conv1_x(torch.cat([x1, x2, x3, x4], dim=1)))

        lf3 = self.up_lf_3_1(lf[2])
        lf2 = self.up_lf_2_1(lf[1])
        lf1 = lf[0]
        lf = self.conv2_lf(self.conv1_lf(torch.cat([lf1, lf2, lf3], dim=1)))

        hf3 = self.up_hf_3_1(hf[2])
        hf2 = self.up_hf_2_1(hf[1])
        hf1 = hf[0]
        hf = self.conv2_hf(self.conv1_hf(torch.cat([hf1, hf2, hf3], dim=1)))

        return x, lf, hf
    

class ProjectorWise(nn.Module):
    def __init__(self):
        super(ProjectorWise, self).__init__()
    
        self.up_x_4_1 = UpsampleModule(in_planes=512, out_planes=512, scale_factor=8)
        self.up_x_3_1 = UpsampleModule(in_planes=256, out_planes=256, scale_factor=4)
        self.up_x_2_1 = UpsampleModule(in_planes=128, out_planes=128, scale_factor=2)
        self.conv1_x = DepthWiseConv2d(64+128+256+512, 256, 3, padding=1)
        self.conv2_x = nn.Conv2d(256, 256, 1)

        self.up_lf_3_1 = UpsampleModule(in_planes=256, out_planes=256, scale_factor=4)
        self.up_lf_2_1 = UpsampleModule(in_planes=128, out_planes=128, scale_factor=2)
        
        self.conv1_lf = DepthWiseConv2d(64+128+256, 256, 3, padding=1)
        self.conv2_lf = nn.Conv2d(256, 256, 1)
        
        self.up_hf_3_1 = UpsampleModule(in_planes=256, out_planes=256, scale_factor=4)
        self.up_hf_2_1 = UpsampleModule(in_planes=128, out_planes=128, scale_factor=2)
        
        self.conv1_hf = DepthWiseConv2d(64+128+256, 256, 3, padding=1)
        self.conv2_hf = nn.Conv2d(256, 256, 1)

    def forward(self, x, lf, hf):
        x4 = self.up_x_4_1(x[3])
        x3 = self.up_x_3_1(x[2])
        x2 = self.up_x_2_1(x[1])
        x1 = x[0]
        x = self.conv2_x(self.conv1_x(torch.cat([x1, x2, x3, x4], dim=1)))

        lf3 = self.up_lf_3_1(lf[2])
        lf2 = self.up_lf_2_1(lf[1])
        lf1 = lf[0]
        lf = self.conv2_lf(self.conv1_lf(torch.cat([lf1, lf2, lf3], dim=1)))

        hf3 = self.up_hf_3_1(hf[2])
        hf2 = self.up_hf_2_1(hf[1])
        hf1 = hf[0]
        hf = self.conv2_hf(self.conv1_hf(torch.cat([hf1, hf2, hf3], dim=1)))

        return x, lf, hf
   

class SFI(nn.Module):
    def __init__(self, in_planes):
        super(SFI, self).__init__()
        
        self.conv1 = nn.Conv2d(in_planes*2, in_planes, 3, padding=1) 
        
        self.conv2 = nn.Conv2d(2, 2, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()     
         
        self.conv3_1 = nn.Conv2d(in_planes, in_planes, 1)
        self.conv3_2 = nn.Conv2d(in_planes, in_planes, 1)
        
        self.conv4_1 = nn.Conv2d(in_planes, in_planes, 1)
        self.conv4_2 = nn.Conv2d(in_planes, in_planes, 1)
 
    def forward(self, x_, lf, hf): 
        x = torch.cat([lf, hf], dim=1)
        x = self.conv1(x)
        
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv2(x)
        sa = self.sigmoid(x)
        
        att_lf = self.conv3_1(lf * sa[:,0,:,:].unsqueeze(1))
        att_hf = self.conv3_2(hf * sa[:,1,:,:].unsqueeze(1))
        att_x = self.conv4_2(x_ * self.conv4_1(att_lf + att_hf)) 
        return x_ + att_x, lf + att_lf, hf + att_hf
    

class SS2D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        # d_state="auto", # 20240109
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj1 = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.in_proj2 = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.in_proj3 = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=4, inner)
        del self.dt_projs
        
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True) # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True) # (K=4, D, N)

        # self.selective_scan = selective_scan_fn
        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj1 = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.out_proj2 = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn
        
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    # an alternative to forward_corev1
    def forward_corev1(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn_v1

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor, **kwargs):
        B, H, W, C = x1.shape

        x1 = self.in_proj1(x1)
        x2 = self.in_proj2(x2)
        x3 = self.in_proj3(x3)
        # x, z = xz.chunk(2, dim=-1) # (b, h, w, d)

        x1 = x1.permute(0, 3, 1, 2).contiguous()
        x1 = self.act(self.conv2d(x1)) # (b, d, h, w)
        y1, y2, y3, y4 = self.forward_core(x1)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        out1 = y * F.silu(x2)
        out2 = y * F.silu(x3)
        out1 = self.out_proj1(out1)
        out2 = self.out_proj2(out2)
        if self.dropout is not None:
            out = self.dropout(out)
        return out1, out2

class SSIM(nn.Module):
    def __init__(self, embed_dim):
        super(SSIM, self).__init__()
        self.norm_x = nn.LayerNorm(embed_dim, eps=1e-6)
        self.norm_lf = nn.LayerNorm(embed_dim, eps=1e-6)
        self.norm_hf = nn.LayerNorm(embed_dim, eps=1e-6)

        self.conv1 = nn.Conv2d(embed_dim*2, embed_dim, 1)
        self.conv2 = nn.Conv2d(embed_dim*2, embed_dim, 1)
        self.conv3 = nn.Conv2d(embed_dim*2, embed_dim, 1)

        self.atten1 = SS2D(d_model=embed_dim)
        self.atten2 = SS2D(d_model=embed_dim)
        self.droppath1 = DropPath(drop_prob=0.2)
        self.droppath2 = DropPath(drop_prob=0.2)
        self.droppath3 = DropPath(drop_prob=0.2)

    def forward(self, x, lf, hf): 
        x_ = x.permute(0, 2, 3, 1)
        lf_ = lf.permute(0, 2, 3, 1)
        hf_ = hf.permute(0, 2, 3, 1)

        x_n = self.norm_x(x_)
        lf_n = self.norm_lf(lf_)
        hf_n = self.norm_hf(hf_)

        x_lf = self.conv1(torch.cat([x_n, lf_n], dim=-1).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x_hf = self.conv2(torch.cat([x_n, hf_n], dim=-1).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x_a1, lf_a = self.atten1(x_lf, x_n, lf_n)
        x_a2, hf_a = self.atten2(x_hf, x_n, hf_n)

        out_x = self.droppath1(self.conv3(torch.cat([x_a1, x_a2], dim=-1).permute(0, 3, 1, 2))) + x
        out_lf = self.droppath2(lf_a.permute(0, 3, 1, 2)) + lf
        out_hf = self.droppath3(hf_a.permute(0, 3, 1, 2)) + hf

        return out_x, out_lf, out_hf
    
class FSANet(nn.Module):
    def __init__(self):
        super(FSANet, self).__init__()

        self.backbone = spectformer()  

        self.conv_x_4 = DepthWiseConv2d(512, 512, 3, padding=1)
        self.up_x_4_3 = UpsampleModule(in_planes=512, out_planes=256)
        
        self.conv_x_3 = DepthWiseConv2d(512, 256, 3, padding=1)
        self.up_x_3_2 = UpsampleModule(in_planes=256, out_planes=128)

        self.conv_x_2 = DepthWiseConv2d(256, 128, 3, padding=1)
        self.up_x_2_1 = UpsampleModule(in_planes=128, out_planes=64)
        
        self.conv_x_1 = DepthWiseConv2d(128, 64, 3, padding=1)
        self.out_x = nn.Conv2d(64, 1, 1)
        

        self.conv_lf_3 = DepthWiseConv2d(256, 256, 3, padding=1)
        self.up_lf_3_2 = UpsampleModule(in_planes=256, out_planes=128)

        self.conv_lf_2 = DepthWiseConv2d(128, 128, 3, padding=1)
        self.conv_lf_2_1 = DepthWiseConv2d(256, 128, 3, padding=1)
        self.up_lf_2_1 = UpsampleModule(in_planes=128, out_planes=64)
        
        self.conv_lf_1 = DepthWiseConv2d(64, 64, 3, padding=1)
        self.conv_lf_1_0 = DepthWiseConv2d(128, 64, 3, padding=1)
        
        self.out_lf = nn.Conv2d(64, 1, 1)
        
        self.out_lf_2 = nn.Conv2d(128, 1, 1)
        self.out_lf_3 = nn.Conv2d(256, 1, 1)
        

        self.conv_hf_3 = DepthWiseConv2d(256, 256, 3, padding=1)
        self.up_hf_3_2 = UpsampleModule(in_planes=256, out_planes=128)

        self.conv_hf_2 = DepthWiseConv2d(128, 128, 3, padding=1)
        self.conv_hf_2_1 = DepthWiseConv2d(256, 128, 3, padding=1)
        self.up_hf_2_1 = UpsampleModule(in_planes=128, out_planes=64)
        
        self.conv_hf_1 = DepthWiseConv2d(64, 64, 3, padding=1)
        self.conv_hf_1_0 = DepthWiseConv2d(128, 64, 3, padding=1)
        
        self.out_hf = nn.Conv2d(64, 1, 1)
        self.out_hf_2 = nn.Conv2d(128, 1, 1)
        self.out_hf_3 = nn.Conv2d(256, 1, 1)

                
        self.sfi3 = SSIM(embed_dim=256)
        self.sfi2 = SSIM(embed_dim=128)
        self.sfi1 = SSIM(embed_dim=64)

        self.project = Projector()
        
    def forward(self, x):

        # backbone
        x, lf, hf = self.backbone(x)
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]
        
        lf1 = lf[0]
        lf2 = lf[1]
        lf3 = lf[2]
        
        hf1 = hf[0]
        hf2 = hf[1]
        hf3 = hf[2]
        
        x_4 = self.conv_x_4(x4)
        x_4_3 = self.up_x_4_3(x_4)
        
        lf_3 = self.conv_lf_3(lf3)
        hf_3 = self.conv_hf_3(hf3)
        x_3 = self.conv_x_3(torch.cat([x3, x_4_3], dim=1))
        x_3, lf_3, hf_3 = self.sfi3(x_3, lf_3, hf_3)
        
        lf_3_2 = self.up_lf_3_2(lf_3)
        hf_3_2 = self.up_hf_3_2(hf_3)
        x_3_2 = self.up_x_3_2(x_3)
        
        lf_2 = self.conv_lf_2(lf2)
        lf_2 = self.conv_lf_2_1(torch.cat([lf_2, lf_3_2], dim=1))
        hf_2 = self.conv_hf_2(hf2)     
        hf_2 = self.conv_hf_2_1(torch.cat([hf_2, hf_3_2], dim=1))
        x_2 = self.conv_x_2(torch.cat([x2, x_3_2], dim=1))   
        x_2, lf_2, hf_2 = self.sfi2(x_2, lf_2, hf_2)
        
        lf_2_1 = self.up_lf_2_1(lf_2)
        hf_2_1 = self.up_hf_2_1(hf_2)        
        x_2_1 = self.up_x_2_1(x_2)
        
        lf_1 = self.conv_lf_1(lf1)
        lf_1 = self.conv_lf_1_0(torch.cat([lf_1, lf_2_1], dim=1))
        hf_1 = self.conv_hf_1(hf1)
        hf_1 = self.conv_hf_1_0(torch.cat([hf_1, hf_2_1], dim=1))  
        x_1 = self.conv_x_1(torch.cat([x1, x_2_1], dim=1))      
        x_1, lf_1, hf_1 = self.sfi1(x_1, lf_1, hf_1)
        
        proj = self.project((x_1, x_2, x_3, x_4), (lf_1, lf_2, lf_3), (hf_1, hf_2, hf_3))

        lf_0 = self.out_lf(lf_1)
        lf_0_2 = self.out_lf_2(lf_2)
        lf_0_3 = self.out_lf_3(lf_3)
        
        hf_0 = self.out_hf(hf_1)  
        hf_0_2 = self.out_hf_2(hf_2)
        hf_0_3 = self.out_hf_3(hf_3)     
        
        x_0 = self.out_x(x_1)
        
        x_0 = F.interpolate(x_0, scale_factor=4, mode='bilinear', align_corners=True)
          
        return x_0, (lf_0, lf_0_2, lf_0_3),  (hf_0, hf_0_2, hf_0_3), proj
