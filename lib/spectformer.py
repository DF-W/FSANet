import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import pytorch_wavelets as pw
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.act(x + self.dwconv(x, H, W))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class DWT(nn.Module):
    def __init__(self, in_planes):
        super(DWT, self).__init__()
        self.dwt = pw.DWTForward(J=1, wave='haar', mode='zero')
        self.conv = nn.Conv2d(in_planes*3, in_planes, kernel_size=3, padding=1, groups=in_planes)
        
    def forward(self, x):
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
            
        lf, hf = self.dwt(x)

        hf = hf[0]
        hf = hf.reshape(B, -1, H//2, W//2)
        hf = self.conv(hf)

        return lf, hf
    
class IDWT(nn.Module):
    def __init__(self, in_planes):
        super(IDWT, self).__init__()
        self.idwt = pw.DWTInverse(wave='haar', mode='zero')     
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=2, padding=1, groups=in_planes)
        self.conv2 = nn.Conv2d(in_planes, in_planes*3, kernel_size=3, stride=2, padding=1, groups=in_planes)

        
    def forward(self, x_lf, x_hf):
        B, N, _ = x_lf.shape
        H = W = int(math.sqrt(N))
        
        lf = x_lf.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        hf = x_hf.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        
        lf = self.conv1(lf)
        
        hf = self.conv2(hf).reshape(B, -1, 3, H//2, W//2)
        hf = [hf, ]
    
        out = self.idwt((lf, hf))
        out = out.flatten(2).transpose(1, 2)
        return out
    
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        self.q = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 0:
            self.dwt = DWT(in_planes=dim)
            self.idwt = IDWT(in_planes=dim)
            
            if sr_ratio==4:
                self.sr1 = nn.Conv2d(dim, dim, kernel_size=7, stride=4, groups=dim, padding=3, bias=True)
                self.norm1 = nn.BatchNorm2d(dim)
                self.sr2 = nn.Conv2d(dim, dim, kernel_size=7, stride=4, groups=dim, padding=3, bias=True)
                self.norm2 = nn.BatchNorm2d(dim)
            if sr_ratio==2:
                self.sr1 = nn.Conv2d(dim, dim, kernel_size=5, stride=2, groups=dim, padding=2, bias=True)
                self.norm1 = nn.BatchNorm2d(dim)
                self.sr2 = nn.Conv2d(dim, dim, kernel_size=5, stride=2, groups=dim, padding=2, bias=True)
                self.norm2 = nn.BatchNorm2d(dim)
                
            self.kv1 = nn.Linear(dim, dim*2, bias=qkv_bias)
            self.kv2 = nn.Linear(dim, dim*2, bias=qkv_bias)
        else:
            self.kv = nn.Linear(dim, dim*2, bias=qkv_bias)
            
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W, aux_feature):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        if self.sr_ratio > 0:
            x_lf, x_hf = self.dwt(x)
            
            if self.sr_ratio > 1:
                x_lf = self.norm1(self.sr1(x_lf))
                x_hf = self.norm2(self.sr2(x_hf))
            
            x_lf = x_lf.reshape(B, C, -1).permute(0, 2, 1)
            x_hf = x_hf.reshape(B, C, -1).permute(0, 2, 1)
            kv_lf = self.kv1(x_lf).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            kv_hf = self.kv2(x_hf).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            k_lf, v_lf = kv_lf[0], kv_lf[1] #B head N C
            k_hf, v_hf = kv_hf[0], kv_hf[1]
            attn1 = (q @ k_lf.transpose(-2, -1)) * self.scale
            attn1 = attn1.softmax(dim=-1)
            attn1 = self.attn_drop(attn1)
            x_lf = (attn1 @ v_lf).transpose(1, 2).reshape(B, N, C)
            attn2 = (q @ k_hf.transpose(-2, -1)) * self.scale
            attn2 = attn2.softmax(dim=-1)
            attn2 = self.attn_drop(attn2)
            x_hf = (attn2 @ v_hf).transpose(1, 2).reshape(B, N, C)

            # x = torch.cat([x1,x2], dim=-1)
            x = self.idwt(x_lf, x_hf)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        if aux_feature:
            return x, x_lf, x_hf
        else:
            return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W, aux_feature):
        x_a = self.attn(self.norm1(x), H, W, aux_feature)
        if aux_feature:
            x_, lf, hf = x_a[0], x_a[1], x_a[2]
        else:
            x_ = x_a
        x = x + self.drop_path(x_)
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        if aux_feature:
            return x, lf, hf
        else:
            return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size//2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):  
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        
        return x, H, W

class ConvStem(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvStem, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 7, 2, padding=3, bias=False), 
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(True)
            )
        self.proj = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.norm = nn.LayerNorm(out_channels)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
                
    def forward(self, x):
        x = self.conv(x)
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

class SpectTransformer(nn.Module):
    def __init__(self, img_size=224, in_chans=3, num_classes=10, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], num_stages=4):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            if i ==0:
                patch_embed = ConvStem(in_channels=in_chans, out_channels=embed_dims[0])
            else:
                patch_embed = OverlapPatchEmbed(
                                            patch_size=3,
                                            stride=2,
                                            in_chans=embed_dims[i - 1],
                                            embed_dim=embed_dims[i])

            block = nn.ModuleList([Block(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i])
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # classification head
        # self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        out_x = []
        out_lf = []
        out_hf = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for j, blk in enumerate(block):
                aux_feature = (j == self.depths[i]-1 and i != self.num_stages-1)
                x = blk(x, H, W, aux_feature)
                if aux_feature:
                    x, lf, hf = x[0], x[1], x[2]
                    lf = lf.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                    out_lf.append(lf)
                    hf = hf.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                    out_hf.append(hf)
            x = norm(x)
            # if i != self.num_stages - 1:
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            out_x.append(x)
        return out_x, out_lf, out_hf
        # return x.mean(dim=1)

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)

        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict


@register_model
def spectformer(pretrained=False, **kwargs):
    model = SpectTransformer(
        embed_dims=[64, 128, 256, 512], num_heads=[2, 4, 8, 16], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 24, 2], sr_ratios=[4, 2, 1, 0], **kwargs)
    model.default_cfg = _cfg()

    return model


