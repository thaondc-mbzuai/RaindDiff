import os
import sys
import copy
import torch
import torch.nn as nn
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
# from timm.models.layers.helpers import to_2tuple
from timm.layers.helpers import to_2tuple

import einops
from einops import rearrange, reduce
from einops.layers.torch import Rearrange

Tokenwise_width = {
    'XS': [48, 56, 112, 220],
    'S': [48, 64, 168, 224],
    'l1': [48, 96, 192, 384],
    'l3': [64, 128, 320, 512],
}

Tokenwise_depth = {
    'XS': [3, 3, 6, 4],
    'S': [3, 3, 9, 6],
    'l1': [4, 3, 10, 5],
    'l3': [4, 4, 12, 6],
}

def stem(in_chs, out_chs):
    """
    Stem Layer that is implemented by two layers of conv.
    Output: sequence of layers with final shape of [B, C, H/4, W/4]
    """
    return nn.Sequential(
        nn.Conv2d(in_chs, out_chs // 2, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(out_chs // 2),
        nn.ReLU(),
        nn.Conv2d(out_chs // 2, out_chs, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(out_chs),
        nn.ReLU(), )


class Embedding(nn.Module):
    """
    Patch Embedding that is implemented by a layer of conv.
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H/stride, W/stride]
    """

    def __init__(self, patch_size=16, stride=16, padding=0,
                 in_chans=3, embed_dim=768, norm_layer=nn.BatchNorm2d):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x

class ConvEncoder(nn.Module):
    """
    Implementation of ConvEncoder with 3*3 and 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    Output: tensor with shape [B, C, H, W]
    """

    def __init__(self, dim, hidden_dim=64, kernel_size=3, drop_path=0., use_layer_scale=True):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(hidden_dim, dim, kernel_size=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale = nn.Parameter(torch.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.use_layer_scale:
            x = input + self.drop_path(self.layer_scale * x)
        else:
            x = input + self.drop_path(x)
        return x

class TokenwiseLocalRepresentation(nn.Module):
    """
    Local Representation module for Tokenwise that is implemented by 3*3 depth-wise and point-wise convolutions.
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H, W]
    """

    def __init__(self, dim, kernel_size=3, drop_path=0., use_layer_scale=True):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(dim, dim, kernel_size=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale = nn.Parameter(torch.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.use_layer_scale:
            x = input + self.drop_path(self.layer_scale * x)
        else:
            x = input + self.drop_path(x)
        return x

class Tokenkenwise(nn.Module):
    def __init__(self, in_dims=512, token_dim=256, num_heads=2): #token_dim = in_dims 
        super(Tokenkenwise, self).__init__()
        # Generate weight for Wquery、Wkey and Wvalue
        self.weight_q = nn.Linear(in_dims, token_dim*num_heads)
        self.weight_k = nn.Linear(in_dims, token_dim*num_heads)
        self.weight_v = nn.Linear(in_dims, token_dim*num_heads)

        # self.Proj = nn.Linear(token_dim * num_heads, token_dim * num_heads)
        self.weight_r = nn.Linear(token_dim*num_heads, token_dim)
        self.weight_alpha = nn.Parameter(torch.randn(token_dim*num_heads, 1))
        self.weight_beta = nn.Parameter(torch.randn(token_dim*num_heads, 1))

        self.scale_factor = token_dim ** -0.5

    def forward(self, x, y = None):

        if len(x.shape) > 3:
            B, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)

        query = self.weight_q(x)

        if y!= None:
            if len(y.shape) > 3:
                B, C, H, W = y.shape
                y = y.permute(0, 2, 3, 1).reshape(B, H * W, C)
            key = self.weight_k(y)
            value = self.weight_v(y)
        else:
            key = self.weight_k(x)
            value = self.weight_v(x)

        
        b, n, d = query.shape
        query = torch.nn.functional.normalize(query, dim=-1) #BxNxD
        key = torch.nn.functional.normalize(key, dim=-1) #BxNxD
        value = torch.nn.functional.normalize(key, dim=-1) #BxNxD
        
        # Caculate the global query
        alpha_weight = (query @ self.weight_alpha) * self.scale_factor #BxNx1
        alpha_weight = torch.nn.functional.normalize(alpha_weight, dim=1) # BxNx1

        global_query = torch.sum(alpha_weight * query, dim=1) # BxD
        global_query = torch.softmax(global_query, dim = -1) # BxD

        # Model the interaction between global query vector and the key vector
        repeat_global_query = einops.repeat(global_query, 'b d -> b copy d', copy = n)
        p = repeat_global_query * key

        beta_weight = (p @ self.weight_beta) * self.scale_factor
        beta_weight = torch.nn.functional.normalize(beta_weight, dim=1)

        global_key = torch.sum(beta_weight * p, dim=1)
        global_key = torch.softmax(global_key, dim = -1)

        # key-value
        key_value_interaction = torch.einsum('b j, b n j -> b n j', global_key, value)
        # key_value_interaction = self.Proj(key_value_interaction) + query #BxNxD
        key_value_interaction = key_value_interaction + query #BxNxD
            
        output = self.weight_r(key_value_interaction) #BxNxD
        
        return output

class Mlp(nn.Module):
    """
    Implementation of MLP layer with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    Output: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm1 = nn.BatchNorm2d(in_features)
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.norm1(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class TokenwiseEncoder(nn.Module):
    """
    TokenwiseEncoder Encoder Block for Tokenwise Attention. It consists of (1) Local representation module, (2) EfficientAdditiveAttention, and (3) MLP block.
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H, W]
    """

    def __init__(self, dim, mlp_ratio=4.,
                 act_layer=nn.GELU,
                 drop=0., drop_path=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5):

        super().__init__()

        self.local_representation = TokenwiseLocalRepresentation(dim=dim, kernel_size=3, drop_path=0.,
                                                                   use_layer_scale=True)
        self.attn = Tokenkenwise(in_dims=dim, token_dim=dim, num_heads=1)
        self.linear = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True)

    def forward(self, x, y = None):
        x = self.local_representation(x)
        B, C, H, W = x.shape
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1 * self.attn(x.permute(0, 2, 3, 1).reshape(B, H * W, C), y).reshape(B, H, W, C).permute(
                    0, 3, 1, 2))
            x = x + self.drop_path(self.layer_scale_2 * self.linear(x))

        else:
            x = x + self.drop_path(
                self.attn(x.permute(0, 2, 3, 1).reshape(B, H * W, C), y).reshape(B, H, W, C).permute(0, 3, 1, 2))
            x = x + self.drop_path(self.linear(x))
        return x


def TokenwiseStage(dim, layers=2, mlp_ratio=4.,
          act_layer=nn.GELU,
          drop_rate=.0, drop_path_rate=0.,
          use_layer_scale=True, layer_scale_init_value=1e-5, vit_num=1):
    """
    Implementation of each Tokenwise stages. Here, TokenwiseEncoder used as the last block in all stages, while ConvEncoder used in the rest of the blocks.
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H, W]
    """
    blocks = []

    for block_idx in range(layers):
        if layers - block_idx <= vit_num:
            blocks.append(TokenwiseEncoder(
                dim, mlp_ratio=mlp_ratio,
                act_layer=act_layer, drop_path=drop_path_rate,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value))

        else:
            blocks.append(ConvEncoder(dim=dim, hidden_dim=int(mlp_ratio * dim), kernel_size=3))

    blocks = nn.Sequential(*blocks)
    return blocks
