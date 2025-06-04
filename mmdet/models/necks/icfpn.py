# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from torch import Tensor
from einops import rearrange

from mmdet.registry import MODELS
from mmdet.utils import ConfigType, MultiConfig, OptConfigType


@MODELS.register_module()
class ICFPN(BaseModule):
    """
    Information Compensation Feature Pyramid Network with Local Self-Attention and Global Cross-Attention
    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        num_outs: int,
        start_level: int = 0,
        end_level: int = -1,
        heads: int = 4,
        add_extra_convs: Union[bool, str] = False,
        relu_before_extra_convs: bool = False,
        no_norm_on_lateral: bool = False,
        conv_cfg: OptConfigType = None,
        norm_cfg: OptConfigType = None,
        act_cfg: OptConfigType = None,
        upsample_cfg: ConfigType = dict(mode='nearest'),
        init_cfg: MultiConfig = dict(
            type='Xavier', layer='Conv2d', distribution='uniform')
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level is not the last level, no extra level is allowed
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins
            assert num_outs == end_level - start_level + 1
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'

        self.lateral_convs = nn.ModuleList()
        self.local_region_self_attention = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.global_region_cross_attention = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            lrsa = LRSA(dim=out_channels, qk_dim=out_channels, heads=heads)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.local_region_self_attention.append(lrsa)
            self.fpn_convs.append(fpn_conv)
        used_backbone_levels = len(self.local_region_self_attention)
        for i in range(used_backbone_levels - 1, 0, -1):
            grca = GRCA(dim=out_channels, qk_dim=out_channels, heads=heads)
            self.global_region_cross_attention.append(grca)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)


    def forward(self, inputs: Tuple[Tensor]) -> tuple:
        """Forward function.

        Args:
            inputs (tuple[Tensor]): Features from the upstream network, each
                is a 4D-tensor.

        Returns:
            tuple: Feature maps, each is a 4D-tensor.
        """
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            self.local_region_self_attention[i](self.lateral_convs[i](inputs[i + self.start_level]))
            for i in range(len(self.local_region_self_attention))
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                # fix runtime error of "+=" inplace operation in PyTorch 1.10
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] = self.global_region_cross_attention[i - 1](
                    laterals[i - 1], F.interpolate(laterals[i], size=prev_shape, **self.upsample_cfg))

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)


def patch_divide(x, step, ps):
    """
    Divide image into patches with optional padding, using F.unfold.
    Args:
        x (Tensor): Input tensor of shape (b, c, h, w)
        step (int): Stride for patch extraction
        ps (int): Patch size
    Returns:
        patches (Tensor): Tensor of shape (b, n, c, ps, ps)
        nh (int): Number of patches along height
        nw (int): Number of patches along width
        pad_info (tuple): Padding added (pad_left, pad_right, pad_top, pad_bottom)
    """
    b, c, h, w = x.size()

    # 计算需要 padding 的大小
    nh = math.ceil((h - ps) / step) + 1
    nw = math.ceil((w - ps) / step) + 1
    pad_h = max((nh - 1) * step + ps - h, 0)
    pad_w = max((nw - 1) * step + ps - w, 0)

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    # 边缘补齐
    x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))

    # 使用 unfold 提取 patch
    unfold = F.unfold(x, kernel_size=ps, stride=step)
    n_patches = unfold.size(-1)
    patches = unfold.transpose(1, 2).reshape(b, n_patches, c, ps, ps)

    return patches, nh, nw, (pad_left, pad_right, pad_top, pad_bottom)


def patch_reverse(patches, nh, nw, ps, img_shape, pad_info):
    """
    Simply stitch patches back to image (no overlap average), and crop to original size.
    Args:
        patches (Tensor): Shape (b, n, c, ps, ps)
        nh (int): Patches along height
        nw (int): Patches along width
        ps (int): Patch size
        img_shape (tuple): Original image size (h, w)
        pad_info (tuple): (pad_left, pad_right, pad_top, pad_bottom)
    Returns:
        x_reconstructed (Tensor): Reconstructed image of shape (b, c, h, w)
    """
    b, n, c, _, _ = patches.shape
    assert n == nh * nw, "Patch count mismatch"

    # 重排为网格格式 (b, nh, nw, c, ps, ps)
    patches = patches.view(b, nh, nw, c, ps, ps)
    patches = patches.permute(0, 3, 1, 4, 2, 5).contiguous()  # (b, c, nh, ps, nw, ps)
    x_recon = patches.view(b, c, nh * ps, nw * ps)  # (b, c, H_pad, W_pad)

    # 裁剪回原图大小
    pad_left, pad_right, pad_top, pad_bottom = pad_info
    b, c, h, w = img_shape
    x_recon = x_recon[:, :, pad_top:pad_top + h, pad_left:pad_left + w]

    return x_recon


class SelfAttention(BaseModule):
    """Attention module.
    Args:
        dim (int): Base channels.
        heads (int): Head numbers.
        qk_dim (int): Channels of query and key.
    """

    def __init__(self, dim, heads, qk_dim, max_len=512):
        super().__init__()
        self.heads = heads
        self.dim = dim
        self.qk_dim = qk_dim
        self.scale = qk_dim ** -0.5
        self.to_q = nn.Linear(dim, qk_dim, bias=False)
        self.to_k = nn.Linear(dim, qk_dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

        # Learnable positional encoding
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, dim))


    def forward(self, x):
        b, n, _ = x.shape
        pos = self.pos_emb[:, :n, :]  # truncate to input sequence length
        x = x + pos  # add positional encoding
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v)  # scaled_dot_product_attention 需要PyTorch2.0之后版本
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.proj(out)


class CrossAttention(BaseModule):
    """Attention module.
    Args:
        dim (int): Base channels.
        heads (int): Head numbers.
        qk_dim (int): Channels of query and key.
    """

    def __init__(self, dim, heads, qk_dim, max_len=512):
        super().__init__()
        self.heads = heads
        self.dim = dim
        self.qk_dim = qk_dim
        self.scale = qk_dim ** -0.5
        self.to_q1 = nn.Linear(dim, qk_dim, bias=False)
        self.to_k1 = nn.Linear(dim, qk_dim, bias=False)
        self.to_v1 = nn.Linear(dim, dim, bias=False)
        self.to_q2 = nn.Linear(dim, qk_dim, bias=False)
        self.to_k2 = nn.Linear(dim, qk_dim, bias=False)
        self.to_v2 = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

        # Learnable positional encoding
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, dim))

    def forward(self, x1, x2):
        b, n, _ = x1.shape
        pos = self.pos_emb[:, :n, :]  # truncate to input sequence length
        x1 = x1 + pos  # add positional encoding
        x2 = x2 + pos  # add positional encoding
        q1, k1, v1 = self.to_q1(x1), self.to_k1(x1), self.to_v1(x1)
        q2, k2, v2 = self.to_q2(x2), self.to_k2(x2), self.to_v2(x2)
        q1, k1, v1 = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q1, k1, v1))
        q2, k2, v2 = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q2, k2, v2))
        out1 = F.scaled_dot_product_attention(q2, k1, v1)  # scaled_dot_product_attention 需要PyTorch2.0之后版本
        out2 = F.scaled_dot_product_attention(q1, k2, v2)  # scaled_dot_product_attention 需要PyTorch2.0之后版本
        out = out1 + out2
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.proj(out)


class LRSA(BaseModule):
    """Attention module.
    Args:
        dim (int): Base channels.
        num (int): Number of blocks.
        qk_dim (int): Channels of query and key in Attention.
        mlp_dim (int): Channels of hidden mlp in Mlp.
        heads (int): Head numbers of Attention.
    """

    def __init__(self, dim, qk_dim, heads=1, patch_size=16):
        super().__init__()
        self.ps = patch_size
        self.attn = nn.Sequential(SelfAttention(dim, heads, qk_dim),
                                  nn.LayerNorm(dim))
        self.alpha = nn.Parameter(torch.tensor([1.0, 0.0]))

    def forward(self, x):
        step = self.ps
        crop_x, nh, nw, pad_info = patch_divide(x, step, self.ps)  # (b, n, c, ps, ps)
        b, n, c, ph, pw = crop_x.shape
        crop_x = rearrange(crop_x, 'b n c h w -> (b n) (h w) c')
        crop_x = self.attn(crop_x)
        crop_x = rearrange(crop_x, '(b n) (h w) c  -> b n c h w', n=n, w=pw)
        weights = F.sigmoid(self.alpha)
        x = weights[0] * patch_reverse(crop_x, nh, nw, self.ps, x.shape, pad_info) + weights[1] * x
        return x


class GRCA(BaseModule):
    """Attention module.
    Args:
        dim (int): Base channels.
        num (int): Number of blocks.
        qk_dim (int): Channels of query and key in Attention.
        mlp_dim (int): Channels of hidden mlp in Mlp.
        heads (int): Head numbers of Attention.
    """

    def __init__(self, dim, qk_dim, heads=1, patch_size=16):
        super().__init__()
        self.ps = patch_size
        self.attn = CrossAttention(dim, heads, qk_dim)
        self.layer_norm = nn.LayerNorm(dim)
        self.alpha = nn.Parameter(torch.tensor([1.0, 0.0, 0.0]))

    def forward(self, x1, x2):
        step = self.ps
        crop_x1, nh, nw, pad_info = patch_divide(x1, step, self.ps)  # (b, n, c, ps, ps)
        crop_x2, nh, nw, pad_info  = patch_divide(x2, step, self.ps)  # (b, n, c, ps, ps)
        b, n, c, ph, pw = crop_x1.shape
        crop_x1 = rearrange(crop_x1, 'b n c h w -> (b n) (h w) c')
        crop_x2 = rearrange(crop_x2, 'b n c h w -> (b n) (h w) c')
        crop_x = self.attn(crop_x1, crop_x2)
        crop_x = self.layer_norm(crop_x)
        crop_x = rearrange(crop_x, '(b n) (h w) c  -> b n c h w', n=n, w=pw)
        weights = F.sigmoid(self.alpha)
        x = weights[0] * patch_reverse(crop_x, nh, nw, self.ps, x1.shape, pad_info) + weights[1] * x1 + weights[2] * x2
        return x


if __name__ == "__main__":
    # 输入参数配置
    batch_size = 8  # Batch size
    channels = 512  # 输入通道数
    height = 100  # 高度
    width = 168  # 宽度
    ps = 16  # Patch size
    qk_dim = channels  # Query-Key维度
    out_dim = 256  # 输出维度
    heads = 4  # Attention头数

    # 创建一个输入张量，形状为 (batch_size, channels, height, width)
    x = torch.randn(batch_size, channels, height, width)
    # 初始化 LRSA 模块
    lrsa = LRSA(dim=channels, qk_dim=qk_dim, out_dim=out_dim, heads=heads, patch_size=ps)
    print(lrsa)
    # 前向传播，传入输入张量和patch大小
    output = lrsa(x)
    # 打印输入和输出张量的形状
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # 初始化 GRCA 模块
    grca = GRCA(dim=channels, qk_dim=qk_dim, heads=heads, patch_size=ps)
    print(grca)
    # 前向传播，传入输入张量和patch大小
    output = grca(x, x)
    # 打印输入和输出张量的形状
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")



# # Copyright (c) OpenMMLab. All rights reserved.
# from typing import List, Tuple, Union
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math
# from mmcv.cnn import ConvModule
# from mmengine.model import BaseModule
# from torch import Tensor
# from einops import rearrange
#
# from mmdet.registry import MODELS
# from mmdet.utils import ConfigType, MultiConfig, OptConfigType
#
#
# @MODELS.register_module()
# class ICFPN(BaseModule):
#     """
#     Information Compensation Feature Pyramid Network with Local Self-Attention and Global Cross-Attention
#     """
#
#     def __init__(
#         self,
#         in_channels: List[int],
#         out_channels: int,
#         num_outs: int,
#         start_level: int = 0,
#         end_level: int = -1,
#         heads: int = 4,
#         blocks_num: int = 4,
#         add_extra_convs: Union[bool, str] = False,
#         relu_before_extra_convs: bool = False,
#         no_norm_on_lateral: bool = False,
#         conv_cfg: OptConfigType = None,
#         norm_cfg: OptConfigType = None,
#         act_cfg: OptConfigType = None,
#         upsample_cfg: ConfigType = dict(mode='nearest'),
#         init_cfg: MultiConfig = dict(
#             type='Xavier', layer='Conv2d', distribution='uniform')
#     ) -> None:
#         super().__init__(init_cfg=init_cfg)
#         assert isinstance(in_channels, list)
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.num_ins = len(in_channels)
#         self.num_outs = num_outs
#         self.relu_before_extra_convs = relu_before_extra_convs
#         self.no_norm_on_lateral = no_norm_on_lateral
#         self.fp16_enabled = False
#         self.upsample_cfg = upsample_cfg.copy()
#         self.blocks_num = blocks_num
#
#         if end_level == -1 or end_level == self.num_ins - 1:
#             self.backbone_end_level = self.num_ins
#             assert num_outs >= self.num_ins - start_level
#         else:
#             # if end_level is not the last level, no extra level is allowed
#             self.backbone_end_level = end_level + 1
#             assert end_level < self.num_ins
#             assert num_outs == end_level - start_level + 1
#         self.start_level = start_level
#         self.end_level = end_level
#         self.add_extra_convs = add_extra_convs
#         assert isinstance(add_extra_convs, (str, bool))
#         if isinstance(add_extra_convs, str):
#             # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
#             assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
#         elif add_extra_convs:  # True
#             self.add_extra_convs = 'on_input'
#
#         self.lateral_convs = nn.ModuleList()
#         self.local_region_self_attention = nn.ModuleList()
#         self.fpn_convs = nn.ModuleList()
#         self.global_region_cross_attention = nn.ModuleList()
#
#         for i in range(self.start_level, self.backbone_end_level):
#             l_conv = ConvModule(
#                 in_channels[i],
#                 out_channels,
#                 1,
#                 conv_cfg=conv_cfg,
#                 norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
#                 act_cfg=act_cfg,
#                 inplace=False)
#             lrsa = LRSA(dim=out_channels, qk_dim=out_channels, heads=heads, blocks_num=self.blocks_num)
#             fpn_conv = ConvModule(
#                 out_channels,
#                 out_channels,
#                 3,
#                 padding=1,
#                 conv_cfg=conv_cfg,
#                 norm_cfg=norm_cfg,
#                 act_cfg=act_cfg,
#                 inplace=False)
#
#             self.lateral_convs.append(l_conv)
#             self.local_region_self_attention.append(lrsa)
#             self.fpn_convs.append(fpn_conv)
#         used_backbone_levels = len(self.local_region_self_attention)
#         for i in range(used_backbone_levels - 1, 0, -1):
#             grca = GRCA(dim=out_channels, qk_dim=out_channels, heads=heads)
#             self.global_region_cross_attention.append(grca)
#
#         # add extra conv layers (e.g., RetinaNet)
#         extra_levels = num_outs - self.backbone_end_level + self.start_level
#         if self.add_extra_convs and extra_levels >= 1:
#             for i in range(extra_levels):
#                 if i == 0 and self.add_extra_convs == 'on_input':
#                     in_channels = self.in_channels[self.backbone_end_level - 1]
#                 else:
#                     in_channels = out_channels
#                 extra_fpn_conv = ConvModule(
#                     in_channels,
#                     out_channels,
#                     3,
#                     stride=2,
#                     padding=1,
#                     conv_cfg=conv_cfg,
#                     norm_cfg=norm_cfg,
#                     act_cfg=act_cfg,
#                     inplace=False)
#                 self.fpn_convs.append(extra_fpn_conv)
#
#
#
#
#     def forward(self, inputs: Tuple[Tensor]) -> tuple:
#         """Forward function.
#
#         Args:
#             inputs (tuple[Tensor]): Features from the upstream network, each
#                 is a 4D-tensor.
#
#         Returns:
#             tuple: Feature maps, each is a 4D-tensor.
#         """
#         assert len(inputs) == len(self.in_channels)
#
#         # build laterals
#         laterals = [
#             self.local_region_self_attention[i](self.lateral_convs[i](inputs[i + self.start_level]))
#             for i in range(len(self.local_region_self_attention))
#         ]
#
#         # build top-down path
#         used_backbone_levels = len(laterals)
#         for i in range(used_backbone_levels - 1, 0, -1):
#             # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
#             #  it cannot co-exist with `size` in `F.interpolate`.
#             if 'scale_factor' in self.upsample_cfg:
#                 # fix runtime error of "+=" inplace operation in PyTorch 1.10
#                 laterals[i - 1] = laterals[i - 1] + F.interpolate(
#                     laterals[i], **self.upsample_cfg)
#             else:
#                 prev_shape = laterals[i - 1].shape[2:]
#                 laterals[i - 1] = self.global_region_cross_attention[i - 1](
#                     laterals[i - 1], F.interpolate(laterals[i], size=prev_shape, **self.upsample_cfg))
#
#         # build outputs
#         # part 1: from original levels
#         outs = [
#             self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
#         ]
#         # part 2: add extra levels
#         if self.num_outs > len(outs):
#             # use max pool to get more levels on top of outputs
#             # (e.g., Faster R-CNN, Mask R-CNN)
#             if not self.add_extra_convs:
#                 for i in range(self.num_outs - used_backbone_levels):
#                     outs.append(F.max_pool2d(outs[-1], 1, stride=2))
#             # add conv layers on top of original feature maps (RetinaNet)
#             else:
#                 if self.add_extra_convs == 'on_input':
#                     extra_source = inputs[self.backbone_end_level - 1]
#                 elif self.add_extra_convs == 'on_lateral':
#                     extra_source = laterals[-1]
#                 elif self.add_extra_convs == 'on_output':
#                     extra_source = outs[-1]
#                 else:
#                     raise NotImplementedError
#                 outs.append(self.fpn_convs[used_backbone_levels](extra_source))
#                 for i in range(used_backbone_levels + 1, self.num_outs):
#                     if self.relu_before_extra_convs:
#                         outs.append(self.fpn_convs[i](F.relu(outs[-1])))
#                     else:
#                         outs.append(self.fpn_convs[i](outs[-1]))
#         return tuple(outs)
#
#
# def patch_divide(x, step, ps):
#     """
#     Divide image into patches with optional padding, using F.unfold.
#     Args:
#         x (Tensor): Input tensor of shape (b, c, h, w)
#         step (int): Stride for patch extraction
#         ps (int): Patch size
#     Returns:
#         patches (Tensor): Tensor of shape (b, n, c, ps, ps)
#         nh (int): Number of patches along height
#         nw (int): Number of patches along width
#         pad_info (tuple): Padding added (pad_left, pad_right, pad_top, pad_bottom)
#     """
#     b, c, h, w = x.size()
#
#     # 计算需要 padding 的大小
#     nh = math.ceil((h - ps) / step) + 1
#     nw = math.ceil((w - ps) / step) + 1
#     pad_h = max((nh - 1) * step + ps - h, 0)
#     pad_w = max((nw - 1) * step + ps - w, 0)
#
#     pad_top = pad_h // 2
#     pad_bottom = pad_h - pad_top
#     pad_left = pad_w // 2
#     pad_right = pad_w - pad_left
#
#     # 边缘补齐
#     x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
#
#     # 使用 unfold 提取 patch
#     unfold = F.unfold(x, kernel_size=ps, stride=step)
#     n_patches = unfold.size(-1)
#     patches = unfold.transpose(1, 2).reshape(b, n_patches, c, ps, ps)
#
#     return patches, nh, nw, (pad_left, pad_right, pad_top, pad_bottom)
#
#
# def patch_reverse(patches, nh, nw, ps, img_shape, pad_info):
#     """
#     Simply stitch patches back to image (no overlap average), and crop to original size.
#     Args:
#         patches (Tensor): Shape (b, n, c, ps, ps)
#         nh (int): Patches along height
#         nw (int): Patches along width
#         ps (int): Patch size
#         img_shape (tuple): Original image size (h, w)
#         pad_info (tuple): (pad_left, pad_right, pad_top, pad_bottom)
#     Returns:
#         x_reconstructed (Tensor): Reconstructed image of shape (b, c, h, w)
#     """
#     b, n, c, _, _ = patches.shape
#     assert n == nh * nw, "Patch count mismatch"
#
#     # 重排为网格格式 (b, nh, nw, c, ps, ps)
#     patches = patches.view(b, nh, nw, c, ps, ps)
#     patches = patches.permute(0, 3, 1, 4, 2, 5).contiguous()  # (b, c, nh, ps, nw, ps)
#     x_recon = patches.view(b, c, nh * ps, nw * ps)  # (b, c, H_pad, W_pad)
#
#     # 裁剪回原图大小
#     pad_left, pad_right, pad_top, pad_bottom = pad_info
#     b, c, h, w = img_shape
#     x_recon = x_recon[:, :, pad_top:pad_top + h, pad_left:pad_left + w]
#
#     return x_recon
#
#
# class SelfAttention(BaseModule):
#     """Attention module.
#     Args:
#         dim (int): Base channels.
#         heads (int): Head numbers.
#         qk_dim (int): Channels of query and key.
#     """
#
#     def __init__(self, dim, heads, qk_dim, max_len=512):
#         super().__init__()
#         self.heads = heads
#         self.dim = dim
#         self.qk_dim = qk_dim
#         self.scale = qk_dim ** -0.5
#         self.to_q = nn.Linear(dim, qk_dim, bias=False)
#         self.to_k = nn.Linear(dim, qk_dim, bias=False)
#         self.to_v = nn.Linear(dim, dim, bias=False)
#         self.proj = nn.Linear(dim, dim, bias=False)
#
#         # Learnable positional encoding
#         self.pos_emb = nn.Parameter(torch.randn(1, max_len, dim))
#
#
#     def forward(self, x):
#         b, n, _ = x.shape
#         pos = self.pos_emb[:, :n, :]  # truncate to input sequence length
#         x = x + pos  # add positional encoding
#         q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
#         out = F.scaled_dot_product_attention(q, k, v)  # scaled_dot_product_attention 需要PyTorch2.0之后版本
#         out = rearrange(out, 'b h n d -> b n (h d)')
#         return self.proj(out)
#
#
# class CrossAttention(BaseModule):
#     """Attention module.
#     Args:
#         dim (int): Base channels.
#         heads (int): Head numbers.
#         qk_dim (int): Channels of query and key.
#     """
#
#     def __init__(self, dim, heads, qk_dim, max_len=512):
#         super().__init__()
#         self.heads = heads
#         self.dim = dim
#         self.qk_dim = qk_dim
#         self.scale = qk_dim ** -0.5
#         self.to_q1 = nn.Linear(dim, qk_dim, bias=False)
#         self.to_k1 = nn.Linear(dim, qk_dim, bias=False)
#         self.to_v1 = nn.Linear(dim, dim, bias=False)
#         self.to_q2 = nn.Linear(dim, qk_dim, bias=False)
#         self.to_k2 = nn.Linear(dim, qk_dim, bias=False)
#         self.to_v2 = nn.Linear(dim, dim, bias=False)
#         self.proj = nn.Linear(dim, dim, bias=False)
#
#         # Learnable positional encoding
#         self.pos_emb = nn.Parameter(torch.randn(1, max_len, dim))
#
#     def forward(self, x1, x2):
#         b, n, _ = x1.shape
#         pos = self.pos_emb[:, :n, :]  # truncate to input sequence length
#         x1 = x1 + pos  # add positional encoding
#         x2 = x2 + pos  # add positional encoding
#         q1, k1, v1 = self.to_q1(x1), self.to_k1(x1), self.to_v1(x1)
#         q2, k2, v2 = self.to_q2(x2), self.to_k2(x2), self.to_v2(x2)
#         q1, k1, v1 = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q1, k1, v1))
#         q2, k2, v2 = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q2, k2, v2))
#         out1 = F.scaled_dot_product_attention(q2, k1, v1)  # scaled_dot_product_attention 需要PyTorch2.0之后版本
#         out2 = F.scaled_dot_product_attention(q1, k2, v2)  # scaled_dot_product_attention 需要PyTorch2.0之后版本
#         out = out1 + out2
#         out = rearrange(out, 'b h n d -> b n (h d)')
#         return self.proj(out)
#
#
# class LRSA(BaseModule):
#     """Attention module.
#     Args:
#         dim (int): Base channels.
#         num (int): Number of blocks.
#         qk_dim (int): Channels of query and key in Attention.
#         mlp_dim (int): Channels of hidden mlp in Mlp.
#         heads (int): Head numbers of Attention.
#     """
#
#     def __init__(self, dim, qk_dim, heads=1, patch_size=16, blocks_num=4):
#         super().__init__()
#         self.blocks_num = blocks_num
#         self.ps = patch_size
#         self.attn = nn.ModuleList()
#         self.alpha = []
#         for i in range(blocks_num):
#             self.attn.append(nn.Sequential(SelfAttention(dim, heads, qk_dim),
#                                       nn.LayerNorm(dim)))
#             self.alpha.append(nn.Parameter(torch.tensor([1.0, 0.0])))
#
#     def forward(self, x):
#         step = self.ps
#         for i in range(self.blocks_num):
#             crop_x, nh, nw, pad_info = patch_divide(x, step, self.ps)  # (b, n, c, ps, ps)
#             b, n, c, ph, pw = crop_x.shape
#             crop_x = rearrange(crop_x, 'b n c h w -> (b n) (h w) c')
#             crop_x = self.attn[i](crop_x)
#             crop_x = rearrange(crop_x, '(b n) (h w) c  -> b n c h w', n=n, w=pw)
#             weights = F.sigmoid(self.alpha[i])
#             x = weights[0] * patch_reverse(crop_x, nh, nw, self.ps, x.shape, pad_info) + weights[1] * x
#         return x
#
# # class LRSA(BaseModule):
# #     """Attention module.
# #     Args:
# #         dim (int): Base channels.
# #         num (int): Number of blocks.
# #         qk_dim (int): Channels of query and key in Attention.
# #         mlp_dim (int): Channels of hidden mlp in Mlp.
# #         heads (int): Head numbers of Attention.
# #     """
# #
# #     def __init__(self, dim, qk_dim, heads=1, patch_size=16):
# #         super().__init__()
# #         self.ps = patch_size
# #         self.attn = nn.Sequential(SelfAttention(dim, heads, qk_dim),
# #                                   nn.LayerNorm(dim))
# #         self.alpha = nn.Parameter(torch.tensor([1.0, 0.0]))
# #
# #     def forward(self, x):
# #         step = self.ps
# #         crop_x, nh, nw, pad_info = patch_divide(x, step, self.ps)  # (b, n, c, ps, ps)
# #         b, n, c, ph, pw = crop_x.shape
# #         crop_x = rearrange(crop_x, 'b n c h w -> (b n) (h w) c')
# #         crop_x = self.attn(crop_x)
# #         crop_x = rearrange(crop_x, '(b n) (h w) c  -> b n c h w', n=n, w=pw)
# #         weights = F.sigmoid(self.alpha)
# #         x = weights[0] * patch_reverse(crop_x, nh, nw, self.ps, x.shape, pad_info) + weights[1] * x
# #         return x
#
#
# class GRCA(BaseModule):
#     """Attention module.
#     Args:
#         dim (int): Base channels.
#         num (int): Number of blocks.
#         qk_dim (int): Channels of query and key in Attention.
#         mlp_dim (int): Channels of hidden mlp in Mlp.
#         heads (int): Head numbers of Attention.
#     """
#
#     def __init__(self, dim, qk_dim, heads=1, patch_size=16):
#         super().__init__()
#         self.ps = patch_size
#         self.attn = CrossAttention(dim, heads, qk_dim)
#         self.layer_norm = nn.LayerNorm(dim)
#         self.alpha = nn.Parameter(torch.tensor([1.0, 0.0, 0.0]))
#
#     def forward(self, x1, x2):
#         step = self.ps
#         crop_x1, nh, nw, pad_info = patch_divide(x1, step, self.ps)  # (b, n, c, ps, ps)
#         crop_x2, nh, nw, pad_info  = patch_divide(x2, step, self.ps)  # (b, n, c, ps, ps)
#         b, n, c, ph, pw = crop_x1.shape
#         crop_x1 = rearrange(crop_x1, 'b n c h w -> (b n) (h w) c')
#         crop_x2 = rearrange(crop_x2, 'b n c h w -> (b n) (h w) c')
#         crop_x = self.attn(crop_x1, crop_x2)
#         crop_x = self.layer_norm(crop_x)
#         crop_x = rearrange(crop_x, '(b n) (h w) c  -> b n c h w', n=n, w=pw)
#         weights = F.sigmoid(self.alpha)
#         x = weights[0] * patch_reverse(crop_x, nh, nw, self.ps, x1.shape, pad_info) + weights[1] * x1 + weights[2] * x2
#         return x
#
#
# if __name__ == "__main__":
#     # 输入参数配置
#     batch_size = 8  # Batch size
#     channels = 512  # 输入通道数
#     height = 100  # 高度
#     width = 168  # 宽度
#     ps = 16  # Patch size
#     qk_dim = channels  # Query-Key维度
#     out_dim = 256  # 输出维度
#     heads = 4  # Attention头数
#
#     # 创建一个输入张量，形状为 (batch_size, channels, height, width)
#     x = torch.randn(batch_size, channels, height, width)
#     # 初始化 LRSA 模块
#     lrsa = LRSA(dim=channels, qk_dim=qk_dim, heads=heads, patch_size=ps)
#     print(lrsa)
#     # 前向传播，传入输入张量和patch大小
#     output = lrsa(x)
#     # 打印输入和输出张量的形状
#     print(f"Input shape: {x.shape}")
#     print(f"Output shape: {output.shape}")
#
#     # # 初始化 GRCA 模块
#     # grca = GRCA(dim=channels, qk_dim=qk_dim, heads=heads, patch_size=ps)
#     # print(grca)
#     # # 前向传播，传入输入张量和patch大小
#     # output = grca(x, x)
#     # # 打印输入和输出张量的形状
#     # print(f"Input shape: {x.shape}")
#     # print(f"Output shape: {output.shape}")
#
#
