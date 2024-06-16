# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union, Callable
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


import numpy as np
from mmcv.cnn import ConvModule
# from mmcv.ops import DeformConv2dPack

from mmdet.registry import MODELS
from mmdet.models.dense_heads.anchor_head import AnchorHead
from mmdet.models.utils import multi_apply, images_to_levels
from mmdet.structures.bbox import BaseBoxes, cat_boxes, get_box_tensor
from mmdet.utils import (
    ConfigType,
    MultiConfig,
    InstanceList,
    OptConfigType,
    OptInstanceList,
    OptMultiConfig,
)
from mmcv.cnn import build_activation_layer, build_norm_layer


@MODELS.register_module()
class CustomRetinaHead(AnchorHead):
    r"""An anchor-based head used in `RetinaNet
    <https://arxiv.org/pdf/1708.02002.pdf>`_.

    The head contains two subnetworks. The first classifies anchor boxes and
    the second regresses deltas for the anchors.

    Example:
        >>> import torch
        >>> self = RetinaHead(11, 7)
        >>> x = torch.rand(1, 7, 32, 32)
        >>> cls_score, bbox_pred = self.forward_single(x)
        >>> # Each anchor predicts a score for each class except background
        >>> cls_per_anchor = cls_score.shape[1] / self.num_anchors
        >>> box_per_anchor = bbox_pred.shape[1] / self.num_anchors
        >>> assert cls_per_anchor == (self.num_classes)
        >>> assert box_per_anchor == 4
    """

    def __init__(
        self,
        num_classes,
        in_channels,
        feat_channels=256,
        stacked_convs=4,
        conv_cfg=None,
        act_cfg: ConfigType=dict(type="ReLU"),
        norm_cfg: ConfigType=None,
        # task_act_cfg: MultiConfig = (
        #     dict(type="GELU"),
        #     dict(type="HSigmoid", bias=3.0, divisor=6.0),
        # ),
        anchor_generator=dict(
            type="AnchorGenerator",
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128],
        ),
        init_cfg=dict(
            type="Normal",
            layer="Conv2d",
            std=0.01,
            override=dict(type="Normal", name="retina_cls", std=0.01, bias_prob=0.01),
        ),
        **kwargs,
    ):
        assert stacked_convs >= 0, (
            "`stacked_convs` must be non-negative integers, "
            f"but got {stacked_convs} instead."
        )
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        # self.task_act_cfg = task_act_cfg


        super(CustomRetinaHead, self).__init__(
            num_classes,
            in_channels,
            feat_channels=feat_channels,
            anchor_generator=anchor_generator,
            init_cfg=init_cfg,
            **kwargs,
        )


    def _init_layers(self):
        """Initialize layers of the head."""
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()

        in_channels = self.in_channels
        for i in range(self.stacked_convs):
            self.cls_convs.append(
                ConvModule(
                    in_channels,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    groups=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                )
            )
            self.reg_convs.append(
                ConvModule(
                    in_channels,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    groups=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                )
            )
            in_channels = self.feat_channels

        self.retina_cls = nn.Conv2d(
            in_channels, self.num_base_priors * self.cls_out_channels, 3, padding=1
        )
        reg_dim = self.bbox_coder.encode_size
        self.retina_reg = nn.Conv2d(
            in_channels, self.num_base_priors * reg_dim, 3, padding=1
        )

    def forward_single(self, x):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level
                    the channels number is num_anchors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale
                    level, the channels number is num_anchors * 4.
        """
        cls_feat = x
        reg_feat = x

        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)

        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)

        return cls_score, bbox_pred

# import copy
# def clones(module, N):
#     "Produce N identical layers."
#     return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# class TransDWConv(nn.Module):
#     def __init__(
#         self,
#         channels: int = 64,
#         num_heads: int = 4,
#         mlp_ratio: float = 0.5,
#         act_cfg: ConfigType=dict(type="GELU"),
#         norm_cfg: ConfigType=dict(type="LN"),
#         **kwargs
#     ) -> None:
#         super().__init__(**kwargs)

#         self.channels = channels
#         self.num_heads = num_heads

#         # self.linears = clones(nn.Linear(channels, channels), 3)

#         self.norm1 = build_norm_layer(cfg=norm_cfg, num_features=channels)[1]
#         self.norm2 = build_norm_layer(cfg=norm_cfg, num_features=channels)[1]

#         self.dw_feed_forward = DWFeadForward(channels, mlp_ratio, act_cfg)

#         self.act = build_activation_layer(cfg=act_cfg)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = x.permute(0, 2, 3, 1).contiguous()
#         shortcut = x
#         x = self.norm1(x)
#         # query, key, value = [l(x) for l in self.linears]
#         with torch.backends.cuda.sdp_kernel(enable_math=False):
#             x = F.scaled_dot_product_attention(
#                 query=x, key=x, value=x, attn_mask=None
#             )
#         x = shortcut + x
#         x = self.norm2(x)
#         x = x.permute(0, 3, 1, 2).contiguous()
#         x = x + self.dw_feed_forward(x)

#         x = self.act(x)
#         return x


# # ------------------------------------------------------------------------------
# # Feed Forward 模块
# # ------------------------------------------------------------------------------
# class DWFeadForward(nn.Module):
#     def __init__(
#         self,
#         channel: int = 64,
#         mlp_ratio: float = 0.5,
#         act_cfg: ConfigType = dict(type="GELU"),
#     ):
#         super(DWFeadForward, self).__init__()
#         self.channel = channel
#         self.mlp_ratio = mlp_ratio
#         self.hidden_fetures = int(channel * mlp_ratio)

#         self.input_project = nn.Conv2d(
#             channel, self.hidden_fetures, kernel_size=1, bias=True
#         )

#         self.act = build_activation_layer(act_cfg)

#         self.dwconv = nn.Conv2d(
#             self.hidden_fetures,
#             self.hidden_fetures,
#             kernel_size=3,
#             padding=1,
#             groups=self.hidden_fetures,
#             bias=True,
#         )

#         self.output_project = nn.Conv2d(
#             self.hidden_fetures, self.channel, kernel_size=1, bias=True
#         )  # 1x1 conv

#     def forward(self, x):
#         """

#         :param input: [bs, C, H, W]
#         :return: [bs, C, H, W]
#         """

#         # feed forward
#         x = self.input_project(x)
#         x = self.act(x)
#         x = self.dwconv(x)
#         x = self.output_project(x)

#         return x   