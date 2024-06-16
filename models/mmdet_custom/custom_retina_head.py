# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union, Callable
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule

from mmdet.models.layers import DyReLU
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

# from mmyolo.models.layers.yolo_bricks import ELANBlock


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
        # norm_cfg=dict(type="BN", requires_grad=True),
        norm_cfg=None,
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
                # ELANBlock(
                #     in_channels=in_channels,
                #     out_channels=self.feat_channels,
                #     middle_ratio=0.5,
                #     block_ratio=0.25,
                #     num_blocks=4,
                #     num_convs_in_block=1,
                # )
                # customDCNv3Module(
                #     channels=in_channels,
                #     group=int(in_channels/32),
                # )
                # TransformerCVBlock(
                #     d_model=in_channels,
                #     dim_feedforward=in_channels,
                #     norm_first=True,
                # )
                # nn.Sequential(
                #     DeformConv2dPack(
                #         in_channels,
                #         self.feat_channels,
                #         3,
                #         stride=1,
                #         padding=1,
                #         groups=1,
                #     ),
                #     nn.ReLU(inplace=True),
                # )
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
                # ELANBlock(
                #     in_channels=in_channels,
                #     out_channels=self.feat_channels,
                #     middle_ratio=0.5,
                #     block_ratio=0.25,
                #     num_blocks=4,
                #     num_convs_in_block=1,
                # )
                # customDCNv3Module(
                #     channels=in_channels,
                #     group=int(in_channels / 32),
                # )
                # nn.Sequential(
                #     DeformConv2dPack(
                #         in_channels,
                #         self.feat_channels,
                #         3,
                #         stride=1,
                #         padding=1,
                #         groups=1,
                #     ),
                #     nn.ReLU(inplace=True),
                # )
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

    