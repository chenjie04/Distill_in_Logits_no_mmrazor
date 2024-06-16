# Copyright (c) OpenMMLab. All rights reserved.
"""
# ------------------------------------------------------------
# Version :   1.0
# Author  :   chenjie04
# Email   :   gxuchenjie04@gmail.com
# Time    :   2024/03/27 18:38:16
# Descript:   使用标准的KL散度测量学生模型与教师模型的分类距离,
#             以及标准的smooth_l1_loss测量学生模型与教师模型的回归距离
#             效果比v2略好
# ---------------------------------------------------------------
"""


import warnings
import copy

# from inspect import signature
from typing import Dict, List, Optional, Union, Tuple

from mmengine.model import BaseModel
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from mmcv.cnn import ConvModule

from mmdet.registry import MODELS
from models.algorithms.base import LossResults
from mmdet.models.utils import multi_apply
from mmdet.structures.bbox import bbox_overlaps
from models.task_modules.recorder.recoder_manager import RecorderManager

from models.distillers.base_distiller import BaseDistiller

# from mmrazor.models.losses import ChannelWiseDivergence, KLDivergence


@MODELS.register_module()
class ConfigurableDistillerInLogits(BaseDistiller):
    """``ConfigurableDistiller`` is a powerful tool that can reproduce most
    distillation algorithms without modifying the code of teacher or student
    models.

    ``ConfigurableDistiller`` can get various intermediate results of the
    model in a hacky way by ``Recorder``. More details see user-docs for
    ``Recorder``.

    ``ConfigurableDistiller`` can use the teacher's intermediate results to
    override the student's intermediate results in a hacky way by ``Delivery``.
    More details see user-docs for ``Delivery``.

    Args:
        student_recorders (dict, optional): Config for multiple recorders. A
            student model may have more than one recorder. These recorders
            only record the student model's intermediate results. Defaults to
            None.
        teacher_recorders (dict, optional): Config for multiple recorders. A
            teacher model may have more than one recorder. These recorders
            only record the teacher model's intermediate results. Defaults to
            None.
        loss_forward_mappings: (Dict[str, Dict], optional): Mapping between
            distill loss forward arguments and records.

    Note:
        If a distill loss needs to backward, the name of the loss must contain
        "loss". If it is only used as a statistical value, the name can not
        contain "loss". More details see docs for
        :func:`mmengine.model.BaseModel._parse_loss`.

    Note:
        The keys of ``loss_forward_mappings`` should be consistent with the
        keys of ``distill_losses``.

        Each item in ``loss_forward_mappings`` is a mapping between a distill
        loss and its forward arguments. The keys of the mapping are the
        signature of the loss's forward, and the values of the mapping are the
        recorded data location.

        ``from_recorder``refers to the recorder where the data is stored, and
        if ``from_student`` is True, it means the recorder is in `
        `student_recorders``; otherwise, it means the recorder is in
        ``teacher_recorders``.

        A connector can be called according to its `connector_name`, so that a
        input can use a different connector in different loss.

    Examples:
        >>> distill_losses = dict(
        ...     loss_neck=dict(type='L2Loss', loss_weight=5))

        >>> student_recorders = dict(
        ...     feat = dict(type='ModuleOutputs', sources='neck.gap'))

        >>> teacher_recorders = dict(
        ...     feat = dict(type='ModuleOutputs', sources='neck.gap'))

        >>> loss_forward_mappings = dict(
        ...     loss_neck=dict(
        ...         s_feature=dict(from_recorder='feat', from_student=True,
        ...                        connector='loss_neck_sfeat'),
        ...         t_feature=dict(from_recorder='feat', from_student=False,
        ...                        connector='loss_neck_tfeat')))
    """

    def __init__(
        self,
        student_recorders: Optional[Dict[str, Dict]] = None,
        teacher_recorders: Optional[Dict[str, Dict]] = None,
        loss_forward_mappings: Optional[Dict[str, Dict]] = None,
        distill_cls_weight: float = 1.0,
        distill_reg_weight: float = 1.0,
        resize_stu: bool = True,
        projector_cfg: Dict = dict(
            type="CustomRetinaHead",
            num_classes=80,
            in_channels=256,
            stacked_convs=1,
            feat_channels=256,
            anchor_generator=dict(
                type="mmdet.AnchorGenerator",
                octave_base_scale=4,
                scales_per_octave=3,
                ratios=[0.5, 1.0, 2.0],
                strides=[8, 16, 32, 64, 128],
            ),
            bbox_coder=dict(
                type="mmdet.DeltaXYWHBBoxCoder",
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[1.0, 1.0, 1.0, 1.0],
            ),
            loss_cls=dict(
                type="mmdet.FocalLoss",
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0,
            ),
            loss_bbox=dict(type="mmdet.L1Loss", loss_weight=1.0),
            train_cfg=dict(
                assigner=dict(
                    type="mmdet.MaxIoUAssigner",
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.4,
                    min_pos_iou=0,
                    ignore_iof_thr=-1,
                ),
                sampler=dict(
                    type="mmdet.PseudoSampler"
                ),  # Focal loss should use PseudoSampler
                allowed_border=-1,
                pos_weight=-1,
                debug=False,
            ),
            init_cfg=dict(
                type="Normal",
                layer="Conv2d",
                std=0.01,
                override=dict(
                    type="Normal", name="retina_cls", std=0.01, bias_prob=0.01
                ),
            ),
        ),
    ):
        super().__init__()
        # The recorder manager is just constructed, but not really initialized
        # yet. Recorder manager initialization needs to input the corresponding
        # model.
        self.student_recorders = RecorderManager(student_recorders)
        self.teacher_recorders = RecorderManager(teacher_recorders)


        if loss_forward_mappings:
            self.loss_forward_mappings = loss_forward_mappings
        else:
            self.loss_forward_mappings = dict()

        self.resize_stu = resize_stu

        self.distill_cls_weight = distill_cls_weight
        self.distill_reg_weight = distill_reg_weight

        # 下一步验证提高教师映射模块的卷积层数
        self.projector_cfg = projector_cfg
        self.teacher_projector = MODELS.build(self.projector_cfg)



    def prepare_from_student(self, model: BaseModel) -> None:
        """Initialize student recorders."""
        self.student_recorders.initialize(model)

    def prepare_from_teacher(self, model: nn.Module) -> None:
        """Initialize teacher recorders."""
        self.teacher_recorders.initialize(model)


    def get_record(
        self,
        recorder: str,
        from_student: bool,
        record_idx: int = 0,
        data_idx: Optional[int] = None,
        connector: Optional[str] = None,
        connector_idx: Optional[int] = None,
    ) -> List:
        """According to each item in ``record_infos``, get the corresponding
        record in ``recorder_manager``."""

        if from_student:
            recorder_ = self.student_recorders.get_recorder(recorder)
        else:
            recorder_ = self.teacher_recorders.get_recorder(recorder)
        record_data = recorder_.get_record_data(record_idx, data_idx)

        if connector:
            record_data = self.connectors[connector](record_data)
        if connector_idx is not None:
            record_data = record_data[connector_idx]

        return record_data

    # def layer_norm(self, feat: torch.Tensor) -> torch.Tensor:
    #     """Normalize the feature maps to have zero mean and unit variances.

    #     Args:
    #         feat (torch.Tensor): The original feature map with shape
    #             (N, C, H, W).
    #     """
    #     """Normalize the feature maps to have zero mean and unit variances.

    #     Args:
    #         feat (torch.Tensor): The original feature map with shape
    #             (N, C, H, W).
    #     """
    #     assert len(feat.shape) == 4
    #     N, C, H, W = feat.shape
    #     feat = feat.permute(1, 0, 2, 3).reshape(C, -1)
    #     mean = feat.mean(dim=-1, keepdim=True)
    #     std = feat.float().std(dim=-1, keepdim=True)
    #     feat = (feat - mean) / (std)
    #     return feat.reshape(C, N, H, W).permute(1, 0, 2, 3)
    
    def layer_norm(self, feat: torch.Tensor) -> torch.Tensor:
        """Layer normalization.
        """
        assert len(feat.shape) == 4
        mean = feat.mean(dim=[1,2,3],keepdim=True)
        var = feat.var(dim=[1,2,3],keepdim=True)
        feat = (feat - mean) / (torch.sqrt(var + 1e-8))
        return feat
    
    # def layer_norm(self, feat: torch.Tensor) -> torch.Tensor:
    #     """Instance normalization. 不太行
    #     """
    #     assert len(feat.shape) == 4
    #     mean = feat.mean(dim=[2,3],keepdim=True)
    #     var = feat.var(dim=[2,3],keepdim=True)
    #     feat = (feat - mean) / (torch.sqrt(var + 1e-8))
    #     return feat
    
    # def layer_norm(self, feat: torch.Tensor, groups: int) -> torch.Tensor:
    #     """Group normalization.
    #     """
    #     assert len(feat.shape) == 4
    #     shape = feat.shape
    #     feat = feat.view(shape[0], groups, -1, shape[2], shape[3])
    #     mean = feat.mean(dim=[2,3,4],keepdim=True)
    #     var = feat.var(dim=[2,3,4],keepdim=True)
    #     feat = (feat - mean) / (torch.sqrt(var + 1e-8))
    #     feat = feat.view(shape)
    #     return feat



    def compute_distill_losses(
        self, data_samples: Optional[List] = None
    ) -> LossResults:
        """Compute distill losses automatically."""
        # Record all computed losses' results.

        # 获取教师的特征图
        mapping_teacher = self.loss_forward_mappings["loss_prject_optim"]["preds_T"]
        from_student, recorder, data_idx = (
            mapping_teacher["from_student"],
            mapping_teacher["recorder"],
            mapping_teacher["data_idx"],
        )
        feats_teacher = []
        for idx in data_idx:
            feats_teacher.append(self.get_record(recorder, from_student, data_idx=idx))

        # 获取学生的特征图
        mapping_student = self.loss_forward_mappings["loss_prject_optim"]["preds_S"]
        from_student, recorder, data_idx = (
            mapping_student["from_student"],
            mapping_student["recorder"],
            mapping_student["data_idx"],
        )
        feats_student = []
        for idx in data_idx:
            feats_student.append(self.get_record(recorder, from_student, data_idx=idx))

        # ------------------------------------------------------------
        # 将学生模型的特征图调整到教师模型的大小，或者反过来
        resized_feats_student, resized_feats_teacher = [], []
        for feats_student_i, feats_teacher_i in zip(feats_student, feats_teacher):
            if self.resize_stu:
                feats_student_i = F.interpolate(
                    feats_student_i,
                    size=feats_teacher_i.size()[2:],
                    mode="bilinear",
                    align_corners=True,
                )
            else:
                feats_teacher_i = F.interpolate(
                    feats_teacher_i,
                    size=feats_student_i.size()[2:],
                    mode="bilinear",
                    align_corners=True,
                )
            feats_student_i = self.layer_norm(feats_student_i)
            feats_teacher_i = self.layer_norm(feats_teacher_i)
            resized_feats_student.append(feats_student_i)
            resized_feats_teacher.append(feats_teacher_i)


        
            

        # 映射到logits空间
        (
            cls_scores_teacher,
            bbox_preds_teacher,
        ) = self.teacher_projector(resized_feats_teacher)
        (
            cls_scores_student,
            bbox_preds_student,
        ) = self.teacher_projector(resized_feats_student)



        # 计算损失
        losses = dict()

        # ------------------------------------------------------------------
        # 计算教师模型与Ground-truth的损失，看看loss是否正确下降

        batch_gt_instances = []
        for data_sample in data_samples:
            batch_gt_instances.append(data_sample.gt_instances)

        # print(data_samples[0])
        batch_img_metas = []
        for data_sample in data_samples:
            batch_img_metas.append(data_sample.metainfo)

        batch_gt_instances_ignore = []
        for data_sample in data_samples:
            batch_gt_instances_ignore.append(data_sample.ignored_instances)

        teacher_gt_loss = self.teacher_projector.loss_by_feat(
            cls_scores_teacher,
            bbox_preds_teacher,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore,
        )

        for key, value in teacher_gt_loss.items():
            losses[f"teacher.{key}"] = torch.sum(torch.stack(value))


        # ------------------------------------------------------------
        # 计算学生模型与教师模型的距离，使得学生可以学习教师模型的行为
            
        # cls_l1_loss = 0.0
        cls_loss = 0.0
        for cls_score_s, cls_score_t in zip(
            cls_scores_student,
            cls_scores_teacher,
        ):
            cls_score_s = self.layer_norm(cls_score_s)
            cls_score_t = self.layer_norm(cls_score_t.detach())
            # cls_l1_loss += F.smooth_l1_loss(cls_score_s, cls_score_t, reduction="mean") # 我认为需要，下一步验证, 似乎是不需要的？
            cls_score_s = (
                cls_score_s.permute(0, 2, 3, 1).reshape(-1, self.teacher_projector.num_classes)
            )
            cls_score_t = (
                cls_score_t.permute(0, 2, 3, 1).reshape(-1, self.teacher_projector.num_classes)
            )
            cls_loss += soft_cross_entropy(cls_score_s, cls_score_t)
           


        # losses["cls_l1_loss"] = cls_l1_loss * self.distill_cls_weight 
        losses["cls_loss"] = cls_loss * self.distill_cls_weight

        reg_loss = 0.0
        for bbox_pred_s, bbox_pred_t in zip(
            bbox_preds_student,
            bbox_preds_teacher,
        ):
            bbox_pred_s = self.layer_norm(bbox_pred_s)
            bbox_pred_t = self.layer_norm(bbox_pred_t.detach())
            reg_loss += F.smooth_l1_loss(bbox_pred_s, bbox_pred_t, reduction="mean")

        # print("reg_loss: ",reg_loss)
        losses["reg_loss"] = reg_loss * self.distill_reg_weight

        # ------------------------------------------------------------

        return losses

def soft_cross_entropy(logits_student, logits_teacher):
    pred_student = F.softmax(logits_student, dim=1)
    pred_teacher = F.softmax(logits_teacher, dim=1)
    idx = torch.arange(pred_teacher.size(0), dtype=torch.long)
    idy = torch.argmax(pred_teacher, dim=1)
    prob = pred_teacher[idx, idy]
    pred_student = pred_student[idx, idy]
    loss = torch.sum(- (prob) * torch.log(pred_student))
    return loss / prob.size(0)

