# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import torch
from mmengine.structures import BaseDataElement

from models.utils.misc import add_prefix
from mmdet.registry import MODELS
from models.algorithms.base import LossResults
from models.algorithms.single_teacher_distill_ import SingleTeacherDistill_


@MODELS.register_module()
class FpnTeacherDistill_(SingleTeacherDistill_):
    """``FpnTeacherDistill`` means teacher only execute backbone and neck.

    If the intermediate results required for distill algorithm are generated by
    the backbone and neck parts, using ``FpnTeacherDistill`` can speed up
    training.
    """

    def loss(
        self,
        batch_inputs: torch.Tensor,
        data_samples: Optional[List[BaseDataElement]] = None,
    ) -> LossResults:
        """Calculate losses from a batch of inputs and data samples."""

        losses = dict()

        # Unlike ``SingleTeacherDistill``, teacher will only execute
        # back + neck, not head, so there will be no loss.
        if self.teacher_trainable:
            with self.distiller.teacher_recorders:
                _ = self.teacher.extract_feat(batch_inputs)
        else:
            with self.distiller.teacher_recorders:
                with torch.no_grad():
                    _ = self.teacher.extract_feat(batch_inputs)
        
        
        with self.distiller.student_recorders:
            student_losses = self.student(
                batch_inputs, data_samples, mode='loss')
        losses.update(add_prefix(student_losses, 'student'))

        if not self.distillation_stopped:
            # Automatically compute distill losses based on
            # `loss_forward_mappings`.
            # The required data already exists in the recorders.
            distill_losses = self.distiller.compute_distill_losses(data_samples)
            losses.update(add_prefix(distill_losses, "distill"))


        return losses
