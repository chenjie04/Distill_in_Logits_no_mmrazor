# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch.nn as nn
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner

from mmdet.registry import HOOKS


@HOOKS.register_module()
class MeanStudentProjectorHook(Hook):
    """Mean Teacher Hook.

    Mean Student is an efficient semi-supervised learning method in
    `Mean Teacher <https://arxiv.org/abs/1703.01780>`_.
    This method requires two models with exactly the same structure,
    as the student model and the teacher model, respectively.
    The teacher model updates the parameters through gradient descent,
    and the student model updates the parameters through
    exponential moving average of the teacher model.

    Args:
        momentum (float): The momentum used for updating student's parameter.
            Student's parameter are updated with the formula:
           `student = (1-momentum) * student + momentum * teacher`.
            Defaults to 0.001.
        interval (int): Update student's parameter every interval iteration.
            Defaults to 1.
        skip_buffers (bool): Whether to skip the model buffers, such as
            batchnorm running stats (running_mean, running_var), it does not
            perform the ema operation. Default to True.
    """

    def __init__(
        self, momentum: float = 0.001, interval: int = 1, skip_buffer=True
    ) -> None:
        assert 0 < momentum <= 1
        self.momentum = momentum
        self.interval = interval
        self.skip_buffers = skip_buffer

    def before_train(self, runner: Runner) -> None:
        """To check that teacher model and student model exist."""
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        distiller = model.distiller
        assert hasattr(distiller, "teacher_projector")
        assert hasattr(distiller, "student_projector")
        # only do it at initial stage
        if runner.iter == 0:
            self.momentum_update(distiller, 1)

    def after_train_iter(
        self,
        runner: Runner,
        batch_idx: int,
        data_batch: Optional[dict] = None,
        outputs: Optional[dict] = None,
    ) -> None:
        """Update teacher's parameter every self.interval iterations."""
        if (runner.iter + 1) % self.interval != 0:
            return
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        distiller = model.distiller
        self.momentum_update(distiller, self.momentum)

    def momentum_update(self, model: nn.Module, momentum: float) -> None:
        """Compute the moving average of the parameters using exponential
        moving average."""
        # print("*******************************************************")
        # print(model)
        # print("*******************************************************")
        if self.skip_buffers:
            for (src_name, src_parm), (dst_name, dst_parm) in zip(
                model.teacher_projector.named_parameters(), model.student_projector.named_parameters()
            ):
                dst_parm.data.mul_(1 - momentum).add_(src_parm.data, alpha=momentum)
        else:
            for src_parm, dst_parm in zip(
                model.teacher_projector.state_dict().values(), model.student_projector.state_dict().values()
            ):
                # exclude num_tracking
                if dst_parm.dtype.is_floating_point:
                    dst_parm.data.mul_(1 - momentum).add_(src_parm.data, alpha=momentum)
