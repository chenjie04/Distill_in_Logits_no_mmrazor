_base_ = [
    "mmdet/_base_/datasets/coco_detection.py",
    "mmdet/_base_/schedules/schedule_1x.py",
    "mmdet/_base_/default_runtime.py",
]

custom_imports = dict(
    imports=[
        "models.runner.distill_val_loop",
        "models.algorithms.fpn_teacher_distill_",
        "models.distillers.configurable_distiller_retina_test",
        "models.task_modules.recorder.module_outputs_recorder",
    ],
    allow_failed_imports=False,
)

teacher_ckpt = "checkpoints/tood_x101_64x4d_fpn_mstrain_2x_coco_20211211_003519-a4f36113.pth"

model = dict(
    _scope_="mmdet",
    type="FpnTeacherDistill_",
    architecture=dict(
        cfg_path="configs/mmdet/tood/tood_r50_fpn_1x_coco.py", pretrained=False
    ),
    teacher=dict(
        cfg_path="configs/mmdet/tood/tood_x101-64x4d_fpn_ms-2x_coco.py",  # noqa: E251
        pretrained=False,
    ),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type="ConfigurableDistillerInLogits",
        student_recorders=dict(fpn=dict(type="ModuleOutputs", source="neck")),
        teacher_recorders=dict(fpn=dict(type="ModuleOutputs", source="neck")),
        projector_cfg=dict(
            type="TOODHead",
            num_classes=80,
            in_channels=256,
            stacked_convs=2,
            feat_channels=256,
            anchor_type="anchor_free",
            anchor_generator=dict(
                type="AnchorGenerator",
                ratios=[1.0],
                octave_base_scale=8,
                scales_per_octave=1,
                strides=[8, 16, 32, 64, 128],
            ),
            bbox_coder=dict(
                type="DeltaXYWHBBoxCoder",
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2],
            ),
            initial_loss_cls=dict(
                type="FocalLoss",
                use_sigmoid=True,
                activated=True,  # use probability instead of logit as input
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0,
            ),
            loss_cls=dict(
                type="QualityFocalLoss",
                use_sigmoid=True,
                activated=True,  # use probability instead of logit as input
                beta=2.0,
                loss_weight=1.0,
            ),
            loss_bbox=dict(type="GIoULoss", loss_weight=2.0),
            train_cfg=dict(
                initial_epoch=4,
                initial_assigner=dict(type="ATSSAssigner", topk=9),
                assigner=dict(type="TaskAlignedAssigner", topk=13),
                alpha=1,
                beta=6,
                allowed_border=-1,
                pos_weight=-1,
                debug=False,
            ),
            test_cfg=dict(
                nms_pre=1000,
                min_bbox_size=0,
                score_thr=0.05,
                nms=dict(type="nms", iou_threshold=0.6),
                max_per_img=100,
            ),
        ),
        distill_cls_weight=1.0,
        distill_reg_weight=10.0,
        loss_forward_mappings=dict(
            loss_prject_optim=dict(
                preds_S=dict(
                    from_student=True, recorder="fpn", data_idx=[0, 1, 2, 3, 4]
                ),
                preds_T=dict(
                    from_student=False, recorder="fpn", data_idx=[0, 1, 2, 3, 4]
                ),
            ),
        ),
    ),
)

# find_unused_parameters = True

val_cfg = dict(_delete_=True, type="SingleTeacherDistillValLoop")

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    pin_memory=True,
)

# default_hooks = dict(checkpoint=dict(max_keep_ckpts=3, interval=1, save_best="auto"))

# optimizer
# optim_wrapper = dict(type="OptimWrapper", optimizer=dict(lr=0.01))
"""
为了启用混合精度训练，需要做以下修改：
1. 在layer_norm()计算标准差时,如果std为0,则梯度将为NaN
    std = feat.float().std(dim=-1, keepdim=True)
2. 在计算focal loss、cross-entropy等涉及log()的函数时,需要把pred转为float类型
"""
optim_wrapper = dict(type="AmpOptimWrapper", optimizer=dict(lr=0.001))


param_scheduler = [
    dict(type="LinearLR", start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type="MultiStepLR",
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1,
    ),
]
