_base_ = [
    "mmdet/_base_/datasets/coco_detection.py",
    "mmdet/_base_/schedules/schedule_1x.py",
    "mmdet/_base_/default_runtime.py",
]

custom_imports = dict(
    imports=[
        "models.runner.distill_val_loop",
        "models.algorithms.fpn_teacher_distill_",
        "models.distillers.configurable_distiller_retina",
        "models.task_modules.recorder.module_outputs_recorder"
    ],
    allow_failed_imports=False,
)

teacher_ckpt = 'https://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_2x_coco/fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_2x_coco-ede514a8.pth'  # noqa: E501

model = dict(
    _scope_="mmdet",
    type="FpnTeacherDistill_",
    architecture=dict(
        cfg_path='configs/mmdet/retinanet/retinanet_r50_fpn_1x_coco.py', pretrained=False
    ),
    teacher=dict(
        cfg_path=  # noqa: E251
        'configs/mmdet/fcos/fcos_x101-64x4d_fpn_gn-head_ms-640-800-2x_coco.py',
        pretrained=False,
    ),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type="ConfigurableDistillerInLogits",
        student_recorders=dict(fpn=dict(type="ModuleOutputs", source="neck")),
        teacher_recorders=dict(fpn=dict(type="ModuleOutputs", source="neck")),
        projector_cfg=dict(
            type="mmdet.RetinaHead",
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
        distill_cls_weight=1.0,
        distill_reg_weight=1.0,
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

# optimizer
# optim_wrapper = dict(type="OptimWrapper", optimizer=dict(lr=0.01))
"""
为了启用混合精度训练，需要做以下修改：
1. 在layer_norm()计算标准差时,如果std为0,则梯度将为NaN
    std = feat.float().std(dim=-1, keepdim=True)
2. 在计算focal loss、cross-entropy等涉及log()的函数时,需要把pred转为float类型
"""
optim_wrapper = dict(type="AmpOptimWrapper", optimizer=dict(lr=0.01))


default_hooks = dict(checkpoint=dict(max_keep_ckpts=3, interval=1, save_best="auto"))


param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]