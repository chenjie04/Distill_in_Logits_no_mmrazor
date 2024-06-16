auto_scale_lr = dict(base_batch_size=16, enable=False)
backend_args = None
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'models.algorithms.fpn_teacher_distill_',
        'models.distillers.configurable_distiller_retina_v4',
        'models.mmdet_custom.custom_retina_head',
    ])
data_root = 'data/coco/'
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(
        _scope_='mmdet',
        interval=1,
        max_keep_ckpts=3,
        save_best='auto',
        type='CheckpointHook'),
    logger=dict(_scope_='mmdet', interval=50, type='LoggerHook'),
    param_scheduler=dict(_scope_='mmdet', type='ParamSchedulerHook'),
    sampler_seed=dict(_scope_='mmdet', type='DistSamplerSeedHook'),
    timer=dict(_scope_='mmdet', type='IterTimerHook'),
    visualization=dict(_scope_='mmdet', type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'pytorch'
load_from = None
log_level = 'INFO'
log_processor = dict(
    _scope_='mmdet', by_epoch=True, type='LogProcessor', window_size=50)
model = dict(
    _scope_='mmrazor',
    architecture=dict(
        cfg_path='mmdet::retinanet/retinanet_r50_fpn_1x_coco.py',
        pretrained=False),
    distiller=dict(
        distill_cls_weight=1.0,
        distill_losses=None,
        distill_reg_weight=1.0,
        loss_forward_mappings=dict(
            loss_prject_optim=dict(
                preds_S=dict(
                    data_idx=[
                        0,
                        1,
                        2,
                        3,
                        4,
                    ],
                    from_student=True,
                    recorder='fpn'),
                preds_T=dict(
                    data_idx=[
                        0,
                        1,
                        2,
                        3,
                        4,
                    ],
                    from_student=False,
                    recorder='fpn'))),
        projector_cfg=dict(
            _scope_='mmdet',
            anchor_generator=dict(
                _scope_='mmdet',
                octave_base_scale=4,
                ratios=[
                    0.5,
                    1.0,
                    2.0,
                ],
                scales_per_octave=3,
                strides=[
                    8,
                    16,
                    32,
                    64,
                    128,
                ],
                type='mmdet.AnchorGenerator'),
            bbox_coder=dict(
                _scope_='mmdet',
                target_means=[
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                target_stds=[
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                ],
                type='mmdet.DeltaXYWHBBoxCoder'),
            feat_channels=256,
            in_channels=256,
            init_cfg=dict(
                layer='Conv2d',
                override=dict(
                    bias_prob=0.01, name='retina_cls', std=0.01,
                    type='Normal'),
                std=0.01,
                type='Normal'),
            loss_bbox=dict(
                _scope_='mmdet', loss_weight=1.0, type='mmdet.L1Loss'),
            loss_cls=dict(
                _scope_='mmdet',
                alpha=0.25,
                gamma=2.0,
                loss_weight=1.0,
                type='mmdet.FocalLoss',
                use_sigmoid=True),
            num_classes=80,
            stacked_convs=1,
            train_cfg=dict(
                allowed_border=-1,
                assigner=dict(
                    _scope_='mmdet',
                    ignore_iof_thr=-1,
                    min_pos_iou=0,
                    neg_iou_thr=0.4,
                    pos_iou_thr=0.5,
                    type='mmdet.MaxIoUAssigner'),
                debug=False,
                pos_weight=-1,
                sampler=dict(_scope_='mmdet', type='mmdet.PseudoSampler')),
            type='CustomRetinaHead'),
        student_recorders=dict(fpn=dict(source='neck', type='ModuleOutputs')),
        teacher_recorders=dict(fpn=dict(source='neck', type='ModuleOutputs')),
        type='ConfigurableDistillerInLogits'),
    teacher=dict(
        cfg_path=
        'mmdet::fcos/fcos_x101-64x4d_fpn_gn-head_ms-640-800-2x_coco.py',
        pretrained=False),
    teacher_ckpt=
    'https://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_2x_coco/fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_2x_coco-ede514a8.pth',
    type='FpnTeacherDistill_')
optim_wrapper = dict(
    _scope_='mmdet',
    optimizer=dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0001),
    type='AmpOptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=500, start_factor=1e-05, type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=12,
        gamma=0.1,
        milestones=[
            8,
            11,
        ],
        type='MultiStepLR'),
]
resume = False
teacher_ckpt = 'https://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_2x_coco/fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_2x_coco-ede514a8.pth'
test_cfg = dict(_scope_='mmdet', type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        _scope_='mmdet',
        ann_file='annotations/instances_val2017.json',
        backend_args=None,
        data_prefix=dict(img='val2017/'),
        data_root='data/coco/',
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1333,
                800,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(_scope_='mmdet', shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    _scope_='mmdet',
    ann_file='data/coco/annotations/instances_val2017.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
test_pipeline = [
    dict(_scope_='mmdet', backend_args=None, type='LoadImageFromFile'),
    dict(_scope_='mmdet', keep_ratio=True, scale=(
        1333,
        800,
    ), type='Resize'),
    dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
    dict(
        _scope_='mmdet',
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
train_cfg = dict(
    _scope_='mmdet', max_epochs=12, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_sampler=dict(_scope_='mmdet', type='AspectRatioBatchSampler'),
    batch_size=4,
    dataset=dict(
        _scope_='mmdet',
        ann_file='annotations/instances_train2017.json',
        backend_args=None,
        data_prefix=dict(img='train2017/'),
        data_root='data/coco/',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(keep_ratio=True, scale=(
                1333,
                800,
            ), type='Resize'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PackDetInputs'),
        ],
        type='CocoDataset'),
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(_scope_='mmdet', shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(_scope_='mmdet', backend_args=None, type='LoadImageFromFile'),
    dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
    dict(_scope_='mmdet', keep_ratio=True, scale=(
        1333,
        800,
    ), type='Resize'),
    dict(_scope_='mmdet', prob=0.5, type='RandomFlip'),
    dict(_scope_='mmdet', type='PackDetInputs'),
]
val_cfg = dict(type='mmrazor.SingleTeacherDistillValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        _scope_='mmdet',
        ann_file='annotations/instances_val2017.json',
        backend_args=None,
        data_prefix=dict(img='val2017/'),
        data_root='data/coco/',
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1333,
                800,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(_scope_='mmdet', shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    _scope_='mmdet',
    ann_file='data/coco/annotations/instances_val2017.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
vis_backends = [
    dict(_scope_='mmdet', type='LocalVisBackend'),
]
visualizer = dict(
    _scope_='mmdet',
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/fcos_x101_retina_r50_2x_coco_retina_v4'
