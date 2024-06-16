auto_scale_lr = dict(base_batch_size=16, enable=False)
backend_args = None
check_points = 'work_dirs/pkd_fpn_mask-rcnn_swin_retina_r50_2x_coco/last_checkpoint'
data_root = 'data/coco/'
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(_scope_='mmdet', interval=1, type='CheckpointHook'),
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
find_unused_parameters = False
launcher = 'pytorch'
load_from = None
log_level = 'INFO'
log_processor = dict(
    _scope_='mmdet', by_epoch=True, type='LogProcessor', window_size=50)
model = dict(
    _scope_='mmrazor',
    architecture=dict(
        cfg_path='mmdet::retinanet/retinanet_r50_fpn_2x_coco.py',
        pretrained=False),
    distiller=dict(
        distill_losses=dict(
            loss_pkd_fpn0=dict(loss_weight=6, type='PKDLoss'),
            loss_pkd_fpn1=dict(loss_weight=6, type='PKDLoss'),
            loss_pkd_fpn2=dict(loss_weight=6, type='PKDLoss'),
            loss_pkd_fpn3=dict(loss_weight=6, type='PKDLoss')),
        loss_forward_mappings=dict(
            loss_pkd_fpn0=dict(
                preds_S=dict(data_idx=0, from_student=True, recorder='fpn'),
                preds_T=dict(data_idx=0, from_student=False, recorder='fpn')),
            loss_pkd_fpn1=dict(
                preds_S=dict(data_idx=1, from_student=True, recorder='fpn'),
                preds_T=dict(data_idx=1, from_student=False, recorder='fpn')),
            loss_pkd_fpn2=dict(
                preds_S=dict(data_idx=2, from_student=True, recorder='fpn'),
                preds_T=dict(data_idx=2, from_student=False, recorder='fpn')),
            loss_pkd_fpn3=dict(
                preds_S=dict(data_idx=3, from_student=True, recorder='fpn'),
                preds_T=dict(data_idx=3, from_student=False, recorder='fpn'))),
        student_recorders=dict(fpn=dict(source='neck', type='ModuleOutputs')),
        teacher_recorders=dict(fpn=dict(source='neck', type='ModuleOutputs')),
        type='ConfigurableDistiller'),
    teacher=dict(
        cfg_path=
        'mmdet::swin/mask-rcnn_swin-s-p4-w7_fpn_amp-ms-crop-3x_coco.py',
        pretrained=False),
    teacher_ckpt=
    'https://download.openmmlab.com/mmdetection/v2.0/swin/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth',
    type='FpnTeacherDistill')
optim_wrapper = dict(
    _scope_='mmdet',
    optimizer=dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0001),
    type='OptimWrapper')
param_scheduler = [
    dict(
        _scope_='mmdet',
        begin=0,
        by_epoch=False,
        end=500,
        start_factor=0.001,
        type='LinearLR'),
    dict(
        _scope_='mmdet',
        begin=0,
        by_epoch=True,
        end=24,
        gamma=0.1,
        milestones=[
            16,
            22,
        ],
        type='MultiStepLR'),
]
resume = False
teacher_ckpt = 'https://download.openmmlab.com/mmdetection/v2.0/swin/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth'
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
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
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
    metric=[
        'bbox',
    ],
    type='CocoMetric')
test_pipeline = [
    dict(_scope_='mmdet', backend_args=None, type='LoadImageFromFile'),
    dict(_scope_='mmdet', keep_ratio=True, scale=(
        1333,
        800,
    ), type='Resize'),
    dict(
        _scope_='mmdet',
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True),
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
    _scope_='mmdet', max_epochs=24, type='EpochBasedTrainLoop', val_interval=1)
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
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
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
    sampler=dict(_scope_='mmdet', shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(_scope_='mmdet', backend_args=None, type='LoadImageFromFile'),
    dict(
        _scope_='mmdet',
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True),
    dict(_scope_='mmdet', keep_ratio=True, scale=(
        1333,
        800,
    ), type='Resize'),
    dict(_scope_='mmdet', prob=0.5, type='RandomFlip'),
    dict(_scope_='mmdet', type='PackDetInputs'),
]
val_cfg = dict(type='mmrazor.SingleTeacherDistillValLoop')
val_dataloader = dict(
    batch_size=4,
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
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
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
    num_workers=4,
    persistent_workers=True,
    sampler=dict(_scope_='mmdet', shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    _scope_='mmdet',
    ann_file='data/coco/annotations/instances_val2017.json',
    backend_args=None,
    format_only=False,
    metric=[
        'bbox',
    ],
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
work_dir = './work_dirs/pkd_fpn_mask-rcnn_swin_retina_r50_2x_coco'
