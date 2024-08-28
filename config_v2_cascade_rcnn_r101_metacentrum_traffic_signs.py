# CONFIG for MMDET V2.X 

_base_ = "configs_mmdetv2/cascade_rcnn/cascade_rcnn_r101_fpn_1x_coco.py"
load_from="https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_r101_fpn_1x_coco/cascade_rcnn_r101_fpn_1x_coco_20200317-0b6a2fbf.pth"

WANDB_PROJECT = 'traffic_signs'
DATA_ROOT = '/storage/plzen4-ntis/projects/korpusy_cv/Didymos/Traffic_signs_new/'
NUM_CLASSES = 237

MEAN = [122.48580706, 128.34124017, 123.67939062]
STD =  [ 70.03670754,  70.00521472, 123.67939062]


log_level = 'DEBUG'

log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='MMDetWandbHook',
            init_kwargs=dict(
                group='cascade_rcnn_r101_fpn_1x_coco.py', 
                project=WANDB_PROJECT,
                settings=dict(
                    _service_wait=1000
                )
            ),
            log_checkpoint=True,
            log_checkpoint_metadata=True,
            # num_eval_images=100,
            with_step=False,
            ),
        ],
    interval=10)

checkpoint_config = dict(interval=1)

# MODEL -> modify for n classes
model = dict(
    roi_head=dict(
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                num_classes=NUM_CLASSES
            ),
            dict(
                type='Shared2FCBBoxHead',
                num_classes=NUM_CLASSES
            ),
            dict(
                type='Shared2FCBBoxHead',
                num_classes=NUM_CLASSES
            )
        ]
    )
)


# DATASET and DATALOADERS
# Modify dataset related settings
data_root = DATA_ROOT

dataset_type = 'CocoDataset'
classes = (
    'A10', 'A11', 'A12', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19', 'A1a', 'A1b', 'A22', 'A24', 'A28', 'A29', 'A2a', 'A2b', 'A30', 'A31a', 'A31b', 'A31c', 'A32a', 'A32b', 'A4', 'A5a', 'A6a', 'A6b', 'A7a', 'A8', 'A9', 'B1', 'B11', 'B12', 'B13', 'B14', 'B15', 'B16', 'B17', 'B19', 'B2', 'B20a', 'B20b', 'B21a', 'B21b', 'B24a', 'B24b', 'B26', 'B28', 'B29', 'B32', 'B4', 'B5', 'B6', 'C1', 'C10a', 'C10b', 'C13a', 'C14a', 'C2a', 'C2b', 'C2c', 'C2d', 'C2e', 'C2f', 'C3a', 'C3b', 'C4a', 'C4b', 'C4c', 'C7a', 'C9a', 'C9b', 'E1', 'E11', 'E11c', 'E12', 'E13', 'E2a', 'E2b', 'E2c', 'E2d', 'E3a', 'E3b', 'E4', 'E5', 'E6', 'E7a', 'E7b', 'E8a', 'E8b', 'E8c', 'E8d', 'E8e', 'E9', 'I2', 'IJ1', 'IJ10', 'IJ11a', 'IJ11b', 'IJ14c', 'IJ15', 'IJ2', 'IJ3', 'IJ4a', 'IJ4b', 'IJ4c', 'IJ4d', 'IJ4e', 'IJ5', 'IJ6', 'IJ7', 'IJ8', 'IJ9', 'IP10a', 'IP10b', 'IP11a', 'IP11b', 'IP11c', 'IP11e', 'IP11g', 'IP12', 'IP13a', 'IP13b', 'IP13c', 'IP13d', 'IP14a', 'IP15a', 'IP15b', 'IP16', 'IP17', 'IP18a', 'IP18b', 'IP19', 'IP2', 'IP21', 'IP21a', 'IP22', 'IP25a', 'IP25b', 'IP26a', 'IP26b', 'IP27a', 'IP3', 'IP31a', 'IP4a', 'IP4b', 'IP5', 'IP6', 'IP7', 'IP8a', 'IP8b', 'IS10b', 'IS11a', 'IS11b', 'IS11c', 'IS12a', 'IS12b', 'IS12c', 'IS13', 'IS14', 'IS15a', 'IS15b', 'IS16b', 'IS16c', 'IS16d', 'IS17', 'IS18a', 'IS18b', 'IS19a', 'IS19b', 'IS19c', 'IS19d', 'IS1a', 'IS1b', 'IS1c', 'IS1d', 'IS20', 'IS21a', 'IS21b', 'IS21c', 'IS22a', 'IS22c', 'IS22d', 'IS22e', 'IS22f', 'IS23', 'IS24a', 'IS24b', 'IS24c', 'IS2a', 'IS2b', 'IS2c', 'IS2d', 'IS3a', 'IS3b', 'IS3c', 'IS3d', 'IS4a', 'IS4b', 'IS4c', 'IS4d', 'IS5', 'IS6a', 'IS6b', 'IS6c', 'IS6e', 'IS6f', 'IS6g', 'IS7a', 'IS8a', 'IS8b', 'IS9a', 'IS9b', 'IS9c', 'IS9d', 'O2', 'P1', 'P2', 'P3', 'P4', 'P6', 'P7', 'P8', 'UNKNOWN', 'X1', 'X2', 'X3', 'XXX', 'Z2', 'Z3', 'Z4a', 'Z4b', 'Z4c', 'Z4d', 'Z4e', 'Z7', 'Z9'
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(
        1333,
        800,
    ), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0), # has to be here, otherwise mmdet sceams key not found flip -> just set to 0 then
    dict(
        type='Normalize',
        mean=MEAN,
        std=STD,
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=[
        'img',
        'gt_bboxes',
        'gt_labels',
    ]),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(
            1333,
            800,
        ),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(
                type='Normalize',
                mean=MEAN,
                std=STD,
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=[
                'img',
            ]),
            dict(type='Collect', keys=[
                'img',
            ]),
        ]),
]

data = dict(
    # TODO: maybe 2 workers here? 
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        ann_file=DATA_ROOT+'metadata_train_enriched.json',
        img_prefix=DATA_ROOT+'images/',
        classes=classes,
        pipeline=train_pipeline
    ),
    val=dict(
        ann_file=DATA_ROOT+'metadata_val_enriched.json',
        img_prefix=DATA_ROOT+'images/',
        classes=classes,
        pipeline=test_pipeline
    ),
    test=dict(
        ann_file='signs_metadata_val_enriched.json',
        img_prefix='/media/tomas/samQVO_4TB_D/signs_new/images',
        classes=classes,
        pipeline=test_pipeline
    ),
)
dist_params=None

# Modify metric related settings
workflow = [('train', 1), ('val', 1)]