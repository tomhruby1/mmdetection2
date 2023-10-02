# CONFIG for MMDET V2.X 

_base_ = "/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_fp16_1x_coco.py"

WANDB_PROJECT = 'faster_rcnn_experiment'
DATA_ROOT = 'coco_instance_seg_data/'
NUM_CLASSES = 3
log_level = 'DEBUG'

log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='MMDetWandbHook',
            init_kwargs=dict(
                group='faster-rcnn-r50-fpn16-1x-coco', project=WANDB_PROJECT),
            log_checkpoint=True,
            interval=1,
            log_checkpoint_metadata=True,
            num_eval_images=100,
            with_step=False),
    ],
    interval=10)

checkpoint_config = dict(interval=1)

# MODEL -> modify for n classes
model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=NUM_CLASSES
        )
    )
)


# DATASET and DATALOADERS
# Modify dataset related settings
data_root = DATA_ROOT

dataset_type = 'CocoDataset'
classes = (
            "human_face", 
            "human_face_behind_glass",
            "human_face_with_mask"
)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        ann_file=DATA_ROOT+'train/_annotations_faces.coco.json',
        img_prefix=DATA_ROOT+'train/',
        classes=classes
    ),
    val=dict(
        ann_file=DATA_ROOT+'valid/_annotations_faces.coco.json',
        img_prefix=DATA_ROOT+'valid/',
        classes=classes
    ),
    test=dict(
        ann_file=DATA_ROOT+'test/_annotations_faces.coco.json',
        img_prefix=DATA_ROOT+'test/',
        classes=classes
    )
)
dist_params=None

# Modify metric related settings
workflow = [('train', 1), ('val', 1)]
load_from='https://download.openmmlab.com/mmdetection/v2.0/fp16/faster_rcnn_r50_fpn_fp16_1x_coco/faster_rcnn_r50_fpn_fp16_1x_coco_20200204-d4dc1471.pth'