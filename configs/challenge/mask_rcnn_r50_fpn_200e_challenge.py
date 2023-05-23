_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py',
    'deepsportradar_instances.py',
    '../_base_/schedules/schedule_200e.py', '../_base_/default_runtime.py'
]
