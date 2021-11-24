_base_ = [
    '../../_base_/default_runtime.py', '../../_base_/recog_models/crnn.py',
    '../../_base_/recog_pipelines/crnn_pipeline.py',
    '../../_base_/recog_datasets/academic_synthetic_trainset_v2.py',
    '../../_base_/recog_datasets/academic_testset.py',
    '../../_base_/schedules/schedule_adadelta_fix_5e.py'
]

train_list = {{_base_.train_list}}
test_list = {{_base_.test_list}}

train_pipeline = {{_base_.train_pipeline}}
test_pipeline = {{_base_.test_pipeline}}

data = dict(
    samples_per_gpu=64,
    workers_per_gpu=4,
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='UniformConcatDataset',
        datasets=train_list,
        pipeline=train_pipeline),
    val=dict(
        type='UniformConcatDataset',
        datasets=test_list,
        pipeline=test_pipeline),
    test=dict(
        type='UniformConcatDataset',
        datasets=test_list,
        pipeline=test_pipeline))

evaluation = dict(interval=1, metric='acc')

cudnn_benchmark = True
