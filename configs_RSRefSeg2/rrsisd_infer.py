import os
from mmengine.optim import AmpOptimWrapper
from mmengine.runner import EpochBasedTrainLoop
from mmengine.visualization import LocalVisBackend, WandbVisBackend
from torch.optim import AdamW
from mmseg.datasets import LoadAnnotations
from mmseg.visualization import SegLocalVisualizer
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook, LoggerHook, ParamSchedulerHook)
from mmengine.optim.optimizer.optimizer_wrapper import OptimWrapper
from mmengine.optim.scheduler.lr_scheduler import PolyLR, LinearLR, CosineAnnealingLR
from mmengine.runner.loops import IterBasedTrainLoop, TestLoop, ValLoop
from mmseg.engine import SegVisualizationHook
from mmcv.transforms.loading import LoadImageFromFile
from mmcv.transforms.processing import (RandomFlip, RandomResize, Resize, TestTimeAug)
from mmengine.dataset.sampler import DefaultSampler
from mmseg.datasets.transforms.formatting import PackSegInputs
from refseg import RefSegIoUMetric
from refseg.datasets.refdataset import RefSegDataset, LoadSegAnnotations
from refseg.models.models import RefSegEncoderDecoder, RefSegSam, RefSegSiglipTextModel, \
    RefSegSiglipVisionModel, CascadedPrompter, PseudoSegHead

default_scope = 'mmseg'
custom_imports = dict(imports=['refseg'], allow_failed_imports=False)

work_dir = f'./work_dirs/rrsisd'
data_root = f'/mnt/ali-sh-1/dataset/cky_data/RRSIS-D/images/rrsisd/JPEGImages'
cache_dir = f'/mnt/ali-sh-1/usr/chenkeyan/model_cache'



batch_size = 8
max_epochs = 300
val_interval = 10
# sam2.1-hiera-tiny, sam2.1-hiera-small, sam2.1-hiera-base-plus, sam2.1-hiera-large
# facebook/sam-vit-base, facebook/sam-vit-large, facebook/sam-vit-huge
sam_model_name = 'facebook/sam2.1-hiera-large'
# google/siglip-so400m-patch14-224, google/siglip-so400m-patch14-384,
# google/siglip2-so400m-patch14-224, google/siglip2-so400m-patch14-384,
# google/siglip2-so400m-patch16-256, google/siglip2-so400m-patch16-384, google/siglip2-so400m-patch16-512
# google/siglip2-giant-opt-patch16-256, google/siglip2-giant-opt-patch16-384
clip_model_name = 'google/siglip2-so400m-patch16-512'
dst_size = 512 // 16  # 512 is the input size of the clip model, 16 is the patch size, need to be revised if the clip model changes


env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
vis_backends = [
    dict(type=LocalVisBackend),
    # dict(type=WandbVisBackend, init_kwargs=dict(project='RefSeg', group='rrsis-d-l', name=work_dir.split('/')[-1]))
]
visualizer = dict(type=SegLocalVisualizer, vis_backends=vis_backends, name='visualizer', save_dir=work_dir+'/visualizer', alpha=1)
log_processor = dict(by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False


init_from = None

param_scheduler = [
    dict(type=LinearLR, start_factor=0.01, by_epoch=True, begin=0, end=5, convert_to_iter_based=True),
    dict(
        type=CosineAnnealingLR,
        T_max=int(0.9 * max_epochs),
        by_epoch=True,
        begin=int(0.1 * max_epochs),
        eta_min_ratio=0.001,
        end=max_epochs),
]


train_cfg = dict(type=EpochBasedTrainLoop, max_epochs=max_epochs, val_interval=val_interval)
val_cfg = dict(type=ValLoop)
test_cfg = dict(type=TestLoop)

default_hooks = dict(
    timer=dict(type=IterTimerHook),
    logger=dict(type=LoggerHook, interval=20, log_metric_by_epoch=False),
    param_scheduler=dict(type=ParamSchedulerHook),
    checkpoint=dict(type=CheckpointHook, by_epoch=True, interval=val_interval, max_keep_ckpts=5, save_last=True, save_best=['RefSeg/gIoU_1'], rule='greater', greater_keys=['RefSeg/gIoU_1']),
    sampler_seed=dict(type=DistSamplerSeedHook),
    visualization=dict(type=SegVisualizationHook, draw=False, interval=1)
)


crop_size = (1024, 1024)
train_pipeline = [
    dict(type=LoadImageFromFile),
    dict(type=LoadSegAnnotations),
    dict(type=Resize, scale=crop_size, keep_ratio=True),
    dict(type=PackSegInputs,
         meta_keys=('text',
                    'img_path', 'seg_map_path', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'flip_direction', 'reduce_zero_label')
        )
]

test_pipeline = [
    dict(type=LoadImageFromFile),
    dict(type=Resize, scale=crop_size, keep_ratio=True),
    dict(type=LoadSegAnnotations),
    dict(type=PackSegInputs,
         meta_keys=('text', 'category_name',
                    'img_path', 'seg_map_path', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'flip_direction', 'reduce_zero_label')
         )
]

# dataset settings
dataset_type = RefSegDataset
num_workers = 8
persistent_workers = True

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    sampler=dict(type=DefaultSampler, shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='datainfo/rrsisd_train.jsonl',
        pipeline=train_pipeline
    )
)
val_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='datainfo/rrsisd_test.jsonl',
        pipeline=test_pipeline,
        test_mode=True
    )
)
test_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='datainfo/rrsisd_test.jsonl',
        pipeline=test_pipeline,
        test_mode=True,
        # pred_mode=True
    )
)
val_evaluator = dict(type=RefSegIoUMetric)
test_evaluator = val_evaluator


# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[0, 0, 0],
    std=[255., 255., 255.],  # normalize the image in the model internally
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,
    test_cfg=dict(size=crop_size)
)

model = dict(
    type=RefSegEncoderDecoder,
    data_preprocessor=data_preprocessor,
    init_cfg=init_from,
    lora_cfg={
        "backbone.model.image_encoder": dict(
            r=16,
            lora_alpha=16,
            lora_dropout=0.,
            target_modules="^(trunk\.blocks\.\d+\.attn\.(qkv|proj)|neck\.convs\.\d+\.conv)$"
        ),
        "backbone.model.vision_encoder": dict(
            r=16,
            lora_alpha=16,
            lora_dropout=0.,
            target_modules="^(layers\.\d+\.attn\.(qkv|proj)|neck\.conv[12])$"
        ),
        "clip_vision_encoder": dict(
            r=16,
            lora_alpha=16,
            lora_dropout=0.,
            target_modules='^(model\.head\.(attention\.out_proj|mlp\.fc[12])|model\.encoder\.layers\.\d+\.self_attn\.(k_proj|v_proj|q_proj|out_proj))$'
        ),
        "clip_text_encoder": dict(
            r=16,
            lora_alpha=16,
            lora_dropout=0.,
            target_modules='^(model\.head|model\.encoder\.layers\.\d+\.self_attn\.(k_proj|v_proj|q_proj|out_proj))$',
        ),
    },
    backbone=dict(
        type=RefSegSam,
        model_name_or_path=sam_model_name,
        cache_dir=cache_dir,
        with_dense_prompt=True
    ),
    prompter=dict(
        type=CascadedPrompter,
        num_text_queries=3,
        text_query_config=dict(
            has_pe=True,
            init_type='textpool',  # zero, learnable, textpool
        ),
        prompt_config=dict(
            has_pe=True,
            init_type='learnable',  # zero, learnable
        ),

        text_feat_dim=1152,
        two_queries_text_attn_depth=2,
        two_queries_text_attn_operation_order=[
            'self_attn_query_query1', 'norm_query1_1',
            'self_attn_query_query2', 'norm_query2_1',
            'cross_attn_query_text_query1_text', 'norm_query1_2',  'mlp_query1_2', 'norm_query1_3',
            'cross_attn_query_text_query2_text', 'norm_query2_2', 'mlp_query2_2', 'norm_query2_3',
            'cross_attn_query_query_query1_query2', 'norm_query1_4', 'mlp_query1_3', 'norm_query1_5',
            'cross_attn_query_query_query2_query1', 'norm_query2_4','mlp_query2_3', 'norm_query2_5',
            ],

        imgpe_config=dict(type='learnable', size=dst_size), # none, learnable, sine

        queries1_to_img_attn_depth=2,
        queries1_to_img_attn_operation_order=[
            'self_attn_text_text_1', 'norm_text_1',
            'cross_attn_img_text_1', 'norm_img_1', 'mlp_img_1', 'norm_img_2',
        ],

        prompt_queries2_img_attn_depth=2,
        prompt_feat_dim=256,
        prompt_queries2_img_attn_operation_order=[
            'self_attn_text_1', 'norm_text_1',
            'cross_attn_img_text_1','norm_img_1', 'mlp_img_1', 'norm_img_2',
            
            'self_attn_prompt_1', 'norm_prompt_1',
            'cross_attn_prompt_text_1', 'norm_prompt_2', 'mlp_prompt_2', 'norm_prompt_3',
            'cross_attn_prompt_img_1', 'norm_prompt_4', 'mlp_prompt_3', 'norm_prompt_5',
        ],
        num_prompts=9,

        dense_prompt_config=dict(
            proj_from='text_pool',  # prompt, text_pool, text_queries2_pool, refined_text_queries2_pool
            has_internal_proj=False,
            featmap_from='refined_clipfeat', # clipfeat, refined_clipfeat,
            has_internal_conv=True,
            up_strategy='preup', # preup, postup
        ),

        img_feat_dim=1152,
        num_heads=8,
        mlp_dim=512,
        internal_dim=512,
        is_residual=True,
    ),
    clip_vision_encoder=dict(
        type=RefSegSiglipVisionModel,
        model_name_or_path=clip_model_name,
        cache_dir=cache_dir,
        input_size=None,
    ),
    clip_text_encoder=dict(
        type=RefSegSiglipTextModel,
        model_name_or_path=clip_model_name,
        cache_dir=cache_dir
    ),
    decode_head=dict(
        type=PseudoSegHead,
        num_classes=2,
        out_channels=1,
        threshold=0.5,
        align_corners=False,
        loss_decode=[dict(type='mmseg.CrossEntropyLoss', use_sigmoid=True, loss_weight=5.0),
                     dict(type='mmseg.DiceLoss', use_sigmoid=True, loss_weight=5.0)
                     ],
        dst_size=dst_size,
        extra_loss=dict(
            # dense_prompts=1.0,
            text_queries1_queries2=0.5,
            # refined_img_feat_map_perimg=1,
            # refined_img_feat_map_perbatch=0.5,
            clip_visual_feat_map_perbatch=0.5,
            # clip_visual_feat_map_perimg=0.5,

            # use_clip_pool_text=False,
            use_text_queries2=True,

            refined_img_feat_map_perimg_ce=False,
            refined_img_feat_map_perimg_mil=False,

        )
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

base_lr = 0.0001
find_unused_parameters=True


## normal training config
runner_type = 'Runner'
optim_wrapper = dict(
    type=OptimWrapper,
    optimizer=dict(type=AdamW, lr=base_lr, betas=(0.9, 0.999), weight_decay=0.01)
)


# # ### AMP training config
# runner_type = 'Runner'
# optim_wrapper = dict(
#     type=AmpOptimWrapper,
#     dtype='bfloat16',  # float16 if torch.__version__ == 'parrots'
#     optimizer=dict(type=AdamW, lr=base_lr, betas=(0.9, 0.999), weight_decay=0.01)
# )

# ### DeepSpeed training config
# runner_type = 'FlexibleRunner'
# strategy = dict(
#     type='DeepSpeedStrategy',
#     # fp16=dict(
#     #     enabled=True,
#     #     auto_cast=False,
#     #     fp16_master_weights_and_grads=False,
#     #     loss_scale=1024,
#     #     loss_scale_window=1000,
#     #     hysteresis=2,
#     #     min_loss_scale=1,
#     #     initial_scale_power=15,
#     # ),
#     bf16=dict(
#         enabled=True,
#     ),
#     gradient_clipping=3.0,
#     inputs_to_half=['inputs'],
#     zero_optimization=dict(
#         stage=2,
#         allgather_partitions=True,
#         allgather_bucket_size=2e8,
#         reduce_scatter=True,
#         reduce_bucket_size='auto',
#         overlap_comm=True,
#         contiguous_gradients=True,
#     ),
# )
# optim_wrapper = dict(
#     type='DeepSpeedOptimWrapper',
#     optimizer=dict(
#         type='AdamW',
#         lr=base_lr,
#         weight_decay=0.05
#     )
# )