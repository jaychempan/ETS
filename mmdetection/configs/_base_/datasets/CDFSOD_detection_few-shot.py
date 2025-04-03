# dataset settings
dataset_type = 'CocoDataset'

############ on pub dataset ##############
# data_root = '../data/ArTaxOr/'

# metainfo = dict(
# classes = (
#     "Araneae",
#     "Coleoptera",
#     "Diptera",
#     "Hemiptera",
#     "Hymenoptera",
#     "Lepidoptera",
#     "Odonata",
# )
# )
# data_root = '../data/clipart1k/'

# metainfo = dict(
# classes = (
#     "sheep",
#     "chair",
#     "boat",
#     "bottle",
#     "diningtable",
#     "sofa",
#     "cow",
#     "motorbike",
#     "car",
#     "aeroplane",
#     "cat",
#     "train",
#     "person",
#     "bicycle",
#     "pottedplant",
#     "bird",
#     "dog",
#     "bus",
#     "tvmonitor",
#     "horse"
# )
# )

# data_root = '../data/FISH/'

# metainfo = dict(
# classes = (
#     "fish",
# )
# )


############ on pub dataset ##############

########### on unsee dataset ##############
# data_root = '../data/dataset1/'
# data_root = '../data/dataset2/'
# data_root = '../data/dataset3/'
# data_root = '../data/dataset3_augmented/'

# metainfo = dict(
# classes = ("Araneae", "Coleoptera", "Diptera", "Hemiptera", "Hymenoptera", "Lepidoptera", "Odonata") # ArTaxOr
# )

# metainfo = dict(
# classes = ("apple", "avocado", "capsicum", "mango", "orange", "rockmelon", "strawberry") ## dataset1
# )

# metainfo = dict(
# classes = ("car") ## dataset2
# )

# metainfo = dict(
# classes = ("dent", "scratch", "crack", "glass shatter", "lamp broken", "tire flat") ## dataset3
# )
########### on unsee dataset ##############

# train_ann_file = 'annotations/1_shot.json'
train_ann_file = 'annotations/5_shot.json'
# train_ann_file = 'annotations/10_shot.json'



############################
backend_args = None

# # dataset settings
# train_pipeline = [
#     dict(type='LoadImageFromFile', backend_args=backend_args),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(
#         type='PackDetInputs',
#         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
#                    'scale_factor', 'flip', 'flip_direction', 'text',
#                    'custom_entities'))
# ]

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='CachedMosaic', img_scale=(640, 640), pad_val=114.0, prob=0.6),
    # dict(type='CopyPaste', max_num_pasted=5, paste_by_box=True),  # 添加 CopyPaste 数据增强
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='CachedMixUp',
        img_scale=(640, 640),
        ratio_range=(1.0, 1.0),
        max_cached_images=10,
        pad_val=(114, 114, 114),
        prob = 0.3),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    # The radio of all image in train dataset < 7
                    # follow the original implement
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ]
        ]),
    # dict(type='RandomErasing', n_patches=(0,2), ratio=0.3, img_border_value=128, bbox_erased_thr=0.9),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities'))
]


# train_pipeline = [
#     dict(type='LoadImageFromFile', backend_args=backend_args),
#     dict(type='LoadAnnotations', with_bbox=True),
#     # dict(type='CachedMosaic', img_scale=(640, 640), pad_val=114.0),
#     # dict(
#     #     type='RandomResize',
#     #     scale=(1280, 1280),
#     #     ratio_range=(0.5, 2.0),
#     #     keep_ratio=True),
#     # dict(type='RandomCrop', crop_size=(640, 640)),
#     # dict(type='YOLOXHSVRandomAug'),
#     # dict(type='RandomFlip', prob=0.5),
#     # dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
#     # # dict(
#     #     type='CachedMixUp',
#     #     img_scale=(640, 640),
#     #     ratio_range=(1.0, 1.0),
#     #     max_cached_images=20,
#     #     pad_val=(114, 114, 114)),
#     dict(type='PackDetInputs',
#         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
#                    'scale_factor', 'flip', 'flip_direction', 'text',
#                    'custom_entities'))
# ]


# train_pipeline = [ 
#     dict(type='Resize', scale=(1024, 1024), keep_ratio=True),  # 先 Resize
#     dict(type='RandomFlip', prob=0.5),
#     dict(type='CopyPaste', max_num_pasted=30, paste_by_box=True),  # 添加 CopyPaste 数据增强
#     dict(type='Pad', size_divisor=32),  # 统一 mask 形状
#     # dict(
#     #     type='RandomAffine',
#     #     scaling_ratio_range=(0.1, 2),
#     #     border=(-img_scale[0] // 2, -img_scale[1] // 2)), # The image will be enlarged by 4 times after Mosaic processing,so we use affine transformation to restore the image size.
#     dict(type='PackDetInputs',
#         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
#                 'scale_factor', 'text', 'custom_entities'))
# ]

# img_scale=(640, 640)
# train_pipeline = [
#     dict(type='LoadImageFromFile', backend_args=backend_args),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(type='CachedMosaic', img_scale=img_scale, pad_val=114.0),
#     dict(type='YOLOXHSVRandomAug'),
#     # dict( 
#     # type='RandomAffine', 
#     # scaling_ratio_range=(0.1, 2), 
#     # border=(640 // 2, 640 // 2)), 
#     # dict(
#     #     type='CachedMixUp',
#     #     img_scale=img_scale,
#     #     ratio_range=(1.0, 1.0),
#     #     max_cached_images=20,
#     #     pad_val=(114, 114, 114)),
#     dict(type='RandomFlip', prob=0.5),
#     dict(
#         type='RandomChoice',
#         transforms=[
#             [
#                 dict(
#                     type='RandomChoiceResize',
#                     scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
#                             (608, 1333), (640, 1333), (672, 1333), (704, 1333),
#                             (736, 1333), (768, 1333), (800, 1333)],
#                     keep_ratio=True)
#             ],
#             [
#                 dict(
#                     type='RandomChoiceResize',
#                     # The radio of all image in train dataset < 7
#                     # follow the original implement
#                     scales=[(400, 4200), (500, 4200), (600, 4200)],
#                     keep_ratio=True),
#                 dict(
#                     type='RandomCrop',
#                     crop_type='absolute_range',
#                     crop_size=(384, 600),
#                     allow_negative_crop=True),
#                 dict(
#                     type='RandomChoiceResize',
#                     scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
#                             (608, 1333), (640, 1333), (672, 1333), (704, 1333),
#                             (736, 1333), (768, 1333), (800, 1333)],
#                     keep_ratio=True)
#             ]
#         ]),
#     dict(
#         type='PackDetInputs',
#         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
#                    'scale_factor', 'flip', 'flip_direction', 'text',
#                    'custom_entities'))
# ]



test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='FixScaleResize', scale=(800, 1333), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities'))
]


################################## 使用ConcatDataset ##########################################################################
train_real_dataset=dict(
    type=dataset_type,
    data_root=data_root,
    ann_file=train_ann_file,
    metainfo=metainfo,
    data_prefix=dict(img='train/'),
    pipeline=train_pipeline,
    filter_cfg=dict(filter_empty_gt=False),
    return_classes=True)

# train_fake_dataset=dict(
#     type=dataset_type,
#     data_root=data_root,
#     ann_file=train_ann_file_fake,
#     metainfo=metainfo,
#     data_prefix=dict(img='test/'),
#     filter_cfg=dict(filter_empty_gt=False),
#     pipeline=train_pipeline,
#     return_classes=True)

# all_train_dataset = [train_real_dataset]

# train_dataloader = dict(
#     batch_size=2,
#     num_workers=8,
#     persistent_workers=True,
#     sampler=dict(type='DefaultSampler', shuffle=True),
#     batch_sampler=dict(type='AspectRatioBatchSampler'),
#     dataset=dict(type='ConcatDataset', 
#                  datasets=all_train_dataset))
################################## 使用ConcatDataset ##########################################################################

################################## 使用MultiImageMixDataset ##########################################################################

# train_real_dataset = dict(
#     type='MultiImageMixDataset',
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file=train_ann_file,
#         data_prefix=dict(img='train/'),
#         pipeline=[
#             dict(type='LoadImageFromFile', backend_args=backend_args),
#             dict(type='LoadAnnotations', with_bbox=True),
#         ],
        
#         filter_cfg=dict(filter_empty_gt=False),
#         return_classes=True,
#     ),
#     pipeline=train_pipeline,
    
# )

################################## 使用MultiImageMixDataset ##########################################################################

train_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=train_real_dataset,
    )


val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        # ann_file='annotations/10_shot.json',
        ann_file='annotations/test.json',
        data_prefix=dict(img='test/'),
        test_mode=True,
        metainfo=metainfo,
        pipeline=test_pipeline,
        return_classes=True))

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/test.json',
        data_prefix=dict(img='test/'),
        test_mode=True,
        metainfo=metainfo,
        pipeline=test_pipeline,
        return_classes=True))

val_evaluator = dict(
    type='CocoMetric',
    # ann_file=data_root + 'annotations/10_shot.json',
    ann_file=data_root + 'annotations/test.json',
    metric='bbox',
    classwise=True,
    format_only=False,
    backend_args=backend_args)


test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/test.json',
    metric='bbox',
    classwise=True,
    format_only=False,
    backend_args=backend_args)