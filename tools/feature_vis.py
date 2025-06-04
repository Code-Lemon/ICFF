import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'  # 设置使用GPU
data = 'hazydet'
# data = 'visdrone2019'
import copy
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.dist import get_rank, get_world_size
from mmengine.model import revert_sync_batchnorm
from mmdet.registry import MODELS
from mmdet.structures.det_data_sample import DetDataSample
from mmengine.registry import DATASETS, DATA_SAMPLERS, FUNCTIONS
from mmdet.visualization.local_visualizer import DetLocalVisualizer
from mmengine.visualization import Visualizer
import mmcv


from mmengine.config import Config
from mmdet.registry import MODELS
from mmengine.runner import load_checkpoint


def load_model(config_file, checkpoint_file, device):
    cfg = Config.fromfile(config_file)
    init_default_scope(cfg.get('default_scope', 'mmdet'))
    model = MODELS.build(cfg.model)
    load_checkpoint(model, checkpoint_file, map_location='cpu')
    model.cuda()
    model.to(device)
    return model

def remove_pad_predictions(results):
    data_sample = copy.deepcopy(results[0])
    img_shape = data_sample.metainfo['img_shape']
    pad_shape = data_sample.metainfo['pad_shape']
    pad_top, pad_left = (pad_shape[0] - img_shape[0]) / 2, (pad_shape[1] - img_shape[1]) / 2
    data_sample.pred_instances.bboxes[:, [0, 2]] -= pad_left
    data_sample.pred_instances.bboxes[:, [1, 3]] -= pad_top
    # sf = data_sample.metainfo['scale_factor']
    # if len(sf) == 2:
    #     sf = [sf[0], sf[1], sf[0], sf[1]]
    # sf = torch.tensor(sf, device=data_sample.pred_instances.bboxes.device)
    # data_sample.pred_instances.bboxes /= sf[None, :]
    return data_sample

def build_dataloader(dataloader, seed=None, diff_rank_seed=False) -> DataLoader:
    if isinstance(dataloader, DataLoader):
        return dataloader

    dataloader_cfg = copy.deepcopy(dataloader)

    # build dataset
    dataset_cfg = dataloader_cfg.pop('dataset')
    dataset = DATASETS.build(dataset_cfg)
    if hasattr(dataset, 'full_init'):
        dataset.full_init()

    # build sampler
    sampler_cfg = dataloader_cfg.pop('sampler')
    sampler_seed = None if diff_rank_seed else seed
    sampler = DATA_SAMPLERS.build(
        sampler_cfg,
        default_args=dict(dataset=dataset, seed=sampler_seed))

    # build batch sampler
    batch_sampler_cfg = dataloader_cfg.pop('batch_sampler', None)
    if batch_sampler_cfg is None:
        batch_sampler = None
    elif isinstance(batch_sampler_cfg, dict):
        batch_sampler = DATA_SAMPLERS.build(
            batch_sampler_cfg,
            default_args=dict(
                sampler=sampler,
                batch_size=dataloader_cfg.pop('batch_size')))

    if 'worker_init_fn' in dataloader_cfg:
        worker_init_fn_cfg = dataloader_cfg.pop('worker_init_fn')
        worker_init_fn_type = worker_init_fn_cfg.pop('type')
        if isinstance(worker_init_fn_type, str):
            worker_init_fn = FUNCTIONS.get(worker_init_fn_type)
    elif seed is not None:
        init_fn = partial(
            FUNCTIONS.get('default_worker_init_fn'),
            num_workers=dataloader_cfg.get('num_workers'),
            rank=get_rank(),
            seed=seed,
            disable_subprocess_warning=False
        )
    else:
        init_fn = None

    collate_fn_cfg = dataloader_cfg.pop('collate_fn', dict(type='pseudo_collate'))
    if isinstance(collate_fn_cfg, dict):
        collate_fn_type = collate_fn_cfg.pop('type')
        if isinstance(collate_fn_type, str):
            collate_fn = FUNCTIONS.get(collate_fn_type)
        else:
            collate_fn = collate_fn_type
        collate_fn = partial(collate_fn, **collate_fn_cfg)  # type: ignore
    else:
        collate_fn = collate_fn_cfg

    data_loader = DataLoader(
        dataset=dataset,
        sampler=sampler if batch_sampler is None else None,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
        worker_init_fn=init_fn,
        **dataloader_cfg)
    return data_loader
# ------------------------
# 1. 加载模型
if data == 'hazydet':
    root_path = '/mnt/sdb/lx/result/ICFF/visualizer/predict/hazydet'

    tood_icff_config_file = "/mnt/sdb/lx/result/ICFF/hazydet/tood_icff/tood-icff_r50_hazydet-1000/tood-icff_r50_hazydet.py"
    tood_icff_checkpoint_file = "/mnt/sdb/lx/result/ICFF/hazydet/tood_icff/tood-icff_r50_hazydet-1000/epoch_30.pth"

    cascade_config_file = '/home/b1/lx/mmdetection-main/my_config/ICFF/hazydet/cascade-rcnn/cascade-rcnn_r50_hazydet.py'
    cascade_checkpoint_file = '/mnt/sdb/lx/result/ICFF/hazydet/cascade-rcnn/cascade-rcnn_r50_hazydet/epoch_15.pth'

    tood_config_file = '/mnt/sdb/lx/result/ICFF/hazydet/tood/tood_r50_hazydet/tood_r50_hazydet.py'
    tood_checkpoint_file = '/mnt/sdb/lx/result/ICFF/hazydet/tood/tood_r50_hazydet/epoch_15.pth'

    conditional_detr_config_file = '/mnt/sdb/lx/result/ICFF/hazydet/conditional-detr/conditional-detr_r50_hazydet/conditional-detr_hazydet.py'
    conditional_detr_checkpoint_file = '/mnt/sdb/lx/result/ICFF/hazydet/conditional-detr/conditional-detr_r50_hazydet/epoch_100.pth'

    BAF_config_file = '/mnt/sdb/lx/result/ICFF/hazydet/CascadeRCNN_BAF/CascadeRCNN_BAF_r50_hazydet/cascadercnn-baf_r50_hazydet.py'
    BAF_checkpoint_file = '/mnt/sdb/lx/result/ICFF/hazydet/CascadeRCNN_BAF/CascadeRCNN_BAF_r50_hazydet/epoch_15.pth'
elif data == 'visdrone2019':
    root_path = '/mnt/sdb/lx/result/ICFF/visualizer/predict/visdrone2019_test'

    tood_icff_config_file = "/mnt/sdb/lx/result/ICFF/visdrone2019/tood-icff/tood-icff_r50_visdrone2019-1000/tood-icff_r50_visdrone2019.py"
    tood_icff_checkpoint_file = "/mnt/sdb/lx/result/ICFF/visdrone2019/tood-icff/tood-icff_r50_visdrone2019-500/epoch_30.pth"

    cascade_config_file = '/mnt/sdb/lx/result/ICFF/visdrone2019/cascade-rcnn/cascade-rcnn_r50_visdrone2019/cascade-rcnn_r50_visdrone2019.py'
    cascade_checkpoint_file = '/mnt/sdb/lx/result/ICFF/visdrone2019/cascade-rcnn/cascade-rcnn_r50_visdrone2019/epoch_15.pth'

    tood_config_file = '/mnt/sdb/lx/result/ICFF/visdrone2019/tood/tood_r50_visdrone2019/tood_r50_visdrone2019.py'
    tood_checkpoint_file = '/mnt/sdb/lx/result/ICFF/visdrone2019/tood/tood_r50_visdrone2019/epoch_15.pth'

    conditional_detr_config_file = '/mnt/sdb/lx/result/ICFF/visdrone2019/conditional-detr/conditional-detr_r50_visdrone2019/conditional-detr_visdrone2019.py'
    conditional_detr_checkpoint_file = '/mnt/sdb/lx/result/ICFF/visdrone2019/conditional-detr/conditional-detr_r50_visdrone2019/epoch_100.pth'

    BAF_config_file = '/mnt/sdb/lx/result/ICFF/visdrone2019/CascadeRCNN_BAF/CascadeRCNN_BAF_r50_visdrone2019/cascadercnn-baf_r50_visdrone2019.py'
    BAF_checkpoint_file = '/mnt/sdb/lx/result/ICFF/visdrone2019/CascadeRCNN_BAF/CascadeRCNN_BAF_r50_visdrone2019/epoch_15.pth'


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
models = ['tood_icff', 'cascade', 'tood', 'conditional_detr', 'BAF', 'gt']
save_path = []
for i in range(len(models)):
    save_path.append(os.path.join(root_path, models[i]))
    os.makedirs(os.path.join(root_path, models[i]), exist_ok=True)
os.makedirs(os.path.join(root_path, 'image'), exist_ok=True)

tood_icff = load_model(tood_icff_config_file, tood_icff_checkpoint_file, device)
cascade = load_model(cascade_config_file, cascade_checkpoint_file, device)
tood = load_model(tood_config_file, tood_checkpoint_file, device)
conditional_detr = load_model(conditional_detr_config_file, conditional_detr_checkpoint_file, device)
BAF = load_model(BAF_config_file, BAF_checkpoint_file, device)

# # 查看所有模块名称（递归结构）
# for name, module in model.named_modules():
#     print(name)

# ------------------------
# # 2. 注册 hook：提取 neck.fpn_convs.1 输出
# features = {}
#
# def hook_fn(module, input, output):
#     features['feat'] = output.detach()
#
# # 修改这里的路径即可
# target_layer = model.backbone.layer4[2].relu
# handle = target_layer.register_forward_hook(hook_fn)


# -------------------------
# 3. 构建测试集和 dataloader
cfg = Config.fromfile(tood_icff_config_file)
cfg.test_dataloader.batch_size = 1
test_loader = build_dataloader(cfg.test_dataloader)
# 初始化可视化器
visualizer = DetLocalVisualizer()
visualizer.dataset_meta = test_loader.dataset.metainfo  # 设置类名等元信息

# ---------- 取一张图像进行推理 ----------
count = 0
for idx, data_batch in enumerate(test_loader):
    if count not in [81, 224, 228]:
        count += 1
        continue
    img_path = data_batch['data_samples'][0].metainfo['img_path']
    original_img = mmcv.imread(img_path)
    tood_icff_processed = tood_icff.data_preprocessor(data_batch, training=False)
    cascade_processed = cascade.data_preprocessor(data_batch, training=False)
    tood_processed = tood.data_preprocessor(data_batch, training=False)
    conditional_detr_processed = conditional_detr.data_preprocessor(data_batch, training=False)
    BAF_processed = BAF.data_preprocessor(data_batch, training=False)
    with torch.no_grad():
        tood_icff_results = remove_pad_predictions(tood_icff.forward(**tood_icff_processed, mode='predict'))
        cascade_results = remove_pad_predictions(cascade.forward(**cascade_processed, mode='predict'))
        tood_results = remove_pad_predictions(tood.forward(**tood_processed, mode='predict'))
        conditional_detr_results = remove_pad_predictions(conditional_detr.forward(**conditional_detr_processed, mode='predict'))
        BAF_results = remove_pad_predictions(BAF.forward(**BAF_processed, mode='predict'))
    result = [tood_icff_results, cascade_results, tood_results, conditional_detr_results, BAF_results]
    for i in range(len(models) - 1):
        visualizer.add_datasample(
            name='result',
            image=original_img,
            data_sample=result[i],
            draw_gt=False,
            draw_pred=True,
            out_file=os.path.join(save_path[i], str(count) + '.jpg'),
            show=False,
            pred_score_thr=0.4,
        )
    visualizer.add_datasample(
        name='result',
        image=data_batch['inputs'][0].permute(1, 2, 0).cpu().numpy(),
        # image=original_img,
        data_sample=result[i],
        draw_gt=True,
        draw_pred=False,
        out_file=os.path.join(save_path[-1], str(count) + '.jpg'),
        show=False,
        pred_score_thr=0.3,
    )
    visualizer.add_datasample(
        name='result',
        image=original_img,
        data_sample=result[i],
        draw_gt=False,
        draw_pred=False,
        out_file=os.path.join(root_path, 'image', str(count) + '.jpg'),
        show=False,
        pred_score_thr=0.3,
    )
    count += 1
    print(count)

# # ---------- 可视化特征图 ----------
# feat = features['feat'][0].cpu()  # shape: [C, H, W]
# channel_idx = 10  # 可自定义通道编号
# feature = feat[channel_idx].cpu().numpy()
# feature = (feature - feature.min()) / (feature.max() - feature.min())