import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import argparse
import torch
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmcv.cnn.utils.flops_counter import get_model_complexity_info
from mmdet.registry import MODELS, DATASETS
from torch.utils.data import DataLoader
from mmengine.registry import init_default_scope, DATASETS
from mmengine.dataset import pseudo_collate
from mmengine.runner import Runner

def parse_args():
    parser = argparse.ArgumentParser(description='Unified MMDetection Test & Analyze Script')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint')
    parser.add_argument('--device', default='cuda:0', help='Device for inference')
    parser.add_argument('--eval', default='bbox', help='Evaluation metric (bbox, segm, proposal, etc.)')
    return parser.parse_args()

def print_flops_and_params(cfg):
    model = MODELS.build(cfg.model)
    model.eval()
    input_shape = (3, 512, 512)
    with torch.no_grad():
        flops, params = get_model_complexity_info(
            model, input_shape, as_strings=True, print_per_layer_stat=False)
        print(f"\n[Backbone]: {cfg.model.backbone.type}")
        print(f"[GFLOPs]: {flops}")
        print(f"[Params]: {params}")

def run_evaluation(cfg, checkpoint_path, eval_metric, device):
    cfg.device = device
    init_default_scope(cfg.get('default_scope', 'mmdet'))

    model = MODELS.build(cfg.model)
    load_checkpoint(model, checkpoint_path, map_location=device)
    model.to(device)
    model.eval()

    cfg.test_dataloader.dataset.test_mode = True
    dataset = DATASETS.build(cfg.test_dataloader.dataset)

    # 自定义 DataLoader 以适配 MMEngine Runner 接口
    data_loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=pseudo_collate  # 添加这一行
    )

    outputs = []
    prog_bar = range(len(data_loader))
    for data in data_loader:
        with torch.no_grad():
            result = model.test_step(data)
        outputs.extend(result)

    eval_results = dataset.evaluate(outputs, metric=eval_metric)
    print("\n[Evaluation Results]:")
    for k, v in eval_results.items():
        print(f"{k}: {v}")

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    print("==== Step 1: Analyzing FLOPs and Params ====")
    print_flops_and_params(cfg)

    print("\n==== Step 2: Running Evaluation ====")
    run_evaluation(cfg, args.checkpoint, args.eval, args.device)

if __name__ == '__main__':
    main()
