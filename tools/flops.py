import os
import torch
import torch.nn as nn
from pathlib import Path

from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.model import revert_sync_batchnorm
from mmdet.registry import MODELS
from mmdet.structures.det_data_sample import DetDataSample

try:
    from mmengine.analysis import get_model_complexity_info
    from mmengine.analysis.print_helper import _format_size
except ImportError:
    raise ImportError('Please upgrade mmengine >= 0.6.0')

os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # 设置使用GPU

class WrapperModel(nn.Module):
    def __init__(self, model, data_samples):
        super().__init__()
        self.model = model
        self.data_samples = data_samples

    def forward(self, x):
        return self.model(x, self.data_samples)

def compute_flops_params(config_path, input_shape=(3, 512, 512)):
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f'Config file {config_path} not found.')

    cfg = Config.fromfile(config_path)
    init_default_scope(cfg.get('default_scope', 'mmdet'))

    # 构建模型
    model = MODELS.build(cfg.model)
    if torch.cuda.is_available():
        model = model.cuda()
    model = revert_sync_batchnorm(model)
    model.eval()

    dummy_input = torch.randn(1, *input_shape).cuda()

    _, H, W = input_shape
    data_sample = DetDataSample()
    data_sample.set_metainfo(dict(
        img_shape=(H, W, 3),
        ori_shape=(H, W, 3),
        pad_shape=(H, W, 3),
        scale_factor=1.0,
        batch_input_shape=(H, W),
    ))
    data_samples = [data_sample]

    wrapped_model = WrapperModel(model, data_samples).cuda()
    wrapped_model.eval()

    outputs = get_model_complexity_info(
        wrapped_model,
        None,
        inputs=dummy_input,
        show_table=False,
        show_arch=False
    )
    flops = outputs['flops']
    params = outputs['params']

    gflops = flops / 1e9
    flops_str = f"{gflops:.3f} GFLOPs"
    params_str = _format_size(params, sig_figs=6)

    return flops_str, params_str


if __name__ == '__main__':
    config_file = '/mnt/sdb/lx/result/ICFF/hyperparameter/hazydet/tood-icff/tood-icff_r50_hazydet_p2_n0.1/tood-icff_r50_hazydet_p2_n0.1.py'
    input_shape = (3, 1312, 800)
    flops, params = compute_flops_params(config_file, input_shape)
    print(f"FLOPs: {flops}")
    print(f"Params: {params}")
    print(str(round(float(params[: -1]), 1)) + '\t' + str(round(float(flops.split(' ')[0]), 1)))

