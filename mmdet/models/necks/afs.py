import torch
import torch.nn as nn
from mmdet.registry import MODELS
from .afs_modules.sam import SAM
from .afs_modules.cam import CAM

@MODELS.register_module()
class AFSNeck(nn.Module):
    def __init__(self,
                 in_channels,             # List[int], 输入每层特征图的通道数
                 sam_indices=[(0, 2), (1, 3)],
                 with_cam=True,
                 num_classes=20):
        super().__init__()
        self.sam_indices = sam_indices
        self.with_cam = with_cam

        # 构建 SAM 模块（成对输入）
        self.sam_blocks = nn.ModuleList()
        for (i, j) in sam_indices:
            self.sam_blocks.append(SAM(low_channels=in_channels[i],
                                       high_channels=in_channels[j]))

        # 构建 CAM 模块（作用于所有层）
        if with_cam:
            self.cam_blocks = nn.ModuleList([
                CAM(in_channels=c, num_classes=num_classes)
                for c in in_channels
            ])

    def forward(self, feats):
        # feats: Tuple of feature maps
        feats = list(feats)  # 转为可赋值 list
        for idx, (i, j) in enumerate(self.sam_indices):
            assert i < len(feats) and j < len(feats), \
                f"Invalid sam_indices: ({i}, {j}) for feats length {len(feats)}"
            f_low, f_high = feats[i], feats[j]
            new_low, new_high = self.sam_blocks[idx](f_low, f_high)
            feats[i] = new_low
            feats[j] = new_high
        if self.with_cam:
            feats = [cam(f) for cam, f in zip(self.cam_blocks, feats)]
        return tuple(feats)

