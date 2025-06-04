# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .fcos import FCOS
from torch import Tensor


@MODELS.register_module()
class FCOS_ICFF(FCOS):
    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 filter: ConfigType,
                 bbox_head: ConfigType,
                 with_filter=True,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        self.with_filter = with_filter
        if with_filter:
            self.filter = MODELS.build(filter)

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        x = self.backbone(batch_inputs)
        if self.with_neck:
            x = self.neck(x)
        if self.with_filter:
            x = self.filter(x)
        return x