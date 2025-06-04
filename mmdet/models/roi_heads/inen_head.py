import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from mmdet.registry import MODELS

@MODELS.register_module()
class INENHead(nn.Module):
    def __init__(self,
                 in_channels=256,
                 hidden_channels=256,
                 num_classes=3,
                 loss=None):
        super().__init__()
        self.fc_embed = nn.Linear(in_channels, hidden_channels)
        self.gcn1 = GCNConv(hidden_channels, hidden_channels)
        self.gcn2 = GCNConv(hidden_channels, hidden_channels)
        self.cls = nn.Linear(hidden_channels, num_classes)

        if loss is not None:
            self.loss_fn = MODELS.build(loss)
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, edge_index):
        x = F.relu(self.fc_embed(x))
        x = self.gcn1(x, edge_index)
        x = self.gcn2(x, edge_index)
        logits = self.cls(x)
        return logits

    def loss(self, logits, labels):
        return self.loss_fn(logits, labels)
