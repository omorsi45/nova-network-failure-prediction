from __future__ import annotations

import torch
from torch import nn


def bce_with_logits(pos_weight: float | None = None) -> nn.Module:
    if pos_weight is None:
        return nn.BCEWithLogitsLoss()
    w = torch.tensor([pos_weight], dtype=torch.float32)
    return nn.BCEWithLogitsLoss(pos_weight=w)
