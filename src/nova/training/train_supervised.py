from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

from nova.models.losses import bce_with_logits
from nova.models.supervised import MLPClassifier


@dataclass
class TrainConfig:
    lr: float = 1e-3
    epochs: int = 10
    patience: int = 3


def train_classifier(
    train_loader,
    val_loader,
    input_dim: int,
    hidden: list[int],
    dropout: float,
    cfg: TrainConfig,
    device: str,
) -> tuple[nn.Module, dict]:
    model = MLPClassifier(input_dim=input_dim, hidden=hidden, dropout=dropout).to(device)

    # class imbalance handling
    y_all = []
    for _, y in train_loader:
        y_all.append(y.cpu().numpy())
    y_all = np.concatenate(y_all) if y_all else np.array([0, 1])
    pos = float((y_all == 1).sum())
    neg = float((y_all == 0).sum())
    pos_weight = (neg / max(pos, 1.0)) if pos > 0 else None

    criterion = bce_with_logits(pos_weight=pos_weight).to(device)
    opt = Adam(model.parameters(), lr=cfg.lr)

    best_val = math.inf
    best_state = None
    bad = 0

    for epoch in range(cfg.epochs):
        model.train()
        tr_loss = 0.0
        n = 0
        for x, y in tqdm(train_loader, desc=f"train e{epoch+1}", leave=False):
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            tr_loss += float(loss.item()) * len(x)
            n += len(x)
        tr_loss = tr_loss / max(n, 1)

        model.eval()
        va_loss = 0.0
        vn = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                va_loss += float(loss.item()) * len(x)
                vn += len(x)
        va_loss = va_loss / max(vn, 1) if vn else tr_loss

        if va_loss < best_val - 1e-6:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break

    if best_state:
        model.load_state_dict(best_state)

    metrics = {"val_loss": float(best_val), "pos_weight": float(pos_weight) if pos_weight else None}
    return model, metrics
