from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

from nova.models.autoencoder import AutoEncoder


@dataclass
class TrainConfig:
    lr: float = 1e-3
    epochs: int = 10
    patience: int = 3


def train_autoencoder(
    train_loader,
    val_loader,
    input_dim: int,
    hidden: list[int],
    latent_dim: int,
    cfg: TrainConfig,
    device: str,
) -> tuple[nn.Module, dict]:
    model = AutoEncoder(input_dim=input_dim, hidden=hidden, latent_dim=latent_dim).to(device)
    opt = Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.MSELoss()

    best_val = math.inf
    best_state = None
    bad = 0

    for epoch in range(cfg.epochs):
        model.train()
        tr_loss = 0.0
        n = 0
        for (x,) in tqdm(train_loader, desc=f"ae train e{epoch+1}", leave=False):
            x = x.to(device)
            opt.zero_grad(set_to_none=True)
            recon = model(x)
            loss = criterion(recon, x)
            loss.backward()
            opt.step()
            tr_loss += float(loss.item()) * len(x)
            n += len(x)
        tr_loss = tr_loss / max(n, 1)

        model.eval()
        va_loss = 0.0
        vn = 0
        with torch.no_grad():
            for (x,) in val_loader:
                x = x.to(device)
                recon = model(x)
                loss = criterion(recon, x)
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

    metrics = {"val_loss": float(best_val)}
    return model, metrics


def anomaly_score(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        recon = model(x)
        return ((recon - x) ** 2).mean(dim=1)
