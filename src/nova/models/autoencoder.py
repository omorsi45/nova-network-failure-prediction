from __future__ import annotations

import torch
from torch import nn


class AutoEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden: list[int], latent_dim: int):
        super().__init__()
        enc: list[nn.Module] = []
        d = input_dim
        for h in hidden:
            enc += [nn.Linear(d, h), nn.ReLU()]
            d = h
        enc += [nn.Linear(d, latent_dim)]
        self.encoder = nn.Sequential(*enc)

        dec: list[nn.Module] = []
        d = latent_dim
        for h in reversed(hidden):
            dec += [nn.Linear(d, h), nn.ReLU()]
            d = h
        dec += [nn.Linear(d, input_dim)]
        self.decoder = nn.Sequential(*dec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)
