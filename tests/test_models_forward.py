from __future__ import annotations

import torch

from nova.models.supervised import MLPClassifier
from nova.models.autoencoder import AutoEncoder


def test_models_forward():
    x = torch.randn(16, 20)
    clf = MLPClassifier(input_dim=20, hidden=[32, 16], dropout=0.1)
    logits = clf(x)
    assert logits.shape == (16,)

    ae = AutoEncoder(input_dim=20, hidden=[16, 8], latent_dim=4)
    recon = ae(x)
    assert recon.shape == x.shape
