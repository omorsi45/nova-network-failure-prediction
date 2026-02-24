from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from nova.common.io import ensure_dir, read_df
from nova.models.autoencoder import AutoEncoder
from nova.models.supervised import MLPClassifier
from nova.training.train_unsupervised import anomaly_score


def _load_scaler(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=str, default="data/processed")
    ap.add_argument("--artifacts", type=str, default="artifacts")
    args = ap.parse_args()

    processed = Path(args.processed)
    art = Path(args.artifacts)
    reports = ensure_dir(art / "reports")
    models_dir = art / "models"

    df = read_df(processed / "dataset.parquet")
    feature_cols = json.loads((processed / "feature_cols.json").read_text(encoding="utf-8"))
    X = df[feature_cols].to_numpy(dtype=np.float32)

    out = df[["node_id", "window_start"]].copy()

    # supervised probability
    sup_path = models_dir / "supervised.pt"
    if sup_path.exists():
        sup = torch.load(sup_path, map_location="cpu")
        scaler = _load_scaler(models_dir / "supervised_scaler.pkl")
        Xs = scaler.transform(X)
        model = MLPClassifier(input_dim=len(feature_cols), hidden=list(sup["model"]["hidden"]), dropout=float(sup["model"]["dropout"]))
        model.load_state_dict(sup["state_dict"])
        model.eval()
        with torch.no_grad():
            prob = torch.sigmoid(model(torch.tensor(Xs, dtype=torch.float32))).numpy()
        out["risk_supervised"] = prob
    else:
        out["risk_supervised"] = 0.0

    # unsupervised anomaly score normalized
    ae_path = models_dir / "unsupervised_ae.pt"
    if ae_path.exists():
        ae = torch.load(ae_path, map_location="cpu")
        scaler = _load_scaler(models_dir / "unsupervised_scaler.pkl")
        Xu = scaler.transform(X)
        model = AutoEncoder(
            input_dim=len(feature_cols),
            hidden=list(ae["model"]["hidden"]),
            latent_dim=int(ae["model"]["latent_dim"]),
        )
        model.load_state_dict(ae["state_dict"])
        model.eval()
        scores = anomaly_score(model, torch.tensor(Xu, dtype=torch.float32)).numpy()
        # normalize to 0..1
        s = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        out["risk_unsupervised"] = s
    else:
        out["risk_unsupervised"] = 0.0

    out["risk_final"] = 0.5 * out["risk_supervised"] + 0.5 * out["risk_unsupervised"]
    out = out.sort_values(["risk_final"], ascending=False).reset_index(drop=True)

    out_path = Path(reports) / "inference.csv"
    out.to_csv(out_path, index=False)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
