from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from nova.common.io import ensure_dir, read_df
from nova.evaluation.metrics import classification_metrics
from nova.evaluation.plots import save_pr, save_roc, save_score_hist
from nova.models.autoencoder import AutoEncoder
from nova.models.supervised import MLPClassifier
from nova.training.train_unsupervised import anomaly_score


def _time_split(df):
    d = df.sort_values("window_start").reset_index(drop=True)
    times = d["window_start"].sort_values().unique()
    n = len(times)
    n_test = max(1, int(n * 0.3))
    test_times = set(times[-n_test:])
    train_times = set(times[:-n_test])
    return d[d["window_start"].isin(train_times)], d[d["window_start"].isin(test_times)]


def _load_scaler(path: Path) -> StandardScaler:
    with path.open("rb") as f:
        return pickle.load(f)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=str, default="data/processed")
    ap.add_argument("--artifacts", type=str, default="artifacts")
    args = ap.parse_args()

    processed = Path(args.processed)
    art = Path(args.artifacts)
    figs = ensure_dir(art / "figures")
    reports = ensure_dir(art / "reports")
    models_dir = art / "models"

    df = read_df(processed / "dataset.parquet")
    feature_cols = json.loads((processed / "feature_cols.json").read_text(encoding="utf-8"))

    train_df, test_df = _time_split(df)
    X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
    X_test = test_df[feature_cols].to_numpy(dtype=np.float32)
    y_test = test_df["crisis"].to_numpy(dtype=int)

    # ---- supervised ----
    sup_path = models_dir / "supervised.pt"
    if sup_path.exists():
        sup = torch.load(sup_path, map_location="cpu")
        scaler = _load_scaler(models_dir / "supervised_scaler.pkl")
        Xte = scaler.transform(X_test)
        model = MLPClassifier(input_dim=len(feature_cols), hidden=list(sup["model"]["hidden"]), dropout=float(sup["model"]["dropout"]))
        model.load_state_dict(sup["state_dict"])
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(Xte, dtype=torch.float32))
            prob = torch.sigmoid(logits).numpy()

        m = classification_metrics(y_test, prob, threshold=0.5)
        (reports / "eval_supervised.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
        save_roc(y_test, prob, figs / "roc_supervised.png")
        save_pr(y_test, prob, figs / "pr_supervised.png")
        save_score_hist(y_test, prob, figs / "hist_supervised.png")
        print("supervised metrics:", m)

    # ---- unsupervised ----
    ae_path = models_dir / "unsupervised_ae.pt"
    if ae_path.exists():
        ae = torch.load(ae_path, map_location="cpu")
        scaler = _load_scaler(models_dir / "unsupervised_scaler.pkl")
        Xtr = scaler.transform(X_train)
        Xte = scaler.transform(X_test)

        model = AutoEncoder(
            input_dim=len(feature_cols),
            hidden=list(ae["model"]["hidden"]),
            latent_dim=int(ae["model"]["latent_dim"]),
        )
        model.load_state_dict(ae["state_dict"])
        model.eval()

        tr_scores = anomaly_score(model, torch.tensor(Xtr, dtype=torch.float32)).numpy()
        te_scores = anomaly_score(model, torch.tensor(Xte, dtype=torch.float32)).numpy()

        thr = float(np.quantile(tr_scores, 0.95))
        # convert to [0,1] score for plotting convenience
        te_norm = (te_scores - te_scores.min()) / (te_scores.max() - te_scores.min() + 1e-9)
        m = classification_metrics(y_test, (te_scores >= thr).astype(float), threshold=0.5)
        m["threshold_recon_error_p95_train"] = thr
        (reports / "eval_unsupervised.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
        save_score_hist(y_test, te_norm, figs / "hist_unsupervised.png")
        print("unsupervised metrics:", m)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
