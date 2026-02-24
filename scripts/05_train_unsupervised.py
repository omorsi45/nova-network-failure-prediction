from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import torch
import yaml

from nova.common.io import ensure_dir, read_df
from nova.common.seed import set_seed
from nova.training.datamodule import SplitConfig, make_loaders
from nova.training.train_unsupervised import TrainConfig, train_autoencoder


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=str, default="data/processed")
    ap.add_argument("--out", type=str, default="artifacts")
    ap.add_argument("--config", type=str, default="configs/train_unsupervised.yaml")
    args = ap.parse_args()

    processed = Path(args.processed)
    out = Path(args.out)
    models_dir = ensure_dir(out / "models")
    reports_dir = ensure_dir(out / "reports")

    cfg = _load_yaml(Path(args.config))
    set_seed(int(cfg["seed"]))

    df = read_df(processed / "dataset.parquet")
    feature_cols = json.loads((processed / "feature_cols.json").read_text(encoding="utf-8"))

    # train only on normal windows
    df_n = df[df["crisis"] == 0].copy()
    if len(df_n) < 100:
        print("warning: few normal rows; training on full dataset as fallback")
        df_n = df.copy()

    split = SplitConfig(test_days=0.2, val_days=0.2)
    train_loader, val_loader, test_loader, scaler = make_loaders(
        df=df_n,
        feature_cols=feature_cols,
        label_col=None,
        split_cfg=split,
        batch_size=int(cfg["train"]["batch_size"]),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, train_metrics = train_autoencoder(
        train_loader=train_loader,
        val_loader=val_loader,
        input_dim=len(feature_cols),
        hidden=list(cfg["model"]["hidden"]),
        latent_dim=int(cfg["model"]["latent_dim"]),
        cfg=TrainConfig(lr=float(cfg["train"]["lr"]), epochs=int(cfg["train"]["epochs"]), patience=int(cfg["train"]["patience"])),
        device=device,
    )

    ckpt = {
        "state_dict": model.state_dict(),
        "feature_cols": feature_cols,
        "model": cfg["model"],
    }
    torch.save(ckpt, models_dir / "unsupervised_ae.pt")
    with (models_dir / "unsupervised_scaler.pkl").open("wb") as f:
        pickle.dump(scaler, f)

    (reports_dir / "unsupervised_train_metrics.json").write_text(
        json.dumps(train_metrics, indent=2), encoding="utf-8"
    )

    print(f"saved unsupervised AE model to {models_dir/'unsupervised_ae.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
