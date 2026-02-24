from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import yaml

from nova.common.io import ensure_dir, read_df
from nova.common.seed import set_seed
from nova.training.datamodule import SplitConfig, make_loaders
from nova.training.train_supervised import TrainConfig, train_classifier


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=str, default="data/processed")
    ap.add_argument("--out", type=str, default="artifacts")
    ap.add_argument("--config", type=str, default="configs/train_supervised.yaml")
    args = ap.parse_args()

    processed = Path(args.processed)
    out = Path(args.out)
    models_dir = ensure_dir(out / "models")
    reports_dir = ensure_dir(out / "reports")

    cfg = _load_yaml(Path(args.config))
    set_seed(int(cfg["seed"]))

    df = read_df(processed / "dataset.parquet")
    feature_cols = json.loads((processed / "feature_cols.json").read_text(encoding="utf-8"))

    split = SplitConfig(test_days=float(cfg["split"]["test_days"]), val_days=float(cfg["split"]["val_days"]))
    train_loader, val_loader, test_loader, scaler = make_loaders(
        df=df,
        feature_cols=feature_cols,
        label_col="crisis",
        split_cfg=split,
        batch_size=int(cfg["train"]["batch_size"]),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, train_metrics = train_classifier(
        train_loader=train_loader,
        val_loader=val_loader,
        input_dim=len(feature_cols),
        hidden=list(cfg["model"]["hidden"]),
        dropout=float(cfg["model"]["dropout"]),
        cfg=TrainConfig(lr=float(cfg["train"]["lr"]), epochs=int(cfg["train"]["epochs"]), patience=int(cfg["train"]["patience"])),
        device=device,
    )

    # save
    ckpt = {
        "state_dict": model.state_dict(),
        "feature_cols": feature_cols,
        "model": cfg["model"],
    }
    torch.save(ckpt, models_dir / "supervised.pt")
    with (models_dir / "supervised_scaler.pkl").open("wb") as f:
        pickle.dump(scaler, f)

    (reports_dir / "supervised_train_metrics.json").write_text(
        json.dumps(train_metrics, indent=2), encoding="utf-8"
    )

    print(f"saved supervised model to {models_dir/'supervised.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
