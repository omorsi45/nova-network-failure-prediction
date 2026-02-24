from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class SplitConfig:
    test_days: float = 0.3
    val_days: float = 0.2


def _time_split(df: pd.DataFrame, split: SplitConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # assumes df has window_start
    d = df.sort_values("window_start").reset_index(drop=True)
    times = d["window_start"].sort_values().unique()
    n = len(times)
    if n < 10:
        # fallback random split
        idx = np.arange(len(d))
        np.random.shuffle(idx)
        n_test = int(len(d) * split.test_days)
        n_val = int(len(d) * split.val_days)
        test = d.iloc[idx[:n_test]]
        val = d.iloc[idx[n_test:n_test+n_val]]
        train = d.iloc[idx[n_test+n_val:]]
        return train, val, test

    n_test = max(1, int(n * split.test_days))
    n_val = max(1, int(n * split.val_days))
    test_times = set(times[-n_test:])
    val_times = set(times[-(n_test + n_val):-n_test])
    train_times = set(times[: -(n_test + n_val)])

    train = d[d["window_start"].isin(train_times)]
    val = d[d["window_start"].isin(val_times)]
    test = d[d["window_start"].isin(test_times)]
    return train, val, test


def make_loaders(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str | None,
    split_cfg: SplitConfig,
    batch_size: int,
    fit_scaler_on: str = "train",
) -> tuple[DataLoader, DataLoader, DataLoader, StandardScaler]:
    train_df, val_df, test_df = _time_split(df, split_cfg)

    scaler = StandardScaler()
    fit_df = train_df if fit_scaler_on == "train" else df
    X_train = scaler.fit_transform(train_df[feature_cols].to_numpy(dtype=np.float32))
    X_val = scaler.transform(val_df[feature_cols].to_numpy(dtype=np.float32)) if len(val_df) else np.empty((0, len(feature_cols)), dtype=np.float32)
    X_test = scaler.transform(test_df[feature_cols].to_numpy(dtype=np.float32)) if len(test_df) else np.empty((0, len(feature_cols)), dtype=np.float32)

    def to_loader(X: np.ndarray, y: np.ndarray | None) -> DataLoader:
        xt = torch.tensor(X, dtype=torch.float32)
        if y is None:
            ds = TensorDataset(xt)
        else:
            yt = torch.tensor(y.astype(np.float32), dtype=torch.float32)
            ds = TensorDataset(xt, yt)
        return DataLoader(ds, batch_size=batch_size, shuffle=(y is not None))

    y_train = train_df[label_col].to_numpy() if label_col else None
    y_val = val_df[label_col].to_numpy() if label_col else None
    y_test = test_df[label_col].to_numpy() if label_col else None

    return (
        to_loader(X_train, y_train),
        to_loader(X_val, y_val),
        to_loader(X_test, y_test),
        scaler,
    )
