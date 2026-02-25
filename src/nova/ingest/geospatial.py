from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_geospatial(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    keep = [c for c in ["node_id", "site_id", "region", "lat", "lon"] if c in df.columns]
    df = df[keep].copy()
    return df
