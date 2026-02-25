from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class WindowSpec:
    size_minutes: int = 5
    stride_minutes: int = 1


def make_windows(df: pd.DataFrame, spec: WindowSpec, time_col: str = "timestamp", node_col: str = "node_id") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[node_col, "window_start", "window_end"])

    out = []
    step = pd.Timedelta(minutes=spec.stride_minutes)
    size = pd.Timedelta(minutes=spec.size_minutes)

    for node, g in df.groupby(node_col):
        t0 = g[time_col].min().floor(f"{spec.stride_minutes}min")
        t1 = g[time_col].max().ceil(f"{spec.stride_minutes}min")
        ts = pd.date_range(t0, t1, freq=step, tz="UTC")
        starts = ts[:-1]
        for ws in starts:
            we = ws + size
            out.append({node_col: node, "window_start": ws, "window_end": we})

    return pd.DataFrame(out)
