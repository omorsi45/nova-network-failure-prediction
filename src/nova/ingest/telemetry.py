from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


NORMALIZED_COLS = [
    "timestamp",
    "node_id",
    "cpu_pct",
    "latency_ms",
    "throughput_mbps",
    "packet_loss_pct",
    "mem_pct",
    "if_errors",
]


def _to_datetime_utc(s: pd.Series) -> pd.Series:
    # Handles unix seconds, ms, us, ns, or ISO strings
    if np.issubdtype(s.dtype, np.number):
        v = s.astype("float64")
        vmax = float(np.nanmax(v)) if len(v) else float("nan")
        # Heuristic based on typical epoch magnitudes:
        # - seconds: ~1e9
        # - milliseconds: ~1e12
        # - microseconds: ~1e15
        # - nanoseconds: ~1e18
        if np.isfinite(vmax):
            if vmax >= 1e18:
                return pd.to_datetime(v, unit="ns", utc=True, errors="coerce")
            if vmax >= 1e15:
                return pd.to_datetime(v, unit="us", utc=True, errors="coerce")
            if vmax >= 1e12:
                return pd.to_datetime(v, unit="ms", utc=True, errors="coerce")
        return pd.to_datetime(v, unit="s", utc=True, errors="coerce")
    return pd.to_datetime(s, utc=True, errors="coerce")


def load_normalized_telemetry(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(NORMALIZED_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"telemetry.csv missing columns: {sorted(missing)}")
    df["timestamp"] = _to_datetime_utc(df["timestamp"])
    df = df.dropna(subset=["timestamp"])
    return df.sort_values(["node_id", "timestamp"]).reset_index(drop=True)


def _read_cisco_header(header_path: Path) -> list[str]:
    # baseline_header.txt usually lists column numbers and names.
    # We keep only names; if parsing fails, we fallback to generic names.
    names: list[str] = []
    for line in header_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        # common formats: "0: field_name" or "0 field_name"
        line = line.replace(":", " ")
        parts = line.split()
        if len(parts) >= 2 and parts[0].isdigit():
            names.append(parts[1])
    return names


def _pick_first(cols: Iterable[str], keywords: list[str]) -> str | None:
    lc = [c.lower() for c in cols]
    for kw in keywords:
        for c, l in zip(cols, lc):
            if kw in l:
                return c
    return None


def _scale_to_0_100(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    lo, hi = np.nanpercentile(x, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(np.clip(x, 0, 100), index=x.index)
    z = (x - lo) / (hi - lo)
    return pd.Series(np.clip(100 * z, 0, 100), index=x.index)


def load_cisco_csv(
    csv_gz_path: str | Path,
    header_path: str | Path | None = None,
    sample_rows: int | None = None,
) -> pd.DataFrame:
    p = Path(csv_gz_path)
    header_names = None
    if header_path:
        hp = Path(header_path)
        if hp.exists():
            names = _read_cisco_header(hp)
            if names:
                header_names = names

    df = pd.read_csv(
        p,
        compression="gzip" if p.suffixes[-2:] == [".csv", ".gz"] or p.suffix == ".gz" else None,
        header=None if header_names else "infer",
        names=header_names,
        nrows=sample_rows,
        low_memory=False,
    )

    # Find a time column
    time_col = _pick_first(df.columns, ["time", "timestamp", "ts"])
    if time_col is None:
        # last resort: assume first column is time
        time_col = df.columns[0]

    # Node identifier (if any)
    node_col = _pick_first(df.columns, ["hostname", "host", "node", "device", "router", "switch"])
    if node_col is None:
        df["node_id"] = "cisco_0"
    else:
        df["node_id"] = df[node_col].astype(str)

    df["timestamp"] = _to_datetime_utc(df[time_col])

    # Pick candidate columns
    cpu_col = _pick_first(df.columns, ["cpu", "proc", "util"])
    mem_col = _pick_first(df.columns, ["mem", "memory"])
    rtt_col = _pick_first(df.columns, ["rtt", "lat", "delay", "queue"])
    bytes_col = _pick_first(df.columns, ["bytes", "octets"])
    pkts_col = _pick_first(df.columns, ["pkts", "packets"])
    drop_col = _pick_first(df.columns, ["drop", "loss"])
    err_col = _pick_first(df.columns, ["error", "crc"])

    out = pd.DataFrame({"timestamp": df["timestamp"], "node_id": df["node_id"]})
    out["cpu_pct"] = _scale_to_0_100(df[cpu_col]) if cpu_col else 0.0
    out["mem_pct"] = _scale_to_0_100(df[mem_col]) if mem_col else 0.0

    # latency proxy: if rtt exists, scale to ms-ish; else synth from cpu
    if rtt_col:
        v = pd.to_numeric(df[rtt_col], errors="coerce")
        out["latency_ms"] = _scale_to_0_100(v) * 5.0  # 0..500ms-ish
    else:
        out["latency_ms"] = 5.0 + 3.0 * out["cpu_pct"] + np.random.normal(0, 5, len(out))

    # throughput proxy: bytes delta per second -> Mbps
    if bytes_col:
        b = pd.to_numeric(df[bytes_col], errors="coerce")
        # per-node diff
        out["throughput_mbps"] = (
            b.groupby(out["node_id"]).diff().clip(lower=0) * 8.0 / 1_000_000.0
        )
        out["throughput_mbps"] = out["throughput_mbps"].fillna(0.0)
    elif pkts_col:
        p = pd.to_numeric(df[pkts_col], errors="coerce")
        out["throughput_mbps"] = (
            p.groupby(out["node_id"]).diff().clip(lower=0) * 1500 * 8.0 / 1_000_000.0
        ).fillna(0.0)
    else:
        out["throughput_mbps"] = np.maximum(0.0, 200 - out["latency_ms"] + np.random.normal(0, 10, len(out)))

    out["packet_loss_pct"] = _scale_to_0_100(df[drop_col]) / 10.0 if drop_col else 0.0
    out["if_errors"] = pd.to_numeric(df[err_col], errors="coerce").fillna(0.0) if err_col else 0.0

    out = out.dropna(subset=["timestamp"]).sort_values(["node_id", "timestamp"]).reset_index(drop=True)

    # Ensure all normalized cols exist
    for c in NORMALIZED_COLS:
        if c not in out.columns:
            out[c] = 0.0

    return out[NORMALIZED_COLS]
