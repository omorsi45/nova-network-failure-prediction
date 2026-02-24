from __future__ import annotations

import pandas as pd


def label_from_outages(windows: pd.DataFrame, outages: pd.DataFrame, node_col: str = "node_id") -> pd.Series:
    if windows.empty or outages.empty:
        return pd.Series([0] * len(windows), index=windows.index, dtype=int)

    o = outages.copy()
    o["start_ts"] = pd.to_datetime(o["start_ts"], utc=True, errors="coerce")
    o["end_ts"] = pd.to_datetime(o["end_ts"], utc=True, errors="coerce")
    o = o.dropna(subset=["start_ts", "end_ts"])

    labels = []
    for _, w in windows.iterrows():
        node = w[node_col]
        ws, we = w["window_start"], w["window_end"]
        seg = o[o[node_col] == node]
        hit = ((seg["start_ts"] < we) & (seg["end_ts"] > ws)).any()
        labels.append(int(hit))
    return pd.Series(labels, index=windows.index, dtype=int)


def label_from_syslog(windows: pd.DataFrame, syslog: pd.DataFrame, node_col: str = "node_id") -> pd.Series:
    if windows.empty or syslog.empty:
        return pd.Series([0] * len(windows), index=windows.index, dtype=int)

    s = syslog.copy()
    s["timestamp"] = pd.to_datetime(s["timestamp"], utc=True, errors="coerce")
    s = s.dropna(subset=["timestamp"])
    bad_events = {"BGP DOWN", "BGP CLEAR", "LINK DOWN", "PORT FLAP", "OSPF DOWN", "CONGESTION", "CPU SPIKE"}
    bad_sev = {"CRIT", "ERR", "ERROR"}

    labels = []
    for _, w in windows.iterrows():
        node = w[node_col]
        ws, we = w["window_start"], w["window_end"]
        seg = s[(s[node_col] == node) & (s["timestamp"] >= ws) & (s["timestamp"] < we)]
        hit = False
        if not seg.empty:
            hit = ((seg["event_type"].isin(bad_events)) & (seg["severity"].isin(bad_sev))).any()
        labels.append(int(hit))
    return pd.Series(labels, index=windows.index, dtype=int)


def label_synthetic_from_metrics(features: pd.DataFrame) -> pd.Series:
    # fallback label rule if no outages/syslog exist:
    # crisis if latency in top 5% AND throughput in bottom 10% OR packet_loss high
    if features.empty:
        return pd.Series(dtype=int)
    lat = features.get("latency_ms_p95", pd.Series([0]*len(features)))
    thr = features.get("throughput_mbps_mean", pd.Series([0]*len(features)))
    loss = features.get("packet_loss_pct_mean", pd.Series([0]*len(features)))

    lat_thr = lat.quantile(0.95) if len(lat) else 0
    thr_thr = thr.quantile(0.10) if len(thr) else 0

    crisis = ((lat >= lat_thr) & (thr <= thr_thr)) | (loss >= loss.quantile(0.95) if len(loss) else 0)
    return crisis.astype(int)
