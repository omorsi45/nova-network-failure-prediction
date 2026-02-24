from __future__ import annotations

import numpy as np
import pandas as pd


def telemetry_window_features(
    telemetry: pd.DataFrame,
    windows: pd.DataFrame,
    node_col: str = "node_id",
) -> pd.DataFrame:
    if telemetry.empty or windows.empty:
        return pd.DataFrame()

    t = telemetry.copy()
    t = t.sort_values([node_col, "timestamp"])

    feats = []
    for node, w in windows.groupby(node_col):
        g = t[t[node_col] == node]
        if g.empty:
            continue
        for _, row in w.iterrows():
            ws, we = row["window_start"], row["window_end"]
            seg = g[(g["timestamp"] >= ws) & (g["timestamp"] < we)]
            if seg.empty:
                continue

            def agg(col: str):
                v = pd.to_numeric(seg[col], errors="coerce")
                return {
                    f"{col}_mean": float(v.mean()),
                    f"{col}_std": float(v.std(ddof=0)),
                    f"{col}_min": float(v.min()),
                    f"{col}_max": float(v.max()),
                    f"{col}_p95": float(v.quantile(0.95)),
                }

            rowf = {node_col: node, "window_start": ws}
            for col in ["cpu_pct", "latency_ms", "throughput_mbps", "packet_loss_pct", "mem_pct", "if_errors"]:
                if col in seg.columns:
                    rowf.update(agg(col))

            # simple slope for cpu/latency
            for col in ["cpu_pct", "latency_ms", "throughput_mbps"]:
                if col in seg.columns and len(seg) >= 2:
                    x = (seg["timestamp"].astype("int64") / 1e9).to_numpy()
                    y = pd.to_numeric(seg[col], errors="coerce").to_numpy()
                    m = np.polyfit(x, y, 1)[0]
                    rowf[f"{col}_slope"] = float(m)
                else:
                    rowf[f"{col}_slope"] = 0.0

            feats.append(rowf)

    df = pd.DataFrame(feats)
    return df


def flows_window_features(
    flows: pd.DataFrame,
    windows: pd.DataFrame,
    node_col: str = "node_id",
) -> pd.DataFrame:
    if flows.empty or windows.empty:
        return pd.DataFrame()

    f = flows.copy()
    f = f.rename(columns={"StartTime": "timestamp", "SrcAddr": node_col})
    f = f.sort_values([node_col, "timestamp"])

    feats = []
    for node, w in windows.groupby(node_col):
        g = f[f[node_col] == node]
        if g.empty:
            continue

        for _, row in w.iterrows():
            ws, we = row["window_start"], row["window_end"]
            seg = g[(g["timestamp"] >= ws) & (g["timestamp"] < we)]
            if seg.empty:
                continue

            tot_flows = len(seg)
            bot_flows = int(seg["is_botnet"].sum()) if "is_botnet" in seg.columns else 0
            bytes_sum = float(pd.to_numeric(seg["TotBytes"], errors="coerce").fillna(0).sum()) if "TotBytes" in seg.columns else 0.0
            pkts_sum = float(pd.to_numeric(seg["TotPkts"], errors="coerce").fillna(0).sum()) if "TotPkts" in seg.columns else 0.0
            avg_dur = float(pd.to_numeric(seg["Dur"], errors="coerce").fillna(0).mean()) if "Dur" in seg.columns else 0.0

            feats.append(
                {
                    node_col: node,
                    "window_start": ws,
                    "flow_count": tot_flows,
                    "flow_botnet_count": bot_flows,
                    "flow_botnet_frac": bot_flows / max(tot_flows, 1),
                    "flow_bytes_sum": bytes_sum,
                    "flow_pkts_sum": pkts_sum,
                    "flow_dur_mean": avg_dur,
                }
            )

    return pd.DataFrame(feats)
