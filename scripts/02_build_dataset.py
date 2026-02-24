from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from nova.common.io import ensure_dir, read_df, write_df
from nova.features.label import label_from_outages, label_from_syslog, label_synthetic_from_metrics
from nova.features.stats import flows_window_features, telemetry_window_features
from nova.features.windows import WindowSpec, make_windows


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _make_ip_to_node_map(flow_ips: list[str], nodes: list[str]) -> dict[str, str]:
    if not nodes:
        # if no telemetry node ids, keep IPs as nodes
        return {ip: ip for ip in flow_ips}
    out = {}
    for i, ip in enumerate(flow_ips):
        out[ip] = nodes[i % len(nodes)]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interim", type=str, default="data/interim")
    ap.add_argument("--out", type=str, default="data/processed")
    ap.add_argument("--config", type=str, default="configs/data.yaml")
    args = ap.parse_args()

    interim = Path(args.interim)
    out = ensure_dir(args.out)

    cfg = _load_yaml(Path(args.config))
    spec = WindowSpec(
        size_minutes=int(cfg["window"]["size_minutes"]),
        stride_minutes=int(cfg["window"]["stride_minutes"]),
    )

    telemetry = read_df(interim / "telemetry.parquet")
    flows = read_df(interim / "flows.parquet")
    syslog = read_df(interim / "syslog.parquet")
    outages = read_df(interim / "outages.parquet")
    geo = read_df(interim / "geospatial.parquet")

    telemetry["timestamp"] = pd.to_datetime(telemetry["timestamp"], utc=True, errors="coerce")
    telemetry = telemetry.dropna(subset=["timestamp"]).reset_index(drop=True)

    # fuse flow hosts into telemetry nodes (simple mapping for a unified dataset)
    ip_map = {}
    if not flows.empty:
        flow_ips = sorted(flows["SrcAddr"].astype(str).unique())
        nodes = sorted(telemetry["node_id"].astype(str).unique())
        ip_map = _make_ip_to_node_map(flow_ips, nodes)
        flows = flows.copy()
        flows["node_id"] = flows["SrcAddr"].astype(str).map(ip_map)

    # windows from telemetry time range per node
    windows = make_windows(telemetry, spec)

    tf = telemetry_window_features(telemetry, windows)
    if not flows.empty:
        ff = flows_window_features(flows, windows)
    else:
        # Ensure merge keys exist even when flow features are unavailable.
        ff = pd.DataFrame(columns=["node_id", "window_start"])

    # merge features
    feats = tf.merge(ff, on=["node_id", "window_start"], how="left")
    feats = feats.fillna(0.0)

    # geospatial (one-hot region)
    if not geo.empty and "region" in geo.columns:
        feats = feats.merge(geo[["node_id", "region"]], on="node_id", how="left")
        feats["region"] = feats["region"].fillna("unknown")
        d = pd.get_dummies(feats["region"], prefix="region")
        feats = pd.concat([feats.drop(columns=["region"]), d], axis=1)

    # labels
    windows_for_labels = windows.merge(feats[["node_id", "window_start"]], on=["node_id", "window_start"], how="inner")
    windows_for_labels = windows_for_labels.sort_values(["node_id", "window_start"]).reset_index(drop=True)

    y_out = label_from_outages(windows_for_labels, outages)
    y_sys = label_from_syslog(windows_for_labels, syslog)
    y = (y_out | y_sys).astype(int)

    if y.sum() == 0:
        # add synthetic labels from metrics if needed
        y = label_synthetic_from_metrics(feats).astype(int)

    # incorporate botnet bursts as crisis triggers (synthetic outage labels)
    if "flow_botnet_frac" in feats.columns:
        burst = (feats["flow_botnet_frac"] >= 0.5) & (feats["flow_count"] >= feats["flow_count"].quantile(0.90))
        y = (y | burst.astype(int)).astype(int)

    feats["crisis"] = y.to_numpy(dtype=int)

    # select feature columns
    meta_cols = {"node_id", "window_start", "crisis"}
    feature_cols = [c for c in feats.columns if c not in meta_cols]
    # remove anything non-numeric
    numeric_cols = []
    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(feats[c]):
            numeric_cols.append(c)
    feature_cols = numeric_cols

    write_df(feats[["node_id", "window_start"] + feature_cols + ["crisis"]], Path(out) / "dataset.parquet")
    (Path(out) / "feature_cols.json").write_text(json.dumps(feature_cols, indent=2), encoding="utf-8")
    (Path(out) / "ip_to_node_map.json").write_text(json.dumps(ip_map, indent=2), encoding="utf-8")

    print(f"wrote dataset.parquet with {len(feats)} rows and {len(feature_cols)} features")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
