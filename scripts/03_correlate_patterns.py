from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nova.common.io import ensure_dir, read_df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=str, default="data/processed")
    ap.add_argument("--out", type=str, default="artifacts")
    args = ap.parse_args()

    processed = Path(args.processed)
    out = ensure_dir(args.out)
    figs = ensure_dir(Path(out) / "figures")
    reports = ensure_dir(Path(out) / "reports")

    df = read_df(processed / "dataset.parquet")
    # focus on common signals
    candidates = [c for c in df.columns if any(k in c for k in ["cpu_pct", "latency_ms", "throughput_mbps", "packet_loss_pct", "flow_"])]
    X = df[candidates].select_dtypes(include=[np.number])

    if X.shape[1] < 2:
        print("not enough numeric columns for correlation")
        return 0

    corr = X.corr(method="spearman")

    # heatmap
    plt.figure(figsize=(12, 10))
    plt.imshow(corr.to_numpy(), aspect="auto")
    plt.title("Spearman correlation heatmap")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=6)
    plt.yticks(range(len(corr.index)), corr.index, fontsize=6)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(figs / "correlation_heatmap.png", bbox_inches="tight")
    plt.close()

    # top correlations with crisis
    if "crisis" in df.columns:
        y = df["crisis"].astype(int)
        # correlation with label (point-biserial approx with spearman)
        label_corr = {}
        for c in X.columns:
            label_corr[c] = float(pd.Series(X[c]).corr(y, method="spearman"))
        top = sorted(label_corr.items(), key=lambda kv: abs(kv[1]), reverse=True)[:15]
    else:
        top = []

    md = ["# Outage pattern summary", ""]
    md.append("Signals were computed from telemetry windows and labeled flow aggregates.")
    md.append("")
    if top:
        md.append("## Top correlated features with `crisis` (Spearman)")
        md.append("")
        for k, v in top:
            md.append(f"- {k}: {v:.3f}")
        md.append("")
    md.append("## Notes")
    md.append("- Use this to explain patterns like: latency up + throughput down + packet loss up, or botnet flow bursts.")
    md.append("- Correlation does not imply causation; treat this as a quick diagnostic.")
    (reports / "patterns.md").write_text("\n".join(md), encoding="utf-8")

    print(f"wrote {figs/'correlation_heatmap.png'} and {reports/'patterns.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
