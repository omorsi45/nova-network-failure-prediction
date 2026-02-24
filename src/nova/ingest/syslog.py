from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


# Example line:
# 2025-08-20T12:01:10Z node=R1 severity=CRIT msg="BGP session down to 10.0.0.2"
LINE_RE = re.compile(
    r"^(?P<ts>\S+)\s+node=(?P<node>\S+)\s+severity=(?P<sev>\S+)\s+msg=\"(?P<msg>.*)\"\s*$"
)


def load_syslog(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    rows = []
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        m = LINE_RE.match(line)
        if not m:
            continue
        rows.append(
            {
                "timestamp": pd.to_datetime(m.group("ts"), utc=True, errors="coerce"),
                "node_id": m.group("node"),
                "severity": m.group("sev").upper(),
                "message": m.group("msg"),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.dropna(subset=["timestamp"]).sort_values(["node_id", "timestamp"]).reset_index(drop=True)
    df["event_type"] = df["message"].str.upper().str.extract(
        r"(BGP\s+DOWN|BGP\s+CLEAR|LINK\s+DOWN|PORT\s+FLAP|OSPF\s+DOWN|CONGESTION|CPU\s+SPIKE)",
        expand=False,
    ).fillna("OTHER")
    return df
