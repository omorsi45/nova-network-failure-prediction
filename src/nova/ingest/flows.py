from __future__ import annotations

from pathlib import Path
import pandas as pd


CTU13_COLUMNS = [
    "StartTime",
    "Dur",
    "Proto",
    "SrcAddr",
    "Sport",
    "Dir",
    "DstAddr",
    "Dport",
    "State",
    "sTos",
    "dTos",
    "TotPkts",
    "TotBytes",
    "SrcBytes",
    "Label",
]


def load_ctu13_binetflow(path: str | Path, max_rows: int | None = None) -> pd.DataFrame:
    # CTU binetflow is CSV-like, with some comment lines.
    df = pd.read_csv(
        path,
        comment="#",
        names=CTU13_COLUMNS,
        header=None,
        nrows=max_rows,
        low_memory=False,
    )
    # Parse time
    df["StartTime"] = pd.to_datetime(df["StartTime"], utc=True, errors="coerce")
    df = df.dropna(subset=["StartTime"]).reset_index(drop=True)

    # Normalize label into 0/1 botnet flag (conservative)
    lab = df["Label"].astype(str)
    df["is_botnet"] = lab.str.contains("Botnet|From-Botnet", case=False, regex=True).astype(int)

    # numeric columns
    for c in ["Dur", "TotPkts", "TotBytes", "SrcBytes"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    return df
