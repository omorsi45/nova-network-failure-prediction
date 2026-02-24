from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from nova.common.io import ensure_dir, write_df
from nova.common.logging import setup_logging
from nova.ingest.flows import load_ctu13_binetflow
from nova.ingest.geospatial import load_geospatial
from nova.ingest.syslog import load_syslog
from nova.ingest.telemetry import load_cisco_csv, load_normalized_telemetry


def _find_one(raw: Path, patterns: list[str]) -> Path | None:
    for pat in patterns:
        hits = sorted(raw.glob(pat))
        if hits:
            return hits[0]
    return None


def main() -> int:
    setup_logging()

    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=str, default="data/raw")
    ap.add_argument("--out", type=str, default="data/interim")
    ap.add_argument("--cisco_sample_rows", type=int, default=200000)
    ap.add_argument("--ctu_sample_rows", type=int, default=200000)
    args = ap.parse_args()

    raw = Path(args.raw)
    out = ensure_dir(args.out)

    manifest = {"raw_dir": str(raw), "sources": {}}

    # telemetry
    tele = _find_one(raw, ["telemetry.csv"])
    if tele:
        df_t = load_normalized_telemetry(tele)
        manifest["sources"]["telemetry"] = str(tele)
    else:
        # try Cisco telemetry dataset (*.csv.gz)
        cisco = _find_one(raw, ["cisco_*_*.csv.gz", "*.csv.gz"])
        header = _find_one(raw, ["cisco_*_baseline_header.txt", "cisco_*_baseline_header.txt", "*header*.txt"])
        if cisco:
            df_t = load_cisco_csv(cisco, header_path=header, sample_rows=args.cisco_sample_rows)
            manifest["sources"]["telemetry"] = str(cisco)
            if header:
                manifest["sources"]["telemetry_header"] = str(header)
        else:
            raise RuntimeError(
                "No telemetry source found. Put telemetry.csv under data/raw/ or download a Cisco telemetry CSV.gz "
                "into data/raw/ (see scripts/00_download_datasets.py / `make download`)."
            )

    write_df(df_t, Path(out) / "telemetry.parquet")

    # flows (labeled bidirectional flow capture)
    flow = _find_one(raw, ["ctu13_*.binetflow", "ctu13_sample.binetflow.csv", "*.binetflow", "*.binetflow.csv"])
    if flow:
        df_f = load_ctu13_binetflow(flow, max_rows=args.ctu_sample_rows)
        manifest["sources"]["flows"] = str(flow)
        write_df(df_f, Path(out) / "flows.parquet")
    else:
        df_f = pd.DataFrame()
        write_df(df_f, Path(out) / "flows.parquet")

    # syslog
    syslog_p = _find_one(raw, ["syslog.txt"])
    if syslog_p:
        df_s = load_syslog(syslog_p)
        manifest["sources"]["syslog"] = str(syslog_p)
    else:
        df_s = pd.DataFrame()
    write_df(df_s, Path(out) / "syslog.parquet")

    # outages
    outages_p = _find_one(raw, ["outages.csv"])
    if outages_p:
        df_o = pd.read_csv(outages_p)
        manifest["sources"]["outages"] = str(outages_p)
    else:
        df_o = pd.DataFrame(columns=["node_id", "start_ts", "end_ts", "cause"])
    write_df(df_o, Path(out) / "outages.parquet")

    # geospatial
    geo_p = _find_one(raw, ["geospatial.csv"])
    if geo_p:
        df_g = load_geospatial(geo_p)
        manifest["sources"]["geospatial"] = str(geo_p)
    else:
        df_g = pd.DataFrame(columns=["node_id", "region", "lat", "lon"])
    write_df(df_g, Path(out) / "geospatial.parquet")

    (Path(out) / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote interim data to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
