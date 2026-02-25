from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class SimConfig:
    seed: int = 7
    nodes: int = 8
    days: int = 2
    failure_rate_per_day: float = 1.5  # expected failures per node per day


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def generate(
    out_dir: str | Path,
    cfg: SimConfig,
) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rng = _rng(cfg.seed)
    random.seed(cfg.seed)

    start = pd.Timestamp("2025-08-20T00:00:00Z")
    minutes = cfg.days * 24 * 60
    ts = pd.date_range(start, periods=minutes, freq="1min", tz="UTC")

    node_ids = [f"R{i+1}" for i in range(cfg.nodes)]

    # baseline telemetry
    rows = []
    for node in node_ids:
        base_cpu = rng.uniform(10, 30)
        base_lat = rng.uniform(5, 20)
        base_thr = rng.uniform(200, 800)  # Mbps
        base_mem = rng.uniform(30, 60)

        for t in ts:
            cpu = base_cpu + rng.normal(0, 3)
            lat = base_lat + rng.normal(0, 2)
            thr = base_thr + rng.normal(0, 30)
            loss = max(0, rng.normal(0.1, 0.05))
            mem = base_mem + rng.normal(0, 2)
            ife = max(0, rng.normal(0.0, 0.2))
            rows.append(
                {
                    "timestamp": t.isoformat(),
                    "node_id": node,
                    "cpu_pct": float(np.clip(cpu, 0, 100)),
                    "latency_ms": float(max(0, lat)),
                    "throughput_mbps": float(max(0, thr)),
                    "packet_loss_pct": float(min(100, loss)),
                    "mem_pct": float(np.clip(mem, 0, 100)),
                    "if_errors": float(ife),
                }
            )
    telemetry = pd.DataFrame(rows)

    # failures
    outages = []
    syslog_lines = []
    for node in node_ids:
        n_fail = int(rng.poisson(cfg.failure_rate_per_day * cfg.days))
        for _ in range(n_fail):
            t0 = start + pd.Timedelta(minutes=int(rng.integers(0, minutes - 60)))
            dur = int(rng.integers(10, 60))
            t1 = t0 + pd.Timedelta(minutes=dur)

            ftype = random.choice(["congestion", "cpu_spike", "link_flap"])
            outages.append({"node_id": node, "start_ts": t0.isoformat(), "end_ts": t1.isoformat(), "cause": ftype})

            mask = (telemetry["node_id"] == node) & (pd.to_datetime(telemetry["timestamp"], utc=True) >= t0) & (
                pd.to_datetime(telemetry["timestamp"], utc=True) < t1
            )

            if ftype == "congestion":
                telemetry.loc[mask, "latency_ms"] *= rng.uniform(3, 8)
                telemetry.loc[mask, "throughput_mbps"] *= rng.uniform(0.1, 0.5)
                telemetry.loc[mask, "packet_loss_pct"] += rng.uniform(2, 10)
                msg = 'msg="CONGESTION detected: queue build-up"'
                sev = "CRIT"
                syslog_lines.append(f"{t0.isoformat()} node={node} severity={sev} {msg}")
            elif ftype == "cpu_spike":
                telemetry.loc[mask, "cpu_pct"] += rng.uniform(40, 70)
                telemetry.loc[mask, "latency_ms"] += rng.uniform(10, 50)
                msg = 'msg="CPU SPIKE on control-plane"'
                sev = "ERR"
                syslog_lines.append(f"{t0.isoformat()} node={node} severity={sev} {msg}")
            else:  # link_flap
                telemetry.loc[mask, "packet_loss_pct"] += rng.uniform(5, 30)
                telemetry.loc[mask, "if_errors"] += rng.uniform(5, 30)
                telemetry.loc[mask, "throughput_mbps"] *= rng.uniform(0.0, 0.3)
                sev = "CRIT"
                syslog_lines.append(f'{t0.isoformat()} node={node} severity={sev} msg="LINK DOWN on interface Gi0/0"')
                syslog_lines.append(f'{(t0 + pd.Timedelta(minutes=2)).isoformat()} node={node} severity={sev} msg="PORT FLAP Gi0/0"')

    telemetry["cpu_pct"] = telemetry["cpu_pct"].clip(0, 100)
    telemetry["mem_pct"] = telemetry["mem_pct"].clip(0, 100)
    telemetry["packet_loss_pct"] = telemetry["packet_loss_pct"].clip(0, 100)
    telemetry["throughput_mbps"] = telemetry["throughput_mbps"].clip(0, None)
    telemetry["latency_ms"] = telemetry["latency_ms"].clip(0, None)

    # geospatial (fake)
    geo = pd.DataFrame(
        {
            "node_id": node_ids,
            "site_id": [f"S{i%3+1}" for i in range(cfg.nodes)],
            "region": [f"Region{i%2+1}" for i in range(cfg.nodes)],
            "lat": rng.uniform(45.4, 45.6, cfg.nodes),
            "lon": rng.uniform(-73.8, -73.5, cfg.nodes),
        }
    )

    # toy flows with CTU-like schema
    flows = []
    ip_map = {node: f"10.0.0.{i+1}" for i, node in enumerate(node_ids)}
    for node in node_ids:
        ip = ip_map[node]
        for t in ts[::2]:  # every 2 min
            bot = rng.random() < 0.02
            dur = float(max(0.01, rng.normal(1.2 if not bot else 3.0, 0.5)))
            pkts = int(max(1, rng.poisson(12 if not bot else 40)))
            bytes_ = int(pkts * rng.integers(60, 1200))
            flows.append(
                {
                    "StartTime": t.isoformat(),
                    "Dur": dur,
                    "Proto": random.choice(["tcp", "udp"]),
                    "SrcAddr": ip,
                    "Sport": int(rng.integers(1024, 65535)),
                    "Dir": "->",
                    "DstAddr": f"172.16.0.{int(rng.integers(1, 254))}",
                    "Dport": int(random.choice([80, 443, 53, 22, 8080])),
                    "State": "CON",
                    "sTos": 0,
                    "dTos": 0,
                    "TotPkts": pkts,
                    "TotBytes": bytes_,
                    "SrcBytes": int(bytes_ * rng.uniform(0.3, 0.7)),
                    "Label": "From-Botnet" if bot else "Normal",
                }
            )
    flows = pd.DataFrame(flows)

    # write files
    tele_path = out / "telemetry.csv"
    sys_path = out / "syslog.txt"
    out_path = out / "outages.csv"
    geo_path = out / "geospatial.csv"
    flow_path = out / "ctu13_sample.binetflow.csv"

    telemetry.to_csv(tele_path, index=False)
    Path(sys_path).write_text("\n".join(syslog_lines) + "\n", encoding="utf-8")
    pd.DataFrame(outages).to_csv(out_path, index=False)
    geo.to_csv(geo_path, index=False)
    flows.to_csv(flow_path, index=False, header=False)  # mimic binetflow without header row

    return {
        "telemetry": tele_path,
        "syslog": sys_path,
        "outages": out_path,
        "geospatial": geo_path,
        "flows": flow_path,
    }
