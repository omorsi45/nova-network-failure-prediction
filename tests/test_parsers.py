from __future__ import annotations

from pathlib import Path

from nova.ingest.telemetry import load_normalized_telemetry
from nova.ingest.syslog import load_syslog
from nova.ingest.flows import load_ctu13_binetflow
from nova.simulate.local_simulator import SimConfig, generate


def test_ingest_parsers(tmp_path: Path):
    paths = generate(tmp_path, SimConfig(seed=1, nodes=3, days=1, failure_rate_per_day=1.0))

    t = load_normalized_telemetry(paths["telemetry"])
    assert {"timestamp", "node_id", "cpu_pct", "latency_ms", "throughput_mbps"}.issubset(set(t.columns))
    assert len(t) > 0

    s = load_syslog(paths["syslog"])
    assert set(["timestamp", "node_id", "severity", "message"]).issubset(set(s.columns)) or s.empty

    f = load_ctu13_binetflow(paths["flows"], max_rows=1000)
    assert {"StartTime", "SrcAddr", "TotBytes", "Label", "is_botnet"}.issubset(set(f.columns))
