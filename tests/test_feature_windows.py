from __future__ import annotations

from pathlib import Path

from nova.features.windows import WindowSpec, make_windows
from nova.ingest.telemetry import load_normalized_telemetry
from nova.simulate.local_simulator import SimConfig, generate


def test_make_windows(tmp_path: Path):
    paths = generate(tmp_path, SimConfig(seed=2, nodes=2, days=1, failure_rate_per_day=0.0))
    telemetry = load_normalized_telemetry(paths["telemetry"])
    w = make_windows(telemetry, WindowSpec(size_minutes=5, stride_minutes=1))
    assert len(w) > 0
    assert {"node_id", "window_start", "window_end"}.issubset(set(w.columns))
