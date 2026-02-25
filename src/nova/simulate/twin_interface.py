from __future__ import annotations

from dataclasses import dataclass
from typing import Any

"""
Optional placeholder for GNS3 digital twin integration.

The repo runs without GNS3. If you want to hook into a real GNS3 lab:
- use the GNS3 REST API to start/stop nodes
- toggle links to simulate failures
- collect telemetry/syslog using your collector
- store outputs under data/raw/

This file is intentionally minimal because GNS3 setups are very lab-specific.
"""


@dataclass
class GNS3Config:
    url: str
    project_id: str
    api_key: str | None = None


class GNS3Twin:
    def __init__(self, cfg: GNS3Config):
        self.cfg = cfg

    def run_scenario(self, scenario: dict[str, Any]) -> None:
        raise NotImplementedError("Implement your lab-specific GNS3 calls here.")
