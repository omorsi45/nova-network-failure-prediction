# NOVA — Network Outage Prediction

End-to-end pipeline for network failure/outage prediction using time-windowed features from telemetry and syslog, with optional labeled flow signals from public datasets.

## Features

- **Pipeline scripts**: download → ingest → build dataset → correlate → train → evaluate → infer
- **Models**: supervised classifier + unsupervised autoencoder (PyTorch)
- **Labeling**: outage intervals (if available), syslog rules, and synthetic heuristics

This project is inspired by work done during a network automation and analytics internship, reimplemented using public datasets.

## Quick start (real data)

```bash
git clone https://github.com/omorsi45/nova-network-failure-prediction
cd nova-network-failure-prediction

python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 1) Download supported public datasets into data/raw/
make download

# 2) Ingest into Parquet (data/interim/)
make ingest

# 3) Build a fused ML table (windowed features + labels) into data/processed/
make build_dataset

# 4) Correlate patterns (CPU/latency/throughput/flows)
make correlate

# 5) Train models
make train_sup
make train_unsup

# 6) Evaluate and save plots + metrics
make eval

# 7) Produce risk scores per node/window
make infer
```

Outputs:
- `data/processed/dataset.parquet`
- `artifacts/models/*.pt`
- `artifacts/reports/*.json` and `artifacts/figures/*.png`

## Datasets

### Supported downloads

The default download script pulls from:

- **Cisco telemetry dataset repository**: `https://github.com/cisco-ie/telemetry`
  - This project downloads files from the repository’s numbered folders (default: folder `0`) using raw GitHub URLs.
- **CTU Malware Capture (labeled bidirectional flows)**: `https://mcfp.felk.cvut.cz/publicDatasets/`
  - This project downloads a single `*.binetflow` file from `CTU-Malware-Capture-Botnet-<scenario>/detailed-bidirectional-flow-labels/` (default: scenario `44`).

To download into `data/raw/`:

```bash
make download
```

If you want the full `*.binetflow` file, set a larger `CTU13_MAX_LINES`:

```bash
CTU13_MAX_LINES=999999999 make download
```

### File placement (manual)

If you don’t want to use `make download`, place files under `data/raw/`:

- `telemetry.csv` (recommended if you have your own telemetry export) with columns:
  `timestamp,node_id,cpu_pct,latency_ms,throughput_mbps,packet_loss_pct,mem_pct,if_errors`
- Optional:
  - `syslog.txt`
  - `outages.csv`
  - `geospatial.csv`
  - `*.binetflow` (CTU bidirectional labeled flow file)

### Ingest and build dataset

```bash
make ingest
make build_dataset
```

Notes:
- The flow dataset uses IPs as “hosts”, while telemetry uses device IDs.
  This repo creates a simple deterministic mapping from CTU IPs → telemetry nodes
  to build a single fused training table (`data/processed/ip_to_node_map.json`).

## Project structure

```
src/nova/
  ingest/        # load telemetry, flows, syslog, geospatial
  features/      # time windows + feature engineering + labeling
  models/        # PyTorch MLP + AutoEncoder
  training/      # train loops + time-based splitting + scaling
  evaluation/    # metrics + plots
  simulate/      # local failure simulator + (optional) GNS3 placeholder

scripts/
  00_download_datasets.py
  01_ingest.py
  02_build_dataset.py
  03_correlate_patterns.py
  04_train_supervised.py
  05_train_unsupervised.py
  06_evaluate.py
  07_infer.py
  08_simulate_failures.py
```

## How labels are created

Priority order:
1. `outages.csv` intervals (ground-truth if you have it)
2. syslog critical events (BGP clear/down, link down, port flap, congestion, cpu spike)
3. synthetic label rule from metrics (latency high + throughput low, or packet loss high)
4. synthetic trigger from labeled flow bursts (botnet fraction spike)

## Development

```bash
ruff check .
pytest -q
```

## GNS3 digital twin (optional)

This repo includes an integration placeholder in `src/nova/simulate/twin_interface.py`.
