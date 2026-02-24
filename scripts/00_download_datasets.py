from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from urllib.request import urlopen, urlretrieve


CISCO_RAW_BASE = "https://raw.githubusercontent.com/cisco-ie/telemetry/master"
CTU_SCENARIO_BASE = "https://mcfp.felk.cvut.cz/publicDatasets"


def _download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"download: {url} -> {out_path}")
    urlretrieve(url, out_path)


def download_cisco(folder: int, out_dir: Path, download_csv: bool = True) -> None:
    # We parse the folder README to discover the file names.
    readme_url = f"{CISCO_RAW_BASE}/{folder}/README.md"
    readme_txt = urlopen(readme_url).read().decode("utf-8", errors="ignore")
    out_readme = out_dir / f"cisco_{folder}_README.md"
    out_readme.write_text(readme_txt, encoding="utf-8")

    # grab files listed (csv.gz + header/case)
    files = set(re.findall(r"\b[\w\-\.]+\.(?:csv\.gz|txt|pdf|docx|md)\b", readme_txt))
    # README sometimes lists "baseline_header.txt" etc on one line
    if not files:
        print("warning: could not parse file list from Cisco README; downloading README only.")
        return

    for f in sorted(files):
        if (not download_csv) and f.endswith(".csv.gz"):
            continue
        url = f"{CISCO_RAW_BASE}/{folder}/{f}"
        try:
            _download(url, out_dir / f"cisco_{folder}_{f}")
        except Exception as e:
            # Some READMEs contain typos or stale filenames; skip and continue.
            print(f"warning: could not download {url}: {e}", file=sys.stderr)


def _ctu_scenario_name(num: int) -> str:
    return f"CTU-Malware-Capture-Botnet-{num}"


def _pick_first_href(html: str, suffix: str) -> str | None:
    m = re.search(r'href="([^"]*%s)"' % re.escape(suffix), html, re.IGNORECASE)
    return m.group(1) if m else None


def download_ctu13_binetflow(scenario: int, out_dir: Path, max_lines: int | None = None) -> None:
    # Find a .binetflow file from the detailed-bidirectional-flow-labels directory.
    scen = _ctu_scenario_name(scenario)
    base = f"{CTU_SCENARIO_BASE}/{scen}"
    index_html = urlopen(base + "/").read().decode("utf-8", errors="ignore")

    # folder link
    if "detailed-bidirectional-flow-labels" not in index_html:
        raise RuntimeError("could not find detailed-bidirectional-flow-labels/ in scenario index")

    det_html = urlopen(base + "/detailed-bidirectional-flow-labels/").read().decode("utf-8", errors="ignore")
    # find the first .binetflow file
    m = re.search(r'href="([^"]+\.binetflow)"', det_html, re.IGNORECASE)
    if not m:
        raise RuntimeError("could not find a .binetflow file in detailed-bidirectional-flow-labels/")
    fname = m.group(1)
    url = base + "/detailed-bidirectional-flow-labels/" + fname

    out_path = out_dir / f"ctu13_{scenario}_{fname}"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"download: {url} -> {out_path}")
    if max_lines is None:
        urlretrieve(url, out_path)
        return

    # stream only the first N lines (quick demo)
    with urlopen(url) as r, out_path.open("wb") as w:
        n = 0
        for line in r:
            w.write(line)
            n += 1
            if n >= max_lines:
                break
    print(f"wrote {n} lines to {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="data/raw")
    ap.add_argument("--cisco_folder", type=int, default=0)
    ap.add_argument("--skip_cisco_csv", action="store_true")
    ap.add_argument("--ctu13_scenario", type=int, default=44)
    ap.add_argument("--ctu13_max_lines", type=int, default=int(os.getenv("CTU13_MAX_LINES", "200000")))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        download_cisco(args.cisco_folder, out_dir, download_csv=not args.skip_cisco_csv)
    except Exception as e:
        print(f"warning: Cisco download failed: {e}", file=sys.stderr)

    try:
        download_ctu13_binetflow(args.ctu13_scenario, out_dir, max_lines=args.ctu13_max_lines)
    except Exception as e:
        print(f"warning: CTU flow download failed: {e}", file=sys.stderr)

    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
