"""Export the API as static JSON for the React site (web/).

The replay only covers 2022-2024, so every API answer can be computed ahead of time. This runs the
same Outlook class as app/server.py and writes web/public/api/: overview.json plus one
day/YYYY-MM-DD.json per day. No Python or model file is deployed.

    WILDFIRE_DIR=private/wildfire-outlook python deploy/vercel/export.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "web/public/api"
sys.path.insert(0, str(ROOT / "app"))
import server  # noqa: E402  (reads WILDFIRE_DIR at import)

# Timings recorded on the LinuxONE server during the datathon (3 CPUs, pinned s390x image).
# The static site can't measure anything live, so it shows these instead of the export machine's.
LINUXONE_RUNTIME = {
    "snapshot": True, "machine": "s390x", "cpus": 3, "python": "3.12.15", "system": "IBM LinuxONE (IBM Z)",
    "load_ms": 616, "score_all_ms": 212, "score_day_ms": 6.3, "importance_ms": 6300,
}


def write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, allow_nan=False, separators=(",", ":")), encoding="utf-8")


def main() -> None:
    outlook = server.Outlook()
    shutil.rmtree(OUT, ignore_errors=True)
    overview = dict(outlook.overview)
    overview["runtime"] = {**overview["runtime"], **LINUXONE_RUNTIME}
    write(OUT / "overview.json", overview)
    dates = overview["timeline"]["dates"]
    for day in dates:
        payload = outlook.day(day)
        del payload["live_ms"], payload["machine"]
        write(OUT / f"day/{day}.json", payload)
    size = sum(f.stat().st_size for f in OUT.rglob("*.json"))
    server.log(f"exported {len(dates)} days to {OUT} ({size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
