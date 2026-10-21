"""Score tomorrow for every cell from the latest weather: the live forecast.

    python ml/forecast.py pack --data private/norcal --cause lightning --out forecast_kit.joblib
    python ml/forecast.py run --kit forecast_kit.joblib --out forecast/

pack bundles what a nightly run needs (the model with its fire history and climate, the cells in the site's
order, the map colour scale) into one file, so the job never needs the training data. run writes
<out>/latest.json and <out>/<target date>.json in the same compact shape as the replay's day files.

Weather: gridMET, the data the model trained on, runs about two days behind. Those days, up to and including
today, come from the Open-Meteo forecast at each cell's centre. A backtest on summer 2024 (Open-Meteo's archived
forecasts) found shifting them by each cell's recent difference from gridMET made the rankings worse, so they're
used as they are.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))
sys.path.insert(0, str(ROOT / "deploy/vercel"))
from build_dataset import WEATHER, gridmet_cells, weather_features  # noqa: E402
from export_norcal import DETAILED, r, reasons  # noqa: E402
from fetch_data import GRIDMET_URL, GRIDMET_VARS, request  # noqa: E402
from regions import REGIONS  # noqa: E402
from train import features, blend  # noqa: E402

TZ = ZoneInfo("America/Los_Angeles")
OPEN_METEO = "https://api.open-meteo.com/v1/forecast"
OM_DAILY = {"temperature_2m_max": "tmax", "relative_humidity_2m_min": "rh", "wind_speed_10m_mean": "wind", "precipitation_sum": "rain"}
GRIDMET = {"tmmx": "tmax", "rmin": "rh", "vs": "wind", "pr": "rain"}
LOOKBACK = 45        # days of gridMET to fetch: the 30-day windows plus gridMET's delay and some slack
WINDOW = 30          # the longest input window (30-day rain): every day in it must have weather
BATCH = 50           # Open-Meteo locations per request


def log(msg: str) -> None:
    print(f"[forecast] {msg}", flush=True)


def centres(cells: list[dict]) -> tuple[list[float], list[float]]:
    return ([round((c["bounds"][0][0] + c["bounds"][1][0]) / 2, 4) for c in cells],
            [round((c["bounds"][0][1] + c["bounds"][1][1]) / 2, 4) for c in cells])


def pack(args) -> None:
    model_dir = args.data if args.cause == "all" else args.data / args.cause
    bundle = joblib.load(model_dir / "model.joblib")
    meta = json.loads((model_dir / "cells.json").read_text())
    overview = json.loads((ROOT / "web/public" / args.site / "overview.json").read_text())
    by_id = {c["id"]: c for c in meta["cells"]}
    # The site keeps per-cell values in its overview's cell order, so the forecast uses the same order.
    cells = [{**by_id[c["id"]], "name": c["name"]} for c in overview["cells"]]
    kit = {"bundle": bundle, "region": meta["region"], "cause": args.cause, "cells": cells,
           "heat_scale": overview["metrics"]["heat_scale"]}
    joblib.dump(kit, args.out, compress=3)
    log(f"packed {len(cells)} cells, {bundle['selected_name']}, into {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")


def fetch_gridmet(region: dict, start: pd.Timestamp, end: pd.Timestamp, folder: Path) -> None:
    west, south, east, north = region["bounds"]
    for year in range(start.year, end.year + 1):
        lo, hi = max(start, pd.Timestamp(f"{year}-01-01")), min(end, pd.Timestamp(f"{year}-12-31"))
        for var in GRIDMET:
            params = {"var": GRIDMET_VARS[var], "north": north, "south": south, "west": west, "east": east,
                      "time_start": f"{lo.date()}T00:00:00Z", "time_end": f"{hi.date()}T00:00:00Z", "accept": "netcdf"}
            data = request(GRIDMET_URL.format(var=var, year=year), params)
            if not data.startswith(b"CDF"):
                raise RuntimeError(f"gridMET {var} {year}: not a netCDF file: {data[:200]!r}")
            (folder / f"{var}_{year}.nc").write_bytes(data)


def gridmet_weather(region: dict, cells: list[dict], start: pd.Timestamp, end: pd.Timestamp) -> dict[str, pd.DataFrame]:
    """Days x cells for each weather variable, averaged over each cell's land pixels like the training data."""
    grid = pd.DataFrame({"cell_id": [c["id"] for c in cells], "row": [c["row"] for c in cells], "column": [c["column"] for c in cells]})
    with tempfile.TemporaryDirectory() as tmp:
        fetch_gridmet(region, start, end, Path(tmp))
        out = {name: gridmet_cells(Path(tmp), var, grid, region)[0] for var, name in GRIDMET.items()}
    out["tmax"] = out["tmax"] - 273.15
    return out


def open_meteo_weather(cells: list[dict]) -> dict[str, pd.DataFrame]:
    """Days x cells from the Open-Meteo forecast at each cell's centre: the past week, today and tomorrow."""
    lat, lon = centres(cells)
    frames = {name: [] for name in OM_DAILY.values()}
    for i in range(0, len(cells), BATCH):
        params = {"latitude": ",".join(map(str, lat[i:i + BATCH])), "longitude": ",".join(map(str, lon[i:i + BATCH])),
                  "daily": ",".join(OM_DAILY), "wind_speed_unit": "ms", "timezone": "America/Los_Angeles",
                  "past_days": 7, "forecast_days": 2}
        for attempt in range(6):
            try:
                raw = json.loads(urllib.request.urlopen(f"{OPEN_METEO}?{urllib.parse.urlencode(params)}", timeout=120).read())
                break
            except OSError as e:
                if attempt == 5:
                    raise
                log(f"Open-Meteo retry after {e}")
                time.sleep(30 * (attempt + 1))
        raw = raw if isinstance(raw, list) else [raw]
        for cell, point in zip(cells[i:i + BATCH], raw):
            daily = point["daily"]
            days = pd.to_datetime(daily["time"])
            for key, name in OM_DAILY.items():
                frames[name].append(pd.Series(daily[key], index=days, name=cell["id"], dtype=float))
    return {name: pd.concat(parts, axis=1) for name, parts in frames.items()}


def combine(gm: dict[str, pd.DataFrame], om: dict[str, pd.DataFrame], today: pd.Timestamp) -> tuple[dict[str, pd.DataFrame], pd.Timestamp]:
    """gridMET up to its last complete day, then Open-Meteo up to today."""
    last = min(f.dropna(how="all").index.max() for f in gm.values())
    out = {name: pd.concat([gm[name].loc[:last], om[name].loc[last + pd.Timedelta(days=1):today, gm[name].columns]])
           for name in WEATHER}
    missing = {n: int(f.loc[today - pd.Timedelta(days=WINDOW):].isna().sum().sum()) for n, f in out.items()}
    if any(missing.values()):
        raise RuntimeError(f"missing weather in the last {WINDOW} days: {missing}")
    return out, last


def run(args) -> None:
    kit = joblib.load(args.kit)
    bundle, cells = kit["bundle"], kit["cells"]
    region = REGIONS[kit["region"]]
    now = datetime.now(timezone.utc)
    today = pd.Timestamp(args.today) if args.today else pd.Timestamp(now.astimezone(TZ).date())
    target = today + pd.Timedelta(days=1)
    log(f"issuing {target.date()} on {today.date()} for {len(cells)} cells ({kit['cause']} fires)")

    gm = gridmet_weather(region, cells, today - pd.Timedelta(days=LOOKBACK), today)
    om = open_meteo_weather(cells)
    weather, gm_last = combine(gm, om, today)
    log(f"gridMET through {gm_last.date()}, Open-Meteo for {(today - gm_last).days} day(s) after that")

    lat, lon = centres(cells)
    angle = 2 * np.pi * (target.dayofyear - 1) / 365.25
    rows = []
    for cell, la, lo in zip(cells, lat, lon):
        w = pd.DataFrame({n: weather[n][cell["id"]] for n in WEATHER})
        w.loc[target] = np.nan  # the target day itself: weather_features reads only the days before it
        x = weather_features(w).iloc[-1].to_dict()
        rows.append({"target_date": target, "cell_id": cell["id"], "latitude": la, "longitude": lo,
                     "season_sin": np.sin(angle), "season_cos": np.cos(angle), "month": target.month, **x})
    rows = pd.DataFrame(rows)
    X = features(rows, bundle["history"], bundle["features"])
    rows["score"] = blend(bundle["model"], X, bundle["weather_weight"])
    for c in X.columns:
        if c not in rows:
            rows[c] = X[c].to_numpy()
    rows["rank"] = rows.score.rank(method="first", ascending=False).astype(int)

    prior_high = 0.3 / kit["heat_scale"]
    lightning = kit["cause"] == "lightning"
    top = rows.sort_values("rank").head(DETAILED)
    date = target.strftime("%Y-%m-%d")
    payload = {
        "date": date, "issued": today.strftime("%Y-%m-%d"),
        "score": [r(v) for v in rows.score], "fires_by_cell": {},
        "weather": {"tmax": [r(v, 1) for v in rows.tmax_lag1], "rh": [r(v, 0) for v in rows.rhmin_lag1],
                    "wind": [r(v, 1) for v in rows.windmax_lag1], "rain30": [r(v, 1) for v in rows.rain_sum30],
                    "dry_days": [int(v) for v in rows.dry_days], "past_rate": [r(v) for v in rows.historical_prior]},
        "anomaly": {"heat": [r(v, 2) for v in rows.tmax_lag1_anomaly], "humidity": [r(v, 2) for v in rows.rhmin_lag1_anomaly],
                    "wind": [r(v, 2) for v in rows.windmax_lag1_anomaly], "rain30": [r(v, 2) for v in rows.rain_sum30_anomaly]},
        "reasons": {int(i): reasons(row, prior_high, lightning) for i, row in top.iterrows()},
        "fires": [],
        "forecast": {"issued_at": now.isoformat(timespec="seconds"), "cause": kit["cause"], "model": bundle["selected_name"],
                     "gridmet_through": gm_last.strftime("%Y-%m-%d"),
                     "open_meteo_days": [d.strftime("%Y-%m-%d") for d in pd.date_range(gm_last + pd.Timedelta(days=1), today)]},
    }
    args.out.mkdir(parents=True, exist_ok=True)
    for name in ("latest.json", f"{date}.json"):
        (args.out / name).write_text(json.dumps(payload, allow_nan=False, separators=(",", ":")), encoding="utf-8")
    best = rows.sort_values("rank").head(3)
    log(f"wrote {args.out / 'latest.json'}; top 3: " + ", ".join(f"{c['name']} {s:.3f}" for c, s in
        zip([cells[i] for i in best.index], best.score)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("pack", help="bundle the model and site metadata for nightly runs")
    p.add_argument("--data", type=Path, required=True, help="region data folder (ml/build_dataset.py)")
    p.add_argument("--cause", default="lightning")
    p.add_argument("--site", default="api-lightning", help="the site's data folder under web/public/, for cell order and colours")
    p.add_argument("--out", type=Path, default=Path("forecast_kit.joblib"))
    p.set_defaults(func=pack)
    p = sub.add_parser("run", help="score tomorrow")
    p.add_argument("--kit", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--today", help="issue as if today were this date (YYYY-MM-DD), for testing")
    p.set_defaults(func=run)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
