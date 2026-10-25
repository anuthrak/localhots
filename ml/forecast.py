"""Score tomorrow for every cell from the latest weather: the live forecast.

    python ml/forecast.py pack --data private/norcal --cause lightning --out forecast_kit.joblib
    python ml/forecast.py run --kit forecast_kit.joblib --out forecast/

pack bundles what a nightly run needs (the model with its fire history and climate, the cells in the site's
order, the map colour scale) into one file, so the job never needs the training data. run writes
<out>/latest.json and <out>/<target date>.json in the same compact shape as the replay's day files.

Weather: gridMET, the data the model trained on, runs about two days behind. Those days, up to and including
today, come from the Open-Meteo forecast at each cell's centre, with rain averaged over the centre and the
centres of the cell's four quarters: storm rain is patchy and gridMET averages it over the cell, so a single
point mostly misses it (ml/rain_points.py: top-37 recall on summer 2024 37% -> 42%, gridMET 44%). A backtest on summer 2024 (Open-Meteo's archived
forecasts) found shifting them by each cell's recent difference from gridMET made the rankings worse, so they're
used as they are.

Storms: the model's inputs say how dry each cell is, not where storms will be, so each score is nudged by NOAA
HRRR's storm forecast for the target day (ml/hrrr.py), with weights from ml/storm_experiment.py, which found it
lifts summer 2024's top-37 recall on the nightly weather from 42% to 53%. If HRRR's run isn't on AWS, the
forecast goes out without it.
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
from hrrr import SIGNALS, standardise, storms  # noqa: E402
from regions import REGIONS  # noqa: E402
from train import features, blend  # noqa: E402

TZ = ZoneInfo("America/Los_Angeles")
OPEN_METEO = "https://api.open-meteo.com/v1/forecast"
OM_DAILY = {"temperature_2m_max": "tmax", "relative_humidity_2m_min": "rh", "wind_speed_10m_mean": "wind", "precipitation_sum": "rain"}
GRIDMET = {"tmmx": "tmax", "rmin": "rh", "vs": "wind", "pr": "rain"}
LOOKBACK = 45        # days of gridMET to fetch: the 30-day windows plus gridMET's delay and some slack
WINDOW = 30          # the longest input window (30-day rain): every day in it must have weather
BATCH = 50           # Open-Meteo locations per request
PACE = 6             # seconds between requests: Open-Meteo allows 600 locations a minute
OM_TIMEOUT = 90      # seconds per Open-Meteo request: a hung handshake should retry, not use up the night's job
OFFSETS = [(0, 0), (-0.25, -0.25), (-0.25, 0.25), (0.25, -0.25), (0.25, 0.25)]  # rain points, in cell widths from the centre


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
    storm_fit = model_dir / "storm_experiment.json"
    if storm_fit.exists():
        # The blend's weights relative to the model's, so the adjusted score keeps the model's scale.
        c = json.loads(storm_fit.read_text())["coefficients"]["model+storms"]
        kit["storm_weights"] = {s: c[s] / c["model"] for s in SIGNALS}
        log(f"storm weights from {storm_fit}: {kit['storm_weights']}")
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


def points(cells: list[dict], step: float) -> list[dict]:
    """Pseudo-cells whose centres are each cell's rain points (id 'r3c4#2'; '#0' is the centre)."""
    out = []
    for c in cells:
        lat, lon = (c["bounds"][0][0] + c["bounds"][1][0]) / 2, (c["bounds"][0][1] + c["bounds"][1][1]) / 2
        for j, (dy, dx) in enumerate(OFFSETS):
            y, x = lat + dy * step, lon + dx * step
            out.append({"id": f"{c['id']}#{j}", "bounds": [[y, x], [y, x]]})
    return out


def open_meteo_weather(cells: list[dict], url: str = OPEN_METEO, step: float | None = None, **when) -> dict[str, pd.DataFrame]:
    """Days x cells from the Open-Meteo forecast at each cell's centre: by default the past week, today and
    tomorrow; when= takes other date parameters (start_date and end_date for archived forecasts). With step
    (the grid's cell width in degrees), rain is the mean over each cell's rain points (OFFSETS)."""
    if step:
        om = open_meteo_weather(points(cells, step), url, **when)
        ids = [c["id"] for c in cells]
        out = {name: f[[f"{i}#0" for i in ids]].set_axis(ids, axis=1) for name, f in om.items()}
        out["rain"] = om["rain"].T.groupby(lambda pid: pid.split("#")[0]).mean().T[ids]
        return out
    lat, lon = centres(cells)
    frames = {name: [] for name in OM_DAILY.values()}
    for i in range(0, len(cells), BATCH):
        if i:
            time.sleep(PACE)
        params = {"latitude": ",".join(map(str, lat[i:i + BATCH])), "longitude": ",".join(map(str, lon[i:i + BATCH])),
                  "daily": ",".join(OM_DAILY), "wind_speed_unit": "ms", "timezone": "America/Los_Angeles",
                  **(when or {"past_days": 7, "forecast_days": 2})}
        for attempt in range(6):
            try:
                raw = json.loads(urllib.request.urlopen(f"{url}?{urllib.parse.urlencode(params)}", timeout=OM_TIMEOUT).read())
                break
            except (OSError, ValueError) as e:  # ValueError: a response cut off mid-way
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


def score_day(kit: dict, weather: dict[str, pd.DataFrame], target: pd.Timestamp, storm: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per cell (in the kit's order) with the model inputs for target, its score and its rank, from
    daily weather (days x cells) covering at least the 30 days before target. With storm (HRRR's SIGNALS for
    target, in the kit's order) and the kit's storm weights, the score is nudged by the storm forecast."""
    bundle, cells = kit["bundle"], kit["cells"]
    lat, lon = centres(cells)
    angle = 2 * np.pi * (target.dayofyear - 1) / 365.25
    rows = []
    for cell, la, lo in zip(cells, lat, lon):
        w = pd.DataFrame({n: weather[n][cell["id"]] for n in WEATHER}).loc[:target - pd.Timedelta(days=1)]
        w.loc[target] = np.nan  # the target day itself: weather_features reads only the days before it
        x = weather_features(w).iloc[-1].to_dict()
        rows.append({"target_date": target, "cell_id": cell["id"], "latitude": la, "longitude": lo,
                     "season_sin": np.sin(angle), "season_cos": np.cos(angle), "month": target.month, **x})
    rows = pd.DataFrame(rows)
    X = features(rows, bundle["history"], bundle["features"])
    rows["score"] = rows["model_score"] = blend(bundle["model"], X, bundle["weather_weight"])
    if storm is not None and kit.get("storm_weights"):
        p = rows.score.clip(1e-6, 1 - 1e-6)
        nudge = sum(w * standardise(storm[s]) for s, w in kit["storm_weights"].items())
        rows["score"] = 1 / (1 + np.exp(-(np.log(p / (1 - p)) + nudge)))
        rows["ltng"] = storm.ltng_max.to_numpy()
    for c in X.columns:
        if c not in rows:
            rows[c] = X[c].to_numpy()
    rows["rank"] = rows.score.rank(method="first", ascending=False).astype(int)
    return rows


def run(args) -> None:
    kit = joblib.load(args.kit)
    bundle, cells = kit["bundle"], kit["cells"]
    region = REGIONS[kit["region"]]
    now = datetime.now(timezone.utc)
    today = pd.Timestamp(args.today) if args.today else pd.Timestamp(now.astimezone(TZ).date())
    target = today + pd.Timedelta(days=1)
    log(f"issuing {target.date()} on {today.date()} for {len(cells)} cells ({kit['cause']} fires)")

    gm = gridmet_weather(region, cells, today - pd.Timedelta(days=LOOKBACK), today)
    om = open_meteo_weather(cells, step=region["step"])
    weather, gm_last = combine(gm, om, today)
    log(f"gridMET through {gm_last.date()}, Open-Meteo for {(today - gm_last).days} day(s) after that")
    storm = storms(region, [c["id"] for c in cells], [target], log=log) if kit.get("storm_weights") else None
    storm = storm if storm is not None and len(storm) else None
    log(f"HRRR storms: {'used' if storm is not None else 'not used'}")
    rows = score_day(kit, weather, target, storm)

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
        "reasons": {int(i): (["Thunderstorms with lightning forecast here"] if row.get("ltng", 0) > 0 else [])
                    + reasons(row, prior_high, lightning)[:2 if row.get("ltng", 0) > 0 else 3] for i, row in top.iterrows()},
        "fires": [],
        "forecast": {"issued_at": now.isoformat(timespec="seconds"), "cause": kit["cause"], "model": bundle["selected_name"],
                     "gridmet_through": gm_last.strftime("%Y-%m-%d"),
                     "open_meteo_days": [d.strftime("%Y-%m-%d") for d in pd.date_range(gm_last + pd.Timedelta(days=1), today)],
                     "storms": storm is not None},
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
