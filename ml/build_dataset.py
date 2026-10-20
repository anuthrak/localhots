"""Build the cell-day table the model trains on: one row per cell per day, weather inputs from the days
before, and whether a fire was discovered that day.

    python ml/build_dataset.py --region norcal --data <dir from fetch_data.py>
    python ml/build_dataset.py --region norcal24 --data <wildfire-outlook>/datasets --check

The first writes <data>/model_inputs.csv.gz and <data>/cells.json; with --cause lightning (or human) the
label counts only those fires and the files go to <data>/lightning/. --check rebuilds the datathon table
from its original Open-Meteo downloads and compares it with the team's model_inputs.csv.gz.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import netcdf_file

from regions import END, REGIONS, START

DRY_MM = 1.0        # a day with less rain than this counts as dry
DRY_CAP = 30        # dry-day streaks are capped
MIN_LAND = 0.25     # gridMET cells need at least this share of land pixels (drops ocean cells)
SPLITS = [(2018, "train"), (2021, "validation"), (2024, "test")]
WEATHER = ["tmax", "rh", "wind", "rain"]
GRIDMET = {"tmmx": "tmax", "rmin": "rh", "vs": "wind", "pr": "rain"}
CAUSES = {"all": None, "human": "Human", "lightning": "Natural"}  # FPA FOD nwcg_cause_classification
COLUMNS = ["target_date", "cell_id", "latitude", "longitude", "feature_window_end", "season_sin", "season_cos",
           "tmax_lag1", "tmax_mean7", "tmax_max7", "rhmin_lag1", "rhmin_mean7", "rhmin_min7", "windmax_lag1",
           "windmax_max3", "rain_lag1", "rain_sum7", "rain_sum14", "rain_sum30", "dry_days", "recorded_fires",
           "fire_next_day", "prediction_date", "split"]


def log(msg: str) -> None:
    print(f"[build] {msg}", flush=True)


def make_grid(region: dict) -> pd.DataFrame:
    west, south, east, north = region["bounds"]
    step = region["step"]
    rows, cols = round((north - south) / step), round((east - west) / step)
    cells = [{"cell_id": f"r{r}c{c}", "row": r, "column": c, "latitude": south + (r + 0.5) * step,
              "longitude": west + (c + 0.5) * step} for r in range(rows) for c in range(cols)]
    return pd.DataFrame(cells)


def cell_of(lat, lon, region: dict) -> tuple[np.ndarray, np.ndarray]:
    west, south, east, north = region["bounds"]
    step = region["step"]
    return np.floor((np.asarray(lat) - south) / step).astype(int), np.floor((np.asarray(lon) - west) / step).astype(int)


def weather_features(w: pd.DataFrame) -> pd.DataFrame:
    """Inputs for each target day from the weather up to the day before (columns tmax, rh, wind, rain)."""
    s = w.shift(1)
    dry = (w.rain < DRY_MM).astype(int)
    streak = dry.groupby((dry == 0).cumsum()).cumsum().clip(upper=DRY_CAP)
    return pd.DataFrame({
        "tmax_lag1": s.tmax, "tmax_mean7": s.tmax.rolling(7).mean(), "tmax_max7": s.tmax.rolling(7).max(),
        "rhmin_lag1": s.rh, "rhmin_mean7": s.rh.rolling(7).mean(), "rhmin_min7": s.rh.rolling(7).min(),
        "windmax_lag1": s.wind, "windmax_max3": s.wind.rolling(3).max(),
        "rain_lag1": s.rain, "rain_sum7": s.rain.rolling(7).sum(), "rain_sum14": s.rain.rolling(14).sum(),
        "rain_sum30": s.rain.rolling(30).sum(), "dry_days": streak.shift(1).astype(float),
    }, index=w.index)


def openmeteo_weather(data: Path, grid: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {}
    for cell in grid.cell_id:
        daily = json.loads((data / f"raw/original/weather/{cell}.json").read_text())["response"]["daily"]
        w = pd.DataFrame(daily).set_index("time")
        w.index = pd.to_datetime(w.index)
        w.columns = WEATHER
        out[cell] = w
    return out


def read_gridmet(path: Path) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray]:
    """Days, pixel latitudes, longitudes and values (day x lat x lon, NaN where missing)."""
    with netcdf_file(path, mmap=False) as f:
        var = next(v for k, v in f.variables.items() if k not in ("day", "lat", "lon", "crs"))
        days = pd.to_datetime("1900-01-01") + pd.to_timedelta(f.variables["day"].data.astype(int), unit="D")
        lat, lon = f.variables["lat"].data.astype(float), f.variables["lon"].data.astype(float)
        raw = var.data
        missing = raw == getattr(var, "_FillValue", None)
        if getattr(var, "_Unsigned", b"") == b"true":
            raw = raw.view(raw.dtype.str.replace("i", "u"))
        values = raw.astype(np.float32) * getattr(var, "scale_factor", 1.0) + getattr(var, "add_offset", 0.0)
        values[missing] = np.nan
    # Axis order isn't labelled reliably by scipy, so match axes by length (the box is never square).
    sizes = {len(days): "day", len(lat): "lat", len(lon): "lon"}
    if len(sizes) != 3:
        raise ValueError(f"{path.name}: can't tell axes apart by length")
    order = [sizes[n] for n in values.shape]
    values = np.transpose(values, [order.index(a) for a in ("day", "lat", "lon")])
    return days, lat, lon, values


def gridmet_cells(folder: Path, var: str, grid: pd.DataFrame, region: dict) -> tuple[pd.DataFrame, pd.Series]:
    """One gridMET variable averaged over each cell's land pixels (days x cells), and each cell's land share.
    grid needs row and column for every cell of the region's full grid."""
    parts, land = [], None
    rows, cols = round((region["bounds"][3] - region["bounds"][1]) / region["step"]), round((region["bounds"][2] - region["bounds"][0]) / region["step"])
    for path in sorted(folder.glob(f"{var}_*.nc")):
        days, lat, lon, values = read_gridmet(path)
        r, c = cell_of(lat[:, None].repeat(len(lon), 1), lon[None, :].repeat(len(lat), 0), region)
        inside = (r >= 0) & (r < rows) & (c >= 0) & (c < cols)
        onehot = np.zeros((inside.sum(), rows * cols), dtype=np.float32)
        onehot[np.arange(inside.sum()), (r * cols + c)[inside]] = 1
        onehot = onehot[:, (grid.row * cols + grid.column).to_numpy()]
        v = values[:, inside]
        sums = np.nan_to_num(v) @ onehot
        counts = (~np.isnan(v)).astype(np.float32) @ onehot
        with np.errstate(invalid="ignore", divide="ignore"):
            parts.append(pd.DataFrame(sums / counts, index=days, columns=grid.cell_id.to_numpy()))
        if land is None:
            land = pd.Series(counts[0] / onehot.sum(0), index=grid.cell_id.to_numpy())
        log(f"read {path.name}")
    frame = pd.concat(parts).sort_index()
    return frame[~frame.index.duplicated()], land


def gridmet_weather(data: Path, grid: pd.DataFrame, region: dict) -> tuple[dict[str, pd.DataFrame], pd.Series]:
    """Average gridMET pixels over each cell; returns per-cell daily weather and each cell's land share."""
    per_var, land = {}, None
    for var, name in GRIDMET.items():
        per_var[name], share = gridmet_cells(data / "gridmet", var, grid, region)
        land = share if land is None else land
    per_var["tmax"] = per_var["tmax"] - 273.15  # kelvin to Celsius
    weather = {cell: pd.DataFrame({n: per_var[n][cell] for n in WEATHER}) for cell in grid.cell_id}
    return weather, land


def fire_days(fires: pd.DataFrame, region: dict, grid: pd.DataFrame) -> pd.DataFrame:
    """Fires per cell per day, from FPA FOD points (discovery dates are stored as UTC midnight)."""
    f = fires.assign(target_date=pd.to_datetime(fires.discovery_date, unit="ms").dt.normalize())
    r, c = cell_of(f.latitude, f.longitude, region)
    f["cell_id"] = [f"r{a}c{b}" for a, b in zip(r, c)]
    f = f[f.cell_id.isin(set(grid.cell_id)) & f.target_date.between(START, END)]
    return f.groupby(["cell_id", "target_date"]).size().rename("recorded_fires").reset_index()


def build(grid: pd.DataFrame, weather: dict[str, pd.DataFrame], fires: pd.DataFrame) -> pd.DataFrame:
    dates = pd.date_range(START, END)
    parts = []
    for cell in grid.itertuples():
        x = weather_features(weather[cell.cell_id]).reindex(dates)
        x.insert(0, "target_date", dates)
        x.insert(1, "cell_id", cell.cell_id)
        x.insert(2, "latitude", cell.latitude)
        x.insert(3, "longitude", cell.longitude)
        parts.append(x)
    table = pd.concat(parts, ignore_index=True)
    angle = 2 * math.pi * (table.target_date.dt.dayofyear - 1) / 365.25
    table["season_sin"], table["season_cos"] = np.sin(angle), np.cos(angle)
    table["feature_window_end"] = table["prediction_date"] = table.target_date - pd.Timedelta(days=1)
    table = table.merge(fires, on=["cell_id", "target_date"], how="left")
    table["recorded_fires"] = table.recorded_fires.fillna(0).astype(int)
    table["fire_next_day"] = (table.recorded_fires > 0).astype(int)
    year = table.target_date.dt.year
    table["split"] = np.select([year <= y for y, _ in SPLITS], [s for _, s in SPLITS], default="")
    return table[COLUMNS]


def check(data: Path, table: pd.DataFrame) -> None:
    original = pd.read_csv(data / "model_inputs.csv.gz", float_precision="round_trip",
                           parse_dates=["target_date", "feature_window_end", "prediction_date"])
    key = ["target_date", "cell_id"]
    a, b = table.sort_values(key).reset_index(drop=True), original.sort_values(key).reset_index(drop=True)
    log(f"rows: rebuilt {len(a):,}, original {len(b):,}")
    worst = 0.0
    for col in COLUMNS:
        if pd.api.types.is_numeric_dtype(b[col]):
            diff = float((a[col] - b[col]).abs().max())
            worst = max(worst, diff)
            log(f"{col:20s} max |diff| {diff:.3g}")
        else:
            bad = int((a[col] != b[col]).sum())
            worst = max(worst, bad)
            log(f"{col:20s} mismatches {bad}")
    log("MATCH" if worst < 1e-9 else "DIFFERENT")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--region", choices=REGIONS, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--check", action="store_true", help="compare with the datathon table (norcal24 only)")
    parser.add_argument("--cause", choices=CAUSES, default="all",
                        help="label only fires of this NWCG cause class; writes to <data>/<cause>/")
    args = parser.parse_args()
    region = REGIONS[args.region]
    grid = make_grid(region)

    if args.region == "norcal24":
        weather = openmeteo_weather(args.data, grid)
        fires = pd.read_csv(args.data / "raw/original/fires.csv")
    else:
        weather, land = gridmet_weather(args.data, grid, region)
        grid["land"] = grid.cell_id.map(land).round(3)
        grid = grid[grid.land >= MIN_LAND].reset_index(drop=True)
        log(f"{len(grid)} cells with at least {MIN_LAND:.0%} land")
        fires = pd.read_csv(args.data / "fires.csv")
    out = args.data
    if args.cause != "all":
        fires = fires[fires.nwcg_cause_classification == CAUSES[args.cause]]
        out = args.data / args.cause
        out.mkdir(exist_ok=True)

    table = build(grid, weather, fire_days(fires, region, grid))
    log(f"{len(table):,} rows, {table.fire_next_day.mean():.4f} with a fire")
    if args.check:
        check(args.data, table)
        return
    if table.drop(columns="recorded_fires").isna().any().any():
        raise ValueError("missing weather in the table:\n" + table.isna().sum()[lambda s: s > 0].to_string())
    table.to_csv(out / "model_inputs.csv.gz", index=False)
    step = region["step"]
    cells = [{"id": c.cell_id, "row": int(c.row), "column": int(c.column), "land": float(c.land),
              "bounds": [[c.latitude - step / 2, c.longitude - step / 2], [c.latitude + step / 2, c.longitude + step / 2]]}
             for c in grid.itertuples()]
    (out / "cells.json").write_text(json.dumps({"region": args.region, "cause": args.cause, **region, "cells": cells}, indent=1))
    log(f"wrote model_inputs.csv.gz and cells.json to {out}")


if __name__ == "__main__":
    main()
