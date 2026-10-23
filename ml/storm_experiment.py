"""Do thunderstorm forecasts help the lightning model pick *where* fires start on a lightning day?

    python ml/storm_experiment.py --data private/norcal

ml/forecast_experiment.py found the model knows *when* (which days) far better than *where*: on days with a
lightning fire, it ranks the burning cells little better than fire history. Its inputs say how dry each cell is,
nothing about where storms will be. This takes NOAA's HRRR forecasts (3 km, archived on AWS) from the 18 UTC run
the day before each target day, about what a nightly run at 6 pm Pacific would have, and averages them over each
cell:

    ltng_max, ltng_mean   HRRR's forecast lightning (flash density) over the target day: highest, and mean
    ltng_prev_max         the same for the issue day's afternoon and evening (forecast hours 0-12): lightning
                          fires are often found a day or more after the strike
    cape_max              surface CAPE, every 3 hours over the target day
    rain                  forecast rain on the target day (wet storms rarely start fires)

It fits a small logistic regression on the model's score plus each signal (log1p, then standardised within the
day) on the 2022-2023 fire seasons' lightning fire days and tests it on 2024's. Only *where* is scored, so only
fire days are fetched. The model's score uses gridMET (the training weather), so the test isolates the storm
signals from the forecast-rain loss studied in ml/rain_points.py.

Set before running: it counts as a win only if model+storms lifts 2024 top-37 recall with a 90% interval above
zero and raises within-day AUC. Otherwise, stop tuning the model's inputs. (A dry run with random signals moved
recall by up to 1.7 points.)

Only the needed fields are downloaded (byte ranges from each file's index), about 10 MB per day, cached per day
under <data>/hrrr_storms/.
"""
from __future__ import annotations

import argparse
import json
import re
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import eccodes
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from build_dataset import cell_of
from forecast_experiment import day_groups, interval, top_k
from regions import REGIONS
from train import blend, features, load

HRRR = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.{day:%Y%m%d}/conus/hrrr.t18z.wrfsfcf{hour:02d}.grib2"
TARGET_HOURS = range(13, 37)   # from 18 UTC the day before: 07 UTC (midnight Pacific in summer) to 06 UTC the day after
PREV_HOURS = range(0, 13)      # the issue day, 11 am to 11 pm Pacific
CAPE_HOURS = range(13, 37, 3)
SEASON = [6, 7, 8, 9]
FIT, TEST = (2022, 2023), (2024,)
SIGNALS = ["ltng_max", "ltng_mean", "ltng_prev_max", "cape_max", "rain"]
THREADS = 8
DRAWS = 500


def log(msg: str) -> None:
    print(f"[storms] {msg}", flush=True)


def get(url: str, start: int | None = None, end: int | None = None) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"} if start is not None else {})
    for attempt in range(5):
        try:
            return urllib.request.urlopen(req, timeout=120).read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise
            if attempt == 4:
                raise
        except OSError:
            if attempt == 4:
                raise


def field(url: str, pattern: str) -> np.ndarray | None:
    """One GRIB field from an HRRR file, found by matching its index line; None if the file or field is missing."""
    try:
        idx = get(url + ".idx").decode().splitlines()
    except urllib.error.HTTPError:
        return None
    for i, line in enumerate(idx):
        if re.search(pattern, line):
            start = int(line.split(":")[1])
            end = int(idx[i + 1].split(":")[1]) - 1 if i + 1 < len(idx) else None
            h = eccodes.codes_new_from_message(get(url, start, end))
            try:
                return eccodes.codes_get_values(h)
            finally:
                eccodes.codes_release(h)
    return None


def grid_cells(region: dict, cell_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """For each HRRR grid point in a cell: its position in the grid, and which cell (index into cell_ids)."""
    url = HRRR.format(day=pd.Timestamp("2024-07-01"), hour=0)
    idx = get(url + ".idx").decode().splitlines()
    line = next(i for i, l in enumerate(idx) if ":LTNG:" in l)
    h = eccodes.codes_new_from_message(get(url, int(idx[line].split(":")[1]), int(idx[line + 1].split(":")[1]) - 1))
    lat, lon = eccodes.codes_get_array(h, "latitudes"), eccodes.codes_get_array(h, "longitudes")
    eccodes.codes_release(h)
    lon = np.where(lon > 180, lon - 360, lon)
    r, c = cell_of(lat, lon, region)
    where = {cid: i for i, cid in enumerate(cell_ids)}
    owner = np.array([where.get(f"r{a}c{b}", -1) for a, b in zip(r, c)])
    keep = np.flatnonzero(owner >= 0)
    return keep, owner[keep]


def day_storms(target: pd.Timestamp, keep: np.ndarray, owner: np.ndarray, n: int, pool: ThreadPoolExecutor) -> pd.DataFrame | None:
    """SIGNALS for every cell on one target day, or None if the run is missing from the archive."""
    run = target - pd.Timedelta(days=1)
    url = lambda hour: HRRR.format(day=run, hour=hour)
    ltng = list(pool.map(lambda h: field(url(h), r":LTNG:entire atmosphere:"), [*PREV_HOURS, *TARGET_HOURS]))
    cape = list(pool.map(lambda h: field(url(h), r":CAPE:surface:"), CAPE_HOURS))
    acc = list(pool.map(lambda h: field(url(h), rf":APCP:surface:0-{h} hour acc"), (12, 36)))
    if any(a is None for a in ltng + cape + acc):
        return None
    cell_mean = lambda v: np.bincount(owner, v[keep], n) / np.bincount(owner, minlength=n)
    cell_max = lambda v: pd.Series(v[keep]).groupby(owner).max().reindex(range(n)).to_numpy()
    prev, day = np.max(ltng[:len(PREV_HOURS)], axis=0), np.array(ltng[len(PREV_HOURS):])
    return pd.DataFrame({"ltng_max": cell_max(day.max(axis=0)), "ltng_mean": cell_mean(day.mean(axis=0)),
                         "ltng_prev_max": cell_max(prev), "cape_max": np.max([cell_mean(v) for v in cape], axis=0),
                         "rain": cell_mean(acc[1] - acc[0])})


def fetch_storms(region: dict, cell_ids: list[str], days: list[pd.Timestamp], folder: Path) -> pd.DataFrame:
    folder.mkdir(parents=True, exist_ok=True)
    keep, owner = grid_cells(region, cell_ids)
    log(f"{len(keep)} HRRR grid points in {len(cell_ids)} cells")
    with ThreadPoolExecutor(THREADS) as pool:
        for i, day in enumerate(days):
            path = folder / f"{day.date()}.csv"
            if path.exists():
                continue
            d = day_storms(day, keep, owner, len(cell_ids), pool)
            if d is None:
                log(f"{day.date()}: the HRRR run is missing from the archive; skipping")
                continue
            d.assign(target_date=day, cell_id=cell_ids).to_csv(path, index=False)
            if i % 10 == 0:
                log(f"fetched {i + 1} of {len(days)} days")
    files = sorted(folder.glob("*.csv"))
    return pd.concat([pd.read_csv(p, parse_dates=["target_date"]) for p in files])


def within_day_z(rows: pd.DataFrame, col: str) -> np.ndarray:
    v = np.log1p(rows[col].clip(lower=0))
    g = v.groupby(rows.target_date)
    std = g.transform("std").replace(0, np.nan)
    return ((v - g.transform("mean")) / std).fillna(0).to_numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", type=Path, required=True, help="region data folder; the lightning model is in <data>/lightning/")
    args = parser.parse_args()
    folder = args.data / "lightning"
    bundle, data = joblib.load(folder / "model.joblib"), load(folder)
    meta = json.loads((folder / "cells.json").read_text())
    k = bundle["review_k"]

    test = data[data.year.isin(FIT + TEST) & data.month.isin(SEASON)]
    fire_days = sorted(pd.Timestamp(d) for d in test.loc[test.fire_next_day == 1, "target_date"].unique())
    rows = test[test.target_date.isin(fire_days)].reset_index(drop=True)
    log(f"{len(fire_days)} lightning fire days in the {FIT + TEST} fire seasons")
    storms = fetch_storms(REGIONS[meta["region"]], sorted(rows.cell_id.unique()), fire_days, args.data / "hrrr_storms")

    rows = rows.merge(storms, on=["target_date", "cell_id"], how="left")
    missing = rows[SIGNALS].isna().any(axis=1)
    if missing.any():
        log(f"no storm forecast for {int(missing.sum())} cell-days on {rows.loc[missing, 'target_date'].nunique()} days; dropping those days")
        rows = rows[~rows.target_date.isin(rows.loc[missing, "target_date"])].reset_index(drop=True)
    base = blend(bundle["model"], features(rows, bundle["history"], bundle["features"]), bundle["weather_weight"])
    p = np.clip(base, 1e-6, 1 - 1e-6)
    X = pd.DataFrame({"model": np.log(p / (1 - p)), **{s: within_day_z(rows, s) for s in SIGNALS}})
    y = rows.fire_next_day.to_numpy()
    fit, held = rows.year.isin(FIT).to_numpy(), rows.year.isin(TEST).to_numpy()
    log(f"fit on {int(y[fit].sum())} fire cell-days ({rows[fit].target_date.nunique()} days), test on {int(y[held].sum())} "
        f"({rows[held].target_date.nunique()} days)")

    variants, coefs = {"model": base[held],
                       "history_only": features(rows, bundle["history"], bundle["features"]).historical_prior.to_numpy()[held]}, {}
    for name, cols in [*[(f"model+{s}", ["model", s]) for s in SIGNALS], ("model+storms", ["model", *SIGNALS])]:
        lr = LogisticRegression(C=1.0).fit(X.loc[fit, cols], y[fit])
        variants[name] = lr.decision_function(X.loc[held, cols])
        coefs[name] = dict(zip(cols, np.round(lr.coef_[0], 4)))
    for s in SIGNALS:  # each signal on its own: more storm = higher, except rain
        variants[f"{s}_alone"] = X.loc[held, s].to_numpy() * (-1 if s == "rain" else 1)
    log("fitted on 2022-2023 fire days: " + json.dumps(coefs, default=float))

    t = rows[held].reset_index(drop=True)
    yt = t.fire_next_day.to_numpy()
    groups = day_groups(t)
    rng = np.random.default_rng(42)
    picks = [rng.integers(0, len(groups), len(groups)) for _ in range(DRAWS)]
    hits = {n: top_k(t, s, k) & (yt == 1) for n, s in variants.items()}
    table = []
    for name, s in variants.items():
        aucs = [roc_auc_score(yt[g], s[g]) for g in groups if 0 < yt[g].sum() < len(g)]
        gap = []
        for pick in picks:
            idx = np.concatenate([groups[i] for i in pick])
            gap.append((hits[name][idx].sum() - hits["model"][idx].sum()) / yt[idx].sum())
        table.append({"score": name, "top_k_recall": round(hits[name].sum() / yt.sum(), 4), "within_day_auc": round(float(np.mean(aucs)), 4),
                      "recall_vs_model_90ci": interval(gap)})
    log(f"2024: {int(yt.sum())} lightning fire cell-days on {len(groups)} days, top {k}:\n" + pd.DataFrame(table).to_string(index=False))
    (folder / "storm_experiment.json").write_text(json.dumps({"fit": FIT, "test": TEST, "signals": SIGNALS, "coefficients": coefs,
                                                               "scores": table}, indent=2, default=float))
    log(f"wrote {folder / 'storm_experiment.json'}")


if __name__ == "__main__":
    main()
