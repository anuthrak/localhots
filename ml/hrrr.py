"""Storm forecasts from NOAA's HRRR (3 km, archived and real-time on AWS), averaged over each cell.

For a target day, the 18 UTC run the day before: about what the nightly run at 6 pm Pacific has. Only the
needed fields are downloaded (byte ranges from each file's index), about 10 MB per day.

    ltng_max, ltng_mean   forecast lightning (flash density) over the target day: highest, and mean
    ltng_prev_max         the same for the issue day's afternoon and evening (forecast hours 0-12)
    cape_max              surface CAPE, every 3 hours over the target day
    rain                  forecast rain on the target day
"""
from __future__ import annotations

import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import eccodes
import numpy as np
import pandas as pd

from build_dataset import cell_of

URL = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.{day:%Y%m%d}/conus/hrrr.t18z.wrfsfcf{hour:02d}.grib2"
TARGET_HOURS = range(13, 37)   # from 18 UTC the day before: 07 UTC (midnight Pacific in summer) to 06 UTC the day after
PREV_HOURS = range(0, 13)      # the issue day, 11 am to 11 pm Pacific
CAPE_HOURS = range(13, 37, 3)
SIGNALS = ["ltng_max", "ltng_mean", "ltng_prev_max", "cape_max", "rain"]
THREADS = 8


def get(url: str, start: int | None = None, end: int | None = None) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"} if start is not None else {})
    for attempt in range(5):
        try:
            return urllib.request.urlopen(req, timeout=120).read()
        except urllib.error.HTTPError as e:
            if e.code == 404 or attempt == 4:
                raise
        except OSError:
            if attempt == 4:
                raise
        time.sleep(5 * (attempt + 1))


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


@lru_cache
def grid_cells(bounds: tuple, step: float, cell_ids: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    """For each HRRR grid point in a cell: its position in the grid, and which cell (index into cell_ids)."""
    url = URL.format(day=pd.Timestamp("2024-07-01"), hour=0)
    idx = get(url + ".idx").decode().splitlines()
    line = next(i for i, l in enumerate(idx) if ":LTNG:" in l)
    h = eccodes.codes_new_from_message(get(url, int(idx[line].split(":")[1]), int(idx[line + 1].split(":")[1]) - 1))
    lat, lon = eccodes.codes_get_array(h, "latitudes"), eccodes.codes_get_array(h, "longitudes")
    eccodes.codes_release(h)
    lon = np.where(lon > 180, lon - 360, lon)
    r, c = cell_of(lat, lon, {"bounds": bounds, "step": step})
    where = {cid: i for i, cid in enumerate(cell_ids)}
    owner = np.array([where.get(f"r{a}c{b}", -1) for a, b in zip(r, c)])
    keep = np.flatnonzero(owner >= 0)
    return keep, owner[keep]


def day_storms(region: dict, cell_ids: list[str], target: pd.Timestamp, pool: ThreadPoolExecutor) -> pd.DataFrame | None:
    """SIGNALS for every cell (in cell_ids order) on one target day, or None if the run isn't (yet) on AWS."""
    keep, owner = grid_cells(tuple(region["bounds"]), region["step"], tuple(cell_ids))
    n = len(cell_ids)
    url = lambda hour: URL.format(day=target - pd.Timedelta(days=1), hour=hour)
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


def storms(region: dict, cell_ids: list[str], days, folder: Path | None = None, log=print) -> pd.DataFrame:
    """SIGNALS per target_date and cell_id for each day HRRR has; with folder, each day is cached there."""
    if folder:
        folder.mkdir(parents=True, exist_ok=True)
    parts = []
    with ThreadPoolExecutor(THREADS) as pool:
        for i, day in enumerate(days):
            path = folder / f"{day.date()}.csv" if folder else None
            if path and path.exists():
                parts.append(pd.read_csv(path, parse_dates=["target_date"]))
                continue
            d = day_storms(region, cell_ids, day, pool)
            if d is None:
                log(f"{day.date()}: no HRRR run on AWS; skipping")
                continue
            d = d.assign(target_date=day, cell_id=cell_ids)
            if path:
                d.to_csv(path, index=False)
            parts.append(d)
            if i % 10 == 9:
                log(f"HRRR: {i + 1} of {len(days)} days")
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["target_date", "cell_id", *SIGNALS])


def standardise(v) -> np.ndarray:
    """A signal across one day's cells: log1p, then mean 0 and std 1 (all zeros if it doesn't vary)."""
    v = np.log1p(np.clip(np.asarray(v, dtype=float), 0, None))
    sd = v.std(ddof=1)
    return (v - v.mean()) / sd if sd > 0 else np.zeros_like(v)
