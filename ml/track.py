"""Keep score of the lightning forecast against the lightning fires actually reported.

    python ml/track.py update --kit forecast_kit.joblib --forecast out/latest.json --out out/
    python ml/track.py backfill --kit forecast_kit.joblib --start 2026-06-01 --end 2026-10-20 --out out/

update adds a forecast's crew list to the record (the published track.json, unless --track names another)
and re-scores every day: fire causes are often filled in days or weeks late, so each night re-checks them all.
backfill reconstructs what the nightly pipeline would have issued for past days, from Open-Meteo's archived
forecasts and gridMET with its usual two-day delay, and marks those days as reconstructed. A season of archived
forecasts is more than Open-Meteo's daily limit, so they're cached (--cache) and a re-run resumes.

Fires: wildfires with cause "Natural" from NIFC's WFIGS incident feed, by discovery date in California time.
The feed leaves most causes "Undetermined" (in 2024 it labelled about 1 in 4 of FPA FOD's lightning fires as
Natural), so the record sees a subset of real lightning fires. Each day compares the model's crew list with
fire history alone (each cell's past rate for the month) and with random picks.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset import WEATHER, cell_of  # noqa: E402
from forecast import TZ, gridmet_weather, open_meteo_weather, score_day  # noqa: E402
from hrrr import storms  # noqa: E402
from regions import REGIONS  # noqa: E402

WFIGS = "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/WFIGS_Incident_Locations/FeatureServer/0/query"
TRACK_URL = "https://tglvkcgk3j17bof7.public.blob.vercel-storage.com/forecast/lightning/track.json"
ARCHIVE = "https://historical-forecast-api.open-meteo.com/v1/forecast"
GRIDMET_DELAY = 2   # days between gridMET's last day and the issue day, as in the nightly runs
PAGE = 2000
SPAN = 14           # days of archived forecasts per download: Open-Meteo counts each two weeks per location as a call
WAIT, TRIES = 600, 24  # after a rate limit: wait 10 minutes, for up to 4 hours (the hourly limit is 5,000 calls)


def log(msg: str) -> None:
    print(f"[track] {msg}", flush=True)


def lightning_fires(region: dict, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Reported lightning wildfires in the region's box: discovery day (California time), latitude, longitude."""
    west, south, east, north = region["bounds"]
    rows, offset = [], 0
    while True:
        q = {"where": f"IncidentTypeCategory = 'WF' AND FireCause = 'Natural' AND FireDiscoveryDateTime >= DATE '{(start - pd.Timedelta(days=1)).date()}'"
                      f" AND FireDiscoveryDateTime < DATE '{(end + pd.Timedelta(days=2)).date()}'",
             "geometry": f"{west},{south},{east},{north}", "geometryType": "esriGeometryEnvelope", "inSR": 4326,
             "spatialRel": "esriSpatialRelIntersects", "outFields": "FireDiscoveryDateTime,InitialLatitude,InitialLongitude",
             "orderByFields": "OBJECTID", "resultOffset": offset, "resultRecordCount": PAGE, "f": "json"}
        page = json.loads(urllib.request.urlopen(f"{WFIGS}?{urllib.parse.urlencode(q)}", timeout=120).read())
        if "error" in page:
            raise RuntimeError(f"WFIGS: {page['error']}")
        rows += [f["attributes"] for f in page["features"]]
        if not page.get("exceededTransferLimit"):
            break
        offset += len(page["features"])
    f = pd.DataFrame(rows, columns=["FireDiscoveryDateTime", "InitialLatitude", "InitialLongitude"]).dropna()
    day = pd.to_datetime(f.FireDiscoveryDateTime, unit="ms", utc=True).dt.tz_convert(TZ).dt.tz_localize(None).dt.normalize()
    f = pd.DataFrame({"day": day, "latitude": f.InitialLatitude, "longitude": f.InitialLongitude})
    return f[f.day.between(start, end)].reset_index(drop=True)


def archived_weather(cells: list[dict], step: float, start: pd.Timestamp, end: pd.Timestamp, folder: Path) -> dict[str, pd.DataFrame] | None:
    """Open-Meteo's archived forecasts (rain over five points per cell), two weeks at a time, cached under folder:
    a season is more than Open-Meteo's daily limit, so a stopped run resumes where it left off. None if it stopped."""
    folder.mkdir(parents=True, exist_ok=True)
    parts = []
    for lo in pd.date_range(start, end, freq=f"{SPAN}D"):
        hi = min(lo + pd.Timedelta(days=SPAN - 1), end)
        path = folder / f"{lo.date()}_{hi.date()}.csv"
        for attempt in range(TRIES):
            if path.exists():
                break
            try:
                om = open_meteo_weather(cells, ARCHIVE, step, start_date=str(lo.date()), end_date=str(hi.date()))
                pd.concat({v: f.stack() for v, f in om.items()}, axis=1).rename_axis(["day", "cell_id"]).to_csv(path)
                log(f"fetched archived forecasts {lo.date()} to {hi.date()}")
            except (urllib.error.HTTPError, OSError) as e:
                if attempt == TRIES - 1:
                    log(f"stopped at {lo.date()}: {e}. Re-run later to resume (Open-Meteo's daily limit resets at midnight UTC).")
                    return None
                log(f"{e}; waiting {WAIT // 60} minutes")
                time.sleep(WAIT)
        parts.append(pd.read_csv(path, parse_dates=["day"]))
    flat = pd.concat(parts)
    ids = [c["id"] for c in cells]
    return {v: flat.pivot(index="day", columns="cell_id", values=v)[ids] for v in WEATHER}


def entry(date: str, issued: str, score, past_rate, k: int, reconstructed: bool) -> dict:
    """A day's crew list (the top k cells, as indexes in the kit's order) and fire history's list for comparison."""
    top = lambda v: pd.Series(v).rank(method="first", ascending=False).sort_values().index[:k].tolist()
    return {"date": date, "issued": issued, "reconstructed": reconstructed, "max_score": round(float(max(score)), 4),
            "top": top(score), "history_top": top(past_rate)}


def load_track(source: str) -> list[dict]:
    try:
        text = urllib.request.urlopen(source, timeout=60).read() if source.startswith("http") else Path(source).read_text()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return []
        raise
    return json.loads(text)["entries"]


def score(kit: dict, entries: list[dict]) -> dict:
    """The record: each day's fires and how many fire cells each crew list caught, plus totals for days that are over."""
    region, cells = REGIONS[kit["region"]], kit["cells"]
    k = kit["bundle"]["review_k"]
    index = {c["id"]: i for i, c in enumerate(cells)}
    entries = sorted(entries, key=lambda e: e["date"])
    today = pd.Timestamp(datetime.now(timezone.utc).astimezone(TZ).date())
    fires = lightning_fires(region, pd.Timestamp(entries[0]["date"]), today)
    r, c = cell_of(fires.latitude, fires.longitude, region)
    fires["cell"] = [index.get(f"r{a}c{b}") for a, b in zip(r, c)]
    fires = fires.dropna(subset=["cell"]).astype({"cell": int})
    days = []
    for e in entries:
        f = fires[fires.day == pd.Timestamp(e["date"])]
        hit = set(f.cell)
        days.append({**e, "over": pd.Timestamp(e["date"]) < today,
                     "fires": [[round(a, 4), round(b, 4), int(i)] for a, b, i in zip(f.latitude, f.longitude, f.cell)],
                     "fire_cells": len(hit), "caught": len(hit & set(e["top"])), "caught_history": len(hit & set(e["history_top"]))})

    def totals(subset):
        done = [d for d in subset if d["over"]]
        n = sum(d["fire_cells"] for d in done)
        return {"days": len(done), "fire_days": sum(d["fire_cells"] > 0 for d in done), "fire_cells": n,
                "caught": sum(d["caught"] for d in done), "caught_history": sum(d["caught_history"] for d in done),
                "recall": round(sum(d["caught"] for d in done) / n, 4) if n else None,
                "recall_history": round(sum(d["caught_history"] for d in done) / n, 4) if n else None}

    return {"updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "cause": kit["cause"],
            "source": "NIFC WFIGS incident locations, FireCause = Natural", "review_k": k, "cells": len(cells),
            "random": round(k / len(cells), 4),
            "summary": {"all": totals(days), "live": totals([d for d in days if not d["reconstructed"]]),
                        "reconstructed": totals([d for d in days if d["reconstructed"]])},
            "entries": [{f: d[f] for f in ("date", "issued", "reconstructed", "max_score", "top", "history_top")} for d in entries],
            "days": [{f: d[f] for f in ("date", "reconstructed", "over", "max_score", "fires", "fire_cells", "caught", "caught_history")}
                     for d in days]}


def write(out: Path, track: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "track.json").write_text(json.dumps(track, allow_nan=False, separators=(",", ":")), encoding="utf-8")
    s = track["summary"]["all"]
    log(f"{s['days']} days scored, {s['fire_cells']} fire cell-days: model caught {s['caught']}, "
        f"fire history {s['caught_history']}, random would catch about {s['fire_cells'] * track['random']:.0f}")


def update(args) -> None:
    kit = joblib.load(args.kit)
    k = kit["bundle"]["review_k"]
    by_date = {e["date"]: e for e in load_track(args.track)}
    for path in args.forecast:
        f = json.loads(Path(path).read_text())
        by_date[f["date"]] = entry(f["date"], f["issued"], f["score"], f["weather"]["past_rate"], k, reconstructed=False)
        log(f"added the live forecast for {f['date']}")
    write(args.out, score(kit, list(by_date.values())))


def backfill(args) -> None:
    kit = joblib.load(args.kit)
    cells, k = kit["cells"], kit["bundle"]["review_k"]
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    region = REGIONS[kit["region"]]
    gm = gridmet_weather(region, cells, start - pd.Timedelta(days=40), end)
    om = archived_weather(cells, region["step"], start - pd.Timedelta(days=5), end, args.cache)
    if om is None:
        return
    log(f"weather: gridMET through {min(f.index.max() for f in gm.values()).date()}, archived forecasts {start.date()}-{end.date()}")
    hr = storms(region, [c["id"] for c in cells], list(pd.date_range(start, end)), args.cache / "hrrr", log)         if kit.get("storm_weights") else None
    by_date = {e["date"]: e for e in load_track(args.track)}
    for target in pd.date_range(start, end):
        issued = target - pd.Timedelta(days=1)
        last = issued - pd.Timedelta(days=GRIDMET_DELAY)  # what gridMET would have reached on the issue day
        weather = {n: pd.concat([gm[n].loc[:last], om[n].loc[last + pd.Timedelta(days=1):issued, gm[n].columns]]) for n in WEATHER}
        day = hr[hr.target_date == target] if hr is not None else None
        rows = score_day(kit, weather, target, day.reset_index(drop=True) if day is not None and len(day) else None)
        date = target.strftime("%Y-%m-%d")
        if date in by_date and not by_date[date]["reconstructed"]:
            continue  # a live forecast for this day beats a reconstruction
        by_date[date] = entry(date, issued.strftime("%Y-%m-%d"), rows.score, rows.historical_prior, k, reconstructed=True)
        if target.day == 1:
            log(f"reconstructed through {date}")
    write(args.out, score(kit, list(by_date.values())))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("update", help="add live forecasts to the record and re-score it")
    p.add_argument("--kit", type=Path, required=True)
    p.add_argument("--forecast", type=Path, nargs="+", required=True, help="forecast files from ml/forecast.py run")
    p.add_argument("--track", default=TRACK_URL, help="existing record (file or URL); a missing one starts fresh")
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=update)
    p = sub.add_parser("backfill", help="reconstruct past days' forecasts")
    p.add_argument("--kit", type=Path, required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--track", default=TRACK_URL, help="existing record to merge into (file or URL)")
    p.add_argument("--cache", type=Path, default=Path("private/openmeteo_track"), help="where archived forecasts (and HRRR, under hrrr/) are kept between runs")
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=backfill)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
