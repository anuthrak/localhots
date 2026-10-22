"""Does averaging Open-Meteo rain over several points per cell win back the lightning model's edge on forecasts?

    python ml/rain_points.py --data private/norcal

ml/forecast_experiment.py found rain is what costs the model its edge on forecasts: storm rain is patchy, gridMET
averages it over each cell, and Open-Meteo at the cell's centre mostly misses it. This fetches Open-Meteo's
archived forecasts at each cell's centre and the centres of its four quarters, averages the rain, and rescores
summer 2024's lightning fire days with the nightly setup (gridMET up to two days before, then forecasts).

Only fire days are scored (top-k recall and within-day AUC need nothing else), so only the two days before each
are fetched. Downloads are cached per batch under <data>/openmeteo_rain_points/ and resume after a dropped
connection or Open-Meteo's daily limit.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from forecast import BATCH, OFFSETS, open_meteo_weather, points
from forecast_experiment import SEASON, archived_forecasts, daily_gridmet, interval, swapped_rows, top_k
from regions import REGIONS
from track import ARCHIVE
from train import blend, features, load

DRAWS = 500
WAIT, TRIES = 600, 24  # after a rate limit: wait 10 minutes, for up to 4 hours (the hourly limit is 5,000 calls)


def log(msg: str) -> None:
    print(f"[rain] {msg}", flush=True)


def runs(days: list[pd.Timestamp]) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Group days into spans of at most 14 days: Open-Meteo counts each span per location as one call."""
    out = []
    for d in sorted(days):
        if out and (d - out[-1][0]).days < 14:
            out[-1][1] = d
        else:
            out.append([d, d])
    return [tuple(r) for r in out]


def fetch_rain(cells: list[dict], step: float, days: list[pd.Timestamp], folder: Path) -> pd.DataFrame | None:
    """Mean rain over each cell's sample points (days x cells), or None if a download limit stopped it early."""
    folder.mkdir(parents=True, exist_ok=True)
    pts = points(cells, step)
    for start, end in runs(days):
        for i in range(0, len(pts), BATCH):
            path = folder / f"{start.date()}_{end.date()}_{i:05d}.csv"
            if path.exists():
                continue
            for attempt in range(TRIES):
                try:
                    rain = open_meteo_weather(pts[i:i + BATCH], ARCHIVE, start_date=str(start.date()), end_date=str(end.date()))["rain"]
                    break
                except (urllib.error.HTTPError, OSError) as e:
                    if attempt == TRIES - 1:
                        log(f"stopped at {path.name}: {e}. Re-run later to resume (Open-Meteo's daily limit resets at midnight UTC).")
                        return None
                    log(f"{e}; waiting {WAIT // 60} minutes")
                    time.sleep(WAIT)
            rain.rename_axis("day").to_csv(path)
        log(f"fetched {start.date()} to {end.date()}")
    rain = pd.concat([pd.read_csv(p, index_col="day", parse_dates=True) for p in sorted(folder.glob("*.csv"))], axis=1)
    rain = rain.T.groupby(level=0).first().T  # spans can share days
    return rain.T.groupby(lambda pid: pid.split("#")[0]).mean().T


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", type=Path, required=True, help="region data folder; the lightning model is in <data>/lightning/")
    parser.add_argument("--kit", type=Path, default=Path("private/forecast_kit.joblib"), help="forecast kit, for cell bounds")
    args = parser.parse_args()
    folder = args.data / "lightning"
    bundle, data = joblib.load(folder / "model.joblib"), load(folder)
    kit = joblib.load(args.kit)
    k = bundle["review_k"]

    season = data[data.target_date.between(*SEASON)]
    fire_days = sorted(season.loc[season.fire_next_day == 1, "target_date"].unique())
    rows = season[season.target_date.isin(fire_days)].reset_index(drop=True)
    need = sorted({pd.Timestamp(t) - pd.Timedelta(days=d) for t in fire_days for d in (1, 2)})
    log(f"{len(fire_days)} lightning fire days in {SEASON[0]} to {SEASON[1]}: {len(need)} days of rain at {len(OFFSETS)} points "
        f"x {len(kit['cells'])} cells, {len(runs(need))} spans")
    multi = fetch_rain(kit["cells"], REGIONS[kit["region"]]["step"], need, args.data / "openmeteo_rain_points")
    if multi is None:
        return

    gm = daily_gridmet(data)
    centre = archived_forecasts(args.data / "openmeteo_archive_2024.csv", kit["cells"])
    averaged = {**centre, "rain": centre["rain"].copy()}
    averaged["rain"].loc[multi.index, multi.columns] = multi
    score = lambda r: blend(bundle["model"], features(r, bundle["history"], bundle["features"]), bundle["weather_weight"])
    variants = {"gridmet": score(rows),
                "history_only": features(rows, bundle["history"], bundle["features"]).historical_prior.to_numpy(),
                "live_centre": score(swapped_rows(rows, gm, centre, ("tmax", "rh", "wind", "rain"), 2)),
                "live_rain_5pt": score(swapped_rows(rows, gm, averaged, ("tmax", "rh", "wind", "rain"), 2)),
                "rain_only_centre": score(swapped_rows(rows, gm, centre, ("rain",), 2)),
                "rain_only_5pt": score(swapped_rows(rows, gm, averaged, ("rain",), 2))}

    # Rain itself: how often each source sees the rain gridMET recorded on these days.
    g = gm["rain"].loc[need, multi.columns]
    rain_check = {}
    for name, frame in [("centre", centre["rain"]), ("5pt", multi)]:
        o = frame.loc[need, multi.columns]
        wet = g >= 1
        rain_check[name] = {"wet_share_of_gridmet_wet": round(float(((o >= 1) & wet).to_numpy().sum() / wet.to_numpy().sum()), 4),
                            "total_vs_gridmet": round(float(o.to_numpy().sum() / g.to_numpy().sum()), 4),
                            "correlation": round(float(np.corrcoef(o.to_numpy().ravel(), g.to_numpy().ravel())[0, 1]), 4)}
    log("rain on the days before fire days, against gridMET:\n" + pd.DataFrame(rain_check).T.to_string())

    y = rows.fire_next_day.to_numpy()
    groups = [g.index.to_numpy() for _, g in rows.groupby("target_date")]
    rng = np.random.default_rng(42)
    picks = [rng.integers(0, len(groups), len(groups)) for _ in range(DRAWS)]
    hits = {n: top_k(rows, s, k) & (y == 1) for n, s in variants.items()}
    table = []
    for name, s in variants.items():
        aucs = np.array([roc_auc_score(y[i], s[i]) for i in groups])
        gap = [(hits[name][np.concatenate([groups[i] for i in p])].sum() - hits["live_centre"][np.concatenate([groups[i] for i in p])].sum())
               / y[np.concatenate([groups[i] for i in p])].sum() for p in picks]
        table.append({"score": name, "top_k_recall": round(hits[name].sum() / y.sum(), 4), "within_day_auc": round(float(aucs.mean()), 4),
                      "recall_vs_live_centre_90ci": interval(gap)})
    log(f"{int(y.sum())} lightning fire cell-days on {len(fire_days)} days, top {k}:\n" + pd.DataFrame(table).to_string(index=False))
    (folder / "rain_points.json").write_text(json.dumps({"season": SEASON, "offsets": OFFSETS, "rain": rain_check, "scores": table}, indent=2))
    log(f"wrote {folder / 'rain_points.json'}")


if __name__ == "__main__":
    main()
