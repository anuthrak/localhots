"""Do thunderstorm forecasts help the lightning model pick *where* fires start on a lightning day?

    python ml/storm_experiment.py --data private/norcal

ml/forecast_experiment.py found the model knows *when* (which days) far better than *where*: on days with a
lightning fire, it ranks the burning cells little better than fire history. Its inputs say how dry each cell is,
nothing about where storms will be. This takes NOAA's HRRR storm forecasts (ml/hrrr.py: lightning, the issue
day's afternoon lightning since lightning fires are often found a day or more later, CAPE, rain), fits a small
logistic regression on the model's score plus each signal (log1p, then standardised within the day) on the
2022-2023 fire seasons' lightning fire days and tests it on 2024's. Only *where* is scored, so only
fire days are fetched. The model's score uses gridMET (the training weather), so the test isolates the storm
signals from the forecast-rain loss studied in ml/rain_points.py.

Set before running: it counts as a win only if model+storms lifts 2024 top-37 recall with a 90% interval above
zero and raises within-day AUC. Otherwise, stop tuning the model's inputs. (A dry run with random signals moved
recall by up to 1.7 points.)

It also scores the fitted blend on the nightly weather (ml/rain_points.py's five-point rain over Open-Meteo's
archived forecasts) in place of gridMET: what a live forecast would score. HRRR downloads (ml/hrrr.py) are
cached per day under <data>/hrrr_storms/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from forecast_experiment import archived_forecasts, daily_gridmet, day_groups, interval, swapped_rows, top_k
from hrrr import SIGNALS, standardise, storms
from rain_points import fetch_rain
from regions import REGIONS
from train import blend, features, load

SEASON = [6, 7, 8, 9]
FIT, TEST = (2022, 2023), (2024,)
DRAWS = 500


def log(msg: str) -> None:
    print(f"[storms] {msg}", flush=True)


def within_day_z(rows: pd.DataFrame, col: str) -> np.ndarray:
    return rows.groupby("target_date")[col].transform(standardise).to_numpy()


def live_scores(rows: pd.DataFrame, data: pd.DataFrame, bundle: dict, kit: dict, root: Path) -> np.ndarray:
    """The model's score for rows (2024) with the nightly weather: gridMET up to two days before, then Open-Meteo's
    archived forecasts with rain over five points per cell (ml/rain_points.py's downloads)."""
    days = sorted({t - pd.Timedelta(days=d) for t in rows.target_date.unique() for d in (1, 2)})
    multi = fetch_rain(kit["cells"], REGIONS[kit["region"]]["step"], days, root / "openmeteo_rain_points")
    if multi is None:
        raise RuntimeError("five-point rain is incomplete: run ml/rain_points.py first")
    centre = archived_forecasts(root / "openmeteo_archive_2024.csv", kit["cells"])
    om = {**centre, "rain": centre["rain"].copy()}
    om["rain"].loc[multi.index, multi.columns] = multi
    live = swapped_rows(rows, daily_gridmet(data), om, ("tmax", "rh", "wind", "rain"), 2)
    return blend(bundle["model"], features(live, bundle["history"], bundle["features"]), bundle["weather_weight"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", type=Path, required=True, help="region data folder; the lightning model is in <data>/lightning/")
    parser.add_argument("--kit", type=Path, default=Path("private/forecast_kit.joblib"), help="forecast kit, for cell bounds")
    args = parser.parse_args()
    folder = args.data / "lightning"
    bundle, data = joblib.load(folder / "model.joblib"), load(folder)
    meta = json.loads((folder / "cells.json").read_text())
    k = bundle["review_k"]

    test = data[data.year.isin(FIT + TEST) & data.month.isin(SEASON)]
    fire_days = sorted(pd.Timestamp(d) for d in test.loc[test.fire_next_day == 1, "target_date"].unique())
    rows = test[test.target_date.isin(fire_days)].reset_index(drop=True)
    log(f"{len(fire_days)} lightning fire days in the {FIT + TEST} fire seasons")
    hr = storms(REGIONS[meta["region"]], sorted(rows.cell_id.unique()), fire_days, args.data / "hrrr_storms", log)

    rows = rows.merge(hr, on=["target_date", "cell_id"], how="left")
    missing = rows[SIGNALS].isna().any(axis=1)
    if missing.any():
        log(f"no storm forecast for {int(missing.sum())} cell-days on {rows.loc[missing, 'target_date'].nunique()} days; dropping those days")
        rows = rows[~rows.target_date.isin(rows.loc[missing, "target_date"])].reset_index(drop=True)
    base = blend(bundle["model"], features(rows, bundle["history"], bundle["features"]), bundle["weather_weight"])
    logit = lambda v: np.log(np.clip(v, 1e-6, 1 - 1e-6) / (1 - np.clip(v, 1e-6, 1 - 1e-6)))
    X = pd.DataFrame({"model": logit(base), **{s: within_day_z(rows, s) for s in SIGNALS}})
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
        if name == "model+storms":
            # The same fit on the nightly weather instead of gridMET: what a live forecast would score.
            live = live_scores(rows[held].reset_index(drop=True), data, bundle, joblib.load(args.kit), args.data)
            variants["live"] = live
            variants["live+storms"] = lr.decision_function(X.loc[held, cols].assign(model=logit(live)))
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
        gap = {ref: [] for ref in ("model", "live")}
        for pick in picks:
            idx = np.concatenate([groups[i] for i in pick])
            for ref in gap:
                gap[ref].append((hits[name][idx].sum() - hits[ref][idx].sum()) / yt[idx].sum())
        table.append({"score": name, "top_k_recall": round(hits[name].sum() / yt.sum(), 4), "within_day_auc": round(float(np.mean(aucs)), 4),
                      "recall_vs_model_90ci": interval(gap["model"]), "recall_vs_live_90ci": interval(gap["live"])})
    log(f"2024: {int(yt.sum())} lightning fire cell-days on {len(groups)} days, top {k}:\n" + pd.DataFrame(table).to_string(index=False))
    (folder / "storm_experiment.json").write_text(json.dumps({"fit": FIT, "test": TEST, "signals": SIGNALS, "coefficients": coefs,
                                                               "scores": table}, indent=2, default=float))
    log(f"wrote {folder / 'storm_experiment.json'}")


if __name__ == "__main__":
    main()
