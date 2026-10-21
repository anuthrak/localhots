"""Why does the lightning model lose its edge on real forecasts? Two experiments on the 2022-2024 test years.

    python ml/forecast_experiment.py --data private/norcal

1. Swap: rebuild the inputs for summer 2024 with Open-Meteo's archived forecasts in place of gridMET for one
   weather variable at a time, for the last 2 days before each target day (as in the nightly runs) and for the
   whole 30-day window (a bigger dose, so a culprit can't hide in the noise). Each version is scored against
   pure gridMET.
2. When vs where: split skill into knowing *when* (does the region's total score pick out the days with
   lightning fires, over the whole year and within fire season?) and *where* (on days with fires, does it rank
   the burning cells high?). Compared with fire history alone and the fire danger index (ERC).

Differences come with 90% intervals from resampling whole days (500 draws). Open-Meteo's archive is downloaded
once to <data>/openmeteo_archive_2024.csv. Writes <data>/lightning/forecast_experiment.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from build_dataset import DRY_CAP, DRY_MM, WEATHER
from compare_indices import index_inputs
from forecast import open_meteo_weather
from regions import REGIONS
from track import ARCHIVE
from train import blend, features, load

SEASON = ("2024-06-01", "2024-09-30")  # swap experiment: the target days with archived forecasts
LAG1 = {"tmax": "tmax_lag1", "rh": "rhmin_lag1", "wind": "windmax_lag1", "rain": "rain_lag1"}
INPUTS = ["tmax_lag1", "tmax_mean7", "tmax_max7", "rhmin_lag1", "rhmin_mean7", "rhmin_min7", "windmax_lag1",
          "windmax_max3", "rain_lag1", "rain_sum7", "rain_sum14", "rain_sum30", "dry_days"]
FIRE_SEASON = [6, 7, 8, 9]
DRAWS = 500


def log(msg: str) -> None:
    print(f"[experiment] {msg}", flush=True)


def daily_gridmet(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Each cell's daily gridMET weather (days x cells), recovered from the lag-1 inputs (weather on target - 1)."""
    day = data.target_date - pd.Timedelta(days=1)
    return {v: data.assign(day=day).pivot(index="day", columns="cell_id", values=c) for v, c in LAG1.items()}


def window_inputs(w: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """weather_features for one target day from the 31 days before it (arrays of days x cells, oldest first)."""
    t, rh, wind, rain = w["tmax"], w["rh"], w["wind"], w["rain"]
    dry = rain < DRY_MM
    # Consecutive dry days ending the day before the target, capped like the training data.
    streak = np.where(dry[::-1].all(axis=0), len(dry), np.argmin(dry[::-1], axis=0))
    return {"tmax_lag1": t[-1], "tmax_mean7": t[-7:].mean(0), "tmax_max7": t[-7:].max(0),
            "rhmin_lag1": rh[-1], "rhmin_mean7": rh[-7:].mean(0), "rhmin_min7": rh[-7:].min(0),
            "windmax_lag1": wind[-1], "windmax_max3": wind[-3:].max(0),
            "rain_lag1": rain[-1], "rain_sum7": rain[-7:].sum(0), "rain_sum14": rain[-14:].sum(0),
            "rain_sum30": rain[-30:].sum(0), "dry_days": np.minimum(streak, DRY_CAP).astype(float)}


def swapped_rows(rows: pd.DataFrame, gm: dict, om: dict, swap: tuple[str, ...], days: int) -> pd.DataFrame:
    """rows with the weather inputs rebuilt, taking the last `days` days of each variable in `swap` from Open-Meteo."""
    cells = list(gm["tmax"].columns)
    out = []
    for target, g in rows.groupby("target_date", sort=True):
        span = pd.date_range(target - pd.Timedelta(days=31), target - pd.Timedelta(days=1))
        w = {}
        for v in WEATHER:
            a = gm[v].loc[span, cells].to_numpy(copy=True)
            if v in swap and days:
                a[-days:] = om[v].loc[span[-days:], cells].to_numpy()
            w[v] = a
        x = pd.DataFrame(window_inputs(w), index=cells)
        out.append(g.drop(columns=INPUTS).join(x, on="cell_id"))
    return pd.concat(out).loc[rows.index]


def archived_forecasts(path: Path, cells: list[dict]) -> dict[str, pd.DataFrame]:
    if not path.exists():
        start = pd.Timestamp(SEASON[0]) - pd.Timedelta(days=35)
        om = open_meteo_weather(cells, ARCHIVE, start_date=str(start.date()), end_date=SEASON[1])
        pd.concat({v: f.stack() for v, f in om.items()}, axis=1).rename_axis(["day", "cell_id"]).to_csv(path)
        log(f"saved {path}")
    flat = pd.read_csv(path, parse_dates=["day"])
    return {v: flat.pivot(index="day", columns="cell_id", values=v) for v in WEATHER}


def day_groups(rows: pd.DataFrame) -> list[np.ndarray]:
    _, inverse = np.unique(rows.target_date.to_numpy(), return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    return np.split(order, np.cumsum(np.bincount(inverse))[:-1])


def top_k(rows: pd.DataFrame, s: np.ndarray, k: int) -> np.ndarray:
    """Per row: is it in its day's top k?"""
    rank = pd.Series(s, index=rows.index).groupby(rows.target_date.to_numpy()).rank(method="first", ascending=False)
    return (rank <= k).to_numpy()


def interval(values: list[float]) -> list[float]:
    return [round(float(np.quantile(values, 0.05)), 4), round(float(np.quantile(values, 0.95)), 4)]


def compare(rows: pd.DataFrame, scores: dict[str, np.ndarray], ref: str, k: int, rng) -> list[dict]:
    """AP and top-k recall for each score, with 90% intervals for the gap to the reference (whole days resampled)."""
    y = rows.fire_next_day.to_numpy()
    groups = day_groups(rows)
    hits = {n: top_k(rows, s, k) & (y == 1) for n, s in scores.items()}
    picks = [rng.integers(0, len(groups), len(groups)) for _ in range(DRAWS)]
    out = []
    for name, s in scores.items():
        ap_gap, rec_gap = [], []
        for p in picks:
            idx = np.concatenate([groups[i] for i in p])
            ap_gap.append(average_precision_score(y[idx], s[idx]) - average_precision_score(y[idx], scores[ref][idx]))
            rec_gap.append((hits[name][idx].sum() - hits[ref][idx].sum()) / y[idx].sum())
        out.append({"score": name, "ap": round(average_precision_score(y, s), 4), "top_k_recall": round(hits[name].sum() / y.sum(), 4),
                    f"ap_vs_{ref}_90ci": interval(ap_gap), f"recall_vs_{ref}_90ci": interval(rec_gap)})
    return out


def swap_experiment(data: pd.DataFrame, kit_cells: list[dict], bundle: dict, archive: Path, rng) -> list[dict]:
    gm = daily_gridmet(data)
    om = archived_forecasts(archive, kit_cells)
    rows = data[data.target_date.between(*SEASON)].reset_index(drop=True)
    check = swapped_rows(rows, gm, om, (), 0)
    worst = float((check[INPUTS] - rows[INPUTS]).abs().max().max())
    if worst > 1e-6:
        raise AssertionError(f"rebuilt gridMET inputs differ from the training table by {worst}")
    log(f"rebuilt inputs match the training table (max difference {worst:.1e})")

    score = lambda r: blend(bundle["model"], features(r, bundle["history"], bundle["features"]), bundle["weather_weight"])
    variants = {"gridmet": score(rows),
                "history_only": features(rows, bundle["history"], bundle["features"]).historical_prior.to_numpy()}
    for days, label in [(2, "last2"), (30, "all30")]:
        variants[f"{label}_all"] = score(swapped_rows(rows, gm, om, tuple(WEATHER), days))
        for v in WEATHER:
            variants[f"{label}_{v}"] = score(swapped_rows(rows, gm, om, (v,), days))
        log(f"scored the {label} swaps")
    table = compare(rows, variants, "gridmet", bundle["review_k"], rng)
    # Does each version still know *when*? Day-level AUC of the region's summed score against lightning fire days.
    groups = rows.target_date.to_numpy()
    fire_day = (rows.fire_next_day.groupby(groups).sum() > 0).astype(int)
    for row in table:
        row["day_auc"] = round(roc_auc_score(fire_day, pd.Series(variants[row["score"]]).groupby(groups).sum()), 4)
    log(f"swap, {SEASON[0]} to {SEASON[1]}: {int(rows.fire_next_day.sum())} lightning fire cell-days on {int(fire_day.sum())} of "
        f"{len(fire_day)} days, top {bundle['review_k']}\n" + pd.DataFrame(table).to_string(index=False))
    return table


def when_where(data: pd.DataFrame, bundle: dict, folder: Path, region: dict, rng) -> dict:
    test = data[data.target_date.dt.year >= 2022].reset_index(drop=True)
    X = features(test, bundle["history"], bundle["features"])
    erc = index_inputs(folder, region, sorted(test.cell_id.unique()))
    test = test.merge(erc[["target_date", "cell_id", "erc"]], on=["target_date", "cell_id"], how="left")
    scores = {"model": blend(bundle["model"], X, bundle["weather_weight"]), "history_only": X.historical_prior.to_numpy(),
              "erc": test.erc.fillna(test.erc.mean()).to_numpy()}

    # When: one number per day for the whole region (summed score, or mean ERC), against whether any cell burned.
    days = test[["target_date"]].assign(fire_cells=test.fire_next_day).groupby("target_date").fire_cells.sum().to_frame()
    for name, s in scores.items():
        days[name] = pd.Series(s).groupby(test.target_date.to_numpy()).sum() if name != "erc" else \
            pd.Series(s).groupby(test.target_date.to_numpy()).mean()
    days["fire_day"] = (days.fire_cells > 0).astype(int)

    def when_scores(d: pd.DataFrame) -> dict:
        out = {}
        for name in scores:
            draws = [roc_auc_score(d.fire_day.iloc[p], d[name].iloc[p]) - roc_auc_score(d.fire_day.iloc[p], d.history_only.iloc[p])
                     for p in (rng.integers(0, len(d), len(d)) for _ in range(DRAWS))]
            out[name] = {"day_auc": round(roc_auc_score(d.fire_day, d[name]), 4), "day_ap": round(average_precision_score(d.fire_day, d[name]), 4),
                         "spearman_with_fire_cells": round(float(spearmanr(d[name], d.fire_cells).statistic), 4),
                         "day_auc_vs_history_90ci": interval(draws)}
        return {"days": len(d), "fire_days": int(d.fire_day.sum()), "fire_day_rate": round(float(d.fire_day.mean()), 4), "scores": out}

    when = {"all_year": when_scores(days), "fire_season": when_scores(days[days.index.month.isin(FIRE_SEASON)])}

    # Where: on days with a lightning fire, how well each score ranks the burning cells within that day.
    where = {}
    fire_days = set(days.index[days.fire_day == 1])
    on = test.target_date.isin(fire_days).to_numpy()
    sub = test[on].reset_index(drop=True)
    for name, s in scores.items():
        s = s[on]
        aucs = [roc_auc_score(g.fire_next_day, s[g.index]) for _, g in sub.groupby("target_date") if 0 < g.fire_next_day.sum() < len(g)]
        where[name] = {"mean_within_day_auc": round(float(np.mean(aucs)), 4),
                       "top_k_recall": round(float((top_k(sub, s, bundle["review_k"]) & (sub.fire_next_day == 1)).sum() / sub.fire_next_day.sum()), 4)}
    result = {"when": when, "where": {"fire_days": len(fire_days), "scores": where}}
    for period, w in when.items():
        log(f"when ({period}: {w['days']} days, {w['fire_days']} with a lightning fire):\n" + pd.DataFrame(w["scores"]).T.to_string())
    log(f"where (within each of {len(fire_days)} fire days):\n" + pd.DataFrame(where).T.to_string())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", type=Path, required=True, help="region data folder; the lightning model is in <data>/lightning/")
    parser.add_argument("--kit", type=Path, default=Path("private/forecast_kit.joblib"), help="forecast kit, for cell centres")
    args = parser.parse_args()
    folder = args.data / "lightning"
    bundle = joblib.load(folder / "model.joblib")
    data = load(folder)
    cells = joblib.load(args.kit)["cells"]
    rng = np.random.default_rng(42)
    result = {"swap": swap_experiment(data, cells, bundle, args.data / "openmeteo_archive_2024.csv", rng),
              "when_where": when_where(data, bundle, args.data / "gridmet", REGIONS["norcal"], rng)}
    (folder / "forecast_experiment.json").write_text(json.dumps(result, indent=2))
    log(f"wrote {folder / 'forecast_experiment.json'}")


if __name__ == "__main__":
    main()
