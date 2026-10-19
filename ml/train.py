"""Rebuild the localhots model from the cell-day dataset.

Re-implements the training behind the datathon model (`history_120__weather_0.75`), recovered
from the saved bundle and its protocol. With the pinned packages it reproduces the original
scores exactly on 2022-2024.

    python ml/train.py --data <wildfire-outlook dir> [--out model.joblib] [--select] [--compare]

--select   re-run model selection (rolling folds x configs x blend weights). The original fold rules
           aren't recorded exactly; this version ranks history_shallow ~0.001 AP above the original
           pick history_120 (within noise), so the final fit keeps history_120.
--compare  check the rebuilt model against <data>/model/model.joblib on the 2022-2024 test years
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

BASE_FEATURES = ["latitude", "longitude", "season_sin", "season_cos", "tmax_lag1", "tmax_mean7", "tmax_max7",
                 "rhmin_lag1", "rhmin_mean7", "rhmin_min7", "windmax_lag1", "windmax_max3", "rain_lag1",
                 "rain_sum7", "rain_sum14", "rain_sum30", "dry_days"]
ANOMALIES = ["tmax_lag1", "tmax_mean7", "rhmin_lag1", "windmax_lag1", "rain_sum7", "rain_sum30"]
FEATURES = BASE_FEATURES + ["cell_code", "month", "weekday", "historical_prior"] + [n + "_anomaly" for n in ANOMALIES]
CATEGORICAL = ["cell_code", "month", "weekday"]

HISTORY_START, FIT_START, FIT_END = 2012, 2013, 2021  # 2012 is warm-up: history features only
SMOOTHING = 60      # pseudo-days when shrinking fire rates: area-month -> month -> global
MIN_STD = 1.0       # floor for climate std so near-constant inputs (no rain) don't blow up anomalies
ANOMALY_CLIP = 12
REVIEW_K = 3        # areas a crew checks per day

# Selection protocol from the original run: validate on each 3-year window, train on the years before it.
FOLDS = [(2015, 2016, 2017), (2017, 2018, 2019), (2019, 2020, 2021)]
CONFIGS = {
    "history_120": dict(max_iter=120, max_leaf_nodes=15, min_samples_leaf=60, l2_regularization=10, learning_rate=0.07),
    "history_220": dict(max_iter=220, max_leaf_nodes=15, min_samples_leaf=60, l2_regularization=10, learning_rate=0.05),
    "history_wide": dict(max_iter=180, max_leaf_nodes=31, min_samples_leaf=120, l2_regularization=10, learning_rate=0.05),
    "history_shallow": dict(max_iter=300, max_leaf_nodes=7, min_samples_leaf=120, l2_regularization=20, learning_rate=0.04),
    "history_regularized": dict(max_iter=220, max_leaf_nodes=31, min_samples_leaf=60, l2_regularization=20, learning_rate=0.05),
}
WEATHER_WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]
FINAL = ("history_120", 0.75)


def log(msg: str) -> None:
    print(f"[train] {msg}", flush=True)


def load(root: Path) -> pd.DataFrame:
    data = pd.read_csv(root / "datasets/model_inputs.csv.gz", float_precision="round_trip", parse_dates=["target_date"])
    data["year"] = data.target_date.dt.year
    data["month"] = data.target_date.dt.month
    return data


def fit_history(rows: pd.DataFrame, cell_codes: dict[str, int]) -> dict:
    """Per area and month: how often it burns (smoothed) and its normal weather (mean, std)."""
    y = rows.fire_next_day
    global_rate = float(y.mean())
    by_month = y.groupby(rows.month).agg(["sum", "count"])
    month_rate = (by_month["sum"] + SMOOTHING * global_rate) / (by_month["count"] + SMOOTHING)
    rates, climate = {}, {}
    for (cell, month), g in rows.groupby(["cell_id", "month"]):
        key = f"{cell}:{month}"
        rates[key] = float((g.fire_next_day.sum() + SMOOTHING * month_rate[month]) / (len(g) + SMOOTHING))
        climate[key] = {n: [float(g[n].mean()), float(max(g[n].std(ddof=1), MIN_STD))] for n in ANOMALIES}
    return {"baseline": {"rates": rates, "global_rate": global_rate}, "climate": climate,
            "cell_codes": cell_codes, "fitted_through": str(rows.target_date.max().date())}


def features(rows: pd.DataFrame, history: dict) -> pd.DataFrame:
    """The 27 model inputs; same assembly as predict.infer."""
    keys = [f"{c}:{m}" for c, m in zip(rows.cell_id, rows.month)]
    base = history["baseline"]
    x = rows[BASE_FEATURES].copy()
    x["cell_code"] = rows.cell_id.map(history["cell_codes"]).astype(int)
    x["month"] = rows.month.to_numpy()
    x["weekday"] = rows.target_date.dt.dayofweek.to_numpy()
    x["historical_prior"] = [base["rates"].get(k, base["global_rate"]) for k in keys]
    for name in ANOMALIES:
        stats = np.array([history["climate"][k][name] for k in keys])
        x[name + "_anomaly"] = ((rows[name].to_numpy() - stats[:, 0]) / stats[:, 1]).clip(-ANOMALY_CLIP, ANOMALY_CLIP)
    return x[FEATURES]


def expanding_features(data: pd.DataFrame, years, cell_codes: dict[str, int]) -> pd.DataFrame:
    """Inputs for each year built only from the years before it, so training rows never see their own year."""
    parts = []
    for year in years:
        past = data[(data.year >= HISTORY_START) & (data.year < year)]
        parts.append(features(data[data.year == year], fit_history(past, cell_codes)))
    return pd.concat(parts).loc[data.index[data.year.isin(years)]]


def make_model(config: dict) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(**config, categorical_features=CATEGORICAL, early_stopping=False, random_state=42)


def blend(model, X: pd.DataFrame, weather_weight: float) -> np.ndarray:
    weather = model.predict_proba(X)[:, 1] if weather_weight else 0.0
    return (1 - weather_weight) * X.historical_prior.to_numpy() + weather_weight * weather


def top_k_recall(rows: pd.DataFrame, score: np.ndarray, k: int = REVIEW_K) -> float:
    """Share of fire area-days that fall in the day's top k areas."""
    rank = pd.Series(score, index=rows.index).groupby(rows.target_date).rank(method="first", ascending=False)
    fires = rows.fire_next_day == 1
    return float((fires & (rank <= k)).sum() / fires.sum())


def select(data: pd.DataFrame, cell_codes: dict[str, int]) -> pd.DataFrame:
    """Mean validation AP and top-3 recall for every config x weather weight."""
    results = []
    for fold in FOLDS:
        train_years = range(FIT_START, fold[0])
        X = expanding_features(data, train_years, cell_codes)
        y = data.loc[X.index, "fire_next_day"]
        val = data[data.year.isin(fold)]
        Xv = expanding_features(data, fold, cell_codes)
        for name, config in CONFIGS.items():
            model = make_model(config).fit(X, y)
            for w in WEATHER_WEIGHTS:
                s = blend(model, Xv, w)
                results.append({"fold": fold[0], "config": name, "weather_weight": w,
                                "ap": average_precision_score(val.fire_next_day, s), "top3": top_k_recall(val, s)})
        log(f"fold {fold}: trained on {train_years.start}-{train_years.stop - 1}")
    table = pd.DataFrame(results).groupby(["config", "weather_weight"])[["ap", "top3"]].mean().reset_index()
    # Rank: higher AP, then higher top-3 recall, then the simpler (lower) weather weight.
    return table.sort_values(["ap", "top3", "weather_weight"], ascending=[False, False, True]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", type=Path, required=True, help="wildfire-outlook directory (datasets/ and model/)")
    parser.add_argument("--out", type=Path, help="where to save the rebuilt bundle (.joblib)")
    parser.add_argument("--select", action="store_true", help="re-run model selection first")
    parser.add_argument("--compare", action="store_true", help="compare with the original model on 2022-2024")
    args = parser.parse_args()

    data = load(args.data)
    cell_codes = {c: i for i, c in enumerate(sorted(data.cell_id.unique()))}
    config_name, weather_weight = FINAL

    if args.select:
        t0 = time.perf_counter()
        table = select(data, cell_codes)
        log(f"selection took {time.perf_counter() - t0:.1f} s; top 8:\n{table.head(8).round(4).to_string()}")
        best = table.iloc[0]
        log(f"winner: {best.config} at weather weight {best.weather_weight} (original pick: {FINAL[0]} at {FINAL[1]})")

    fit_years = range(FIT_START, FIT_END + 1)
    X = expanding_features(data, fit_years, cell_codes)
    t0 = time.perf_counter()
    model = make_model(CONFIGS[config_name]).fit(X, data.loc[X.index, "fire_next_day"])
    log(f"fitted {config_name} on {len(X):,} rows ({FIT_START}-{FIT_END}) in {time.perf_counter() - t0:.1f} s")

    history = fit_history(data[(data.year >= HISTORY_START) & (data.year <= FIT_END)], cell_codes)
    test = data[data.year > FIT_END]
    score = blend(model, features(test, history), weather_weight)
    metrics = {"ap": average_precision_score(test.fire_next_day, score), "roc_auc": roc_auc_score(test.fire_next_day, score),
               "top3_recall": top_k_recall(test, score), "prevalence": float(test.fire_next_day.mean())}
    log("test 2022-2024: " + json.dumps({k: round(v, 4) for k, v in metrics.items()}))

    if args.compare:
        sys.path.insert(0, str(args.data / "model"))
        from predict import infer  # teammate's runtime
        original = joblib.load(args.data / "model/model.joblib")
        diff = np.abs(infer(original, test.drop(columns=["year", "month"])) - score).max()
        log(f"max |score difference| vs original on {len(test):,} test rows: {diff:.3g}")

    if args.out:
        bundle = {"version": 2, "selected_name": f"{config_name}__weather_{weather_weight}", "model": model,
                  "history": history, "weather_weight": weather_weight, "feature_set": "history_weather",
                  "features": FEATURES, "sklearn": sklearn.__version__}
        joblib.dump(bundle, args.out)
        log(f"saved {args.out}")


if __name__ == "__main__":
    main()
