"""Export the Northern California 0.25-degree model as static JSON for the /norcal/ page.

    python deploy/vercel/export_norcal.py --data private/norcal
    python deploy/vercel/export_norcal.py --data private/norcal --cause lightning --out api-lightning --heat-quantile 0.995

With --cause, the model and table come from <data>/<cause>/ (ml/build_dataset.py --cause) and only fires
of that cause are drawn; the results of ml/cause_experiment.py are included when present.

Reads <data>/model.joblib (ml/train.py --region norcal), model_inputs.csv.gz and cells.json
(ml/build_dataset.py) and places.tsv (ml/fetch_data.py), and writes web/public/api-norcal/:
overview.json in the same shape as the main site's, plus one compact day/YYYY-MM-DD.json per replay day.
With ~300 cells, day files keep per-cell values in arrays (in overview.cells order) instead of one
object per cell; the page expands them.
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / "web/public"
sys.path.insert(0, str(ROOT / "ml"))
from build_dataset import CAUSES  # noqa: E402
from train import HISTORY_START, blend, features, fit_history, load  # noqa: E402

DATATHON_P95 = 0.4082         # 95th percentile of the 0.5-degree model's 2022-2024 scores
BADGES = 10                   # numbered badges on the map (the full check list is longer)
WATCH = 5                     # cells listed beside the map
DETAILED = BADGES             # cells per day that get plain-language reasons
IMPORTANCE_ROWS = 150_000
LABELS = {
    "latitude": "Latitude", "longitude": "Longitude", "season_sin": "Time of year", "season_cos": "Time of year",
    "tmax_lag1": "Yesterday's high", "tmax_mean7": "7-day avg high", "tmax_max7": "7-day peak high",
    "rhmin_lag1": "Yesterday's humidity", "rhmin_mean7": "7-day avg humidity", "rhmin_min7": "7-day lowest humidity",
    "windmax_lag1": "Yesterday's wind", "windmax_max3": "3-day peak wind", "rain_lag1": "Yesterday's rain",
    "rain_sum7": "Rain, last 7 days", "rain_sum14": "Rain, last 14 days", "rain_sum30": "Rain, last 30 days",
    "dry_days": "Dry days in a row", "month": "Month", "weekday": "Day of week", "historical_prior": "Past fire rate",
    "tmax_lag1_anomaly": "Heat vs normal", "tmax_mean7_anomaly": "Week heat vs normal",
    "rhmin_lag1_anomaly": "Humidity vs normal", "windmax_lag1_anomaly": "Wind vs normal",
    "rain_sum7_anomaly": "Week rain vs normal", "rain_sum30_anomaly": "Month rain vs normal",
}


def log(msg: str) -> None:
    print(f"[export] {msg}", flush=True)


def write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, allow_nan=False, separators=(",", ":")), encoding="utf-8")


def r(v, d=4):
    return None if v is None or not np.isfinite(v) else round(float(v), d)


def reasons(row, prior_high: float, lightning: bool = False) -> list[str]:
    """Same rules as app/server.py reasons_for, with the 'often has fires' bar scaled to the finer grid.
    For lightning fires, recent rain usually means thunderstorms are about, so it's named as such."""
    out = []
    if row.tmax_lag1_anomaly >= 1 or row.tmax_mean7_anomaly >= 1:
        out.append("Hotter than normal for this time of year")
    if row.rhmin_lag1_anomaly <= -1:
        out.append("Drier air than normal")
    if row.windmax_lag1_anomaly >= 1:
        out.append("Windier than normal")
    if row.dry_days >= 30:
        out.append("No real rain for 30+ days")
    elif row.dry_days >= 14:
        out.append(f"{int(row.dry_days)} dry days in a row")
    if row.rain_sum30_anomaly <= -0.75 and row.dry_days < 14:
        out.append("Less rain than normal this month")
    if row.historical_prior >= prior_high:
        out.append("Area often has fires this month")
    if row.rain_sum7 >= 10:
        out.append("Showers or storms in the past week" if lightning else "Recent rain is lowering risk")
    return out[:3]


def place_names(cells: list[dict], places: pd.DataFrame) -> dict[str, str]:
    """Each cell's main town: its largest city or town by land area, else its largest CDP, else the nearest place."""
    places = places.assign(name=places.NAME.str.replace(r" (city|town|CDP)$", "", regex=True),
                           incorporated=places.LSAD.astype(str).isin(["25", "43"]))
    names = {}
    for c in cells:
        (s, w), (n, e) = c["bounds"]
        inside = places[places.INTPTLAT.between(s, n, inclusive="left") & places.INTPTLONG.between(w, e, inclusive="left")]
        if len(inside):
            best = inside.sort_values(["incorporated", "ALAND"], ascending=False).iloc[0]
            names[c["id"]] = best["name"]
        else:
            lat, lon = (s + n) / 2, (w + e) / 2
            d = np.hypot(places.INTPTLAT - lat, (places.INTPTLONG - lon) * np.cos(np.radians(lat)))
            names[c["id"]] = f"Near {places.loc[d.idxmin(), 'name']}"
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cause", choices=list(CAUSES), default="all")
    parser.add_argument("--out", default="api-norcal", help="folder under web/public/")
    parser.add_argument("--heat-quantile", type=float, default=0.95,
                        help="score percentile coloured like the original map's top 5%%; rarer fire types need a higher one (lightning: 0.995)")
    args = parser.parse_args()
    OUT = PUBLIC / args.out
    model_dir = args.data if args.cause == "all" else args.data / args.cause

    bundle = joblib.load(model_dir / "model.joblib")
    meta = json.loads((model_dir / "cells.json").read_text())
    data = load(model_dir)
    cols, k = bundle["features"], bundle["review_k"]
    history = bundle["history"]
    cutoff = pd.Timestamp(history["fitted_through"])

    bench = data[data.target_date > cutoff].sort_values(["target_date", "cell_id"]).reset_index(drop=True)
    t0 = time.perf_counter()
    X = features(bench, history, cols)
    bench["score"] = blend(bundle["model"], X, bundle["weather_weight"])
    score_ms = (time.perf_counter() - t0) * 1000
    for c in X.columns:
        if c not in bench:
            bench[c] = X[c].to_numpy()
    bench["rank"] = bench.groupby("target_date").score.rank(method="first", ascending=False).astype(int)
    bench["top"] = bench["rank"] <= k
    y, s = bench.fire_next_day.to_numpy(), bench.score.to_numpy()
    prevalence = float(y.mean())
    # Colour scale for the map: by default the top 5% of cells get the same colours as the top 5% of the
    # original areas. Bursty, rarer fire types match a smaller top share so storm days don't turn the map solid red.
    scale = DATATHON_P95 / float(np.quantile(s, args.heat_quantile))

    order = [c["id"] for c in meta["cells"]]
    index = {cid: i for i, cid in enumerate(order)}
    names = place_names(meta["cells"], pd.read_csv(args.data / "places.tsv", sep="\t", dtype={"LSAD": str}))
    log(f"{len(order)} cells, {len(bench):,} replay rows, top {k} checked per day")

    top = bench.top.to_numpy()
    tp, fp = int((top & (y == 1)).sum()), int((top & (y == 0)).sum())
    fn, tn = int((~top & (y == 1)).sum()), int((~top & (y == 0)).sum())
    precision, recall, _ = precision_recall_curve(y, s)
    idx = np.unique(np.linspace(0, len(precision) - 1, 120).astype(int))
    years = [{"year": int(yr), "fires": int(g.recorded_fires.sum()), "positive_cell_days": int(g.fire_next_day.sum()),
              "ap": r(average_precision_score(g.fire_next_day, g.score)),
              "top3_recall": r((g.top & (g.fire_next_day == 1)).sum() / g.fire_next_day.sum())}
             for yr, g in bench.groupby(bench.target_date.dt.year)]

    sample = X.sample(min(IMPORTANCE_ROWS, len(X)), random_state=42)
    t1 = time.perf_counter()
    perm = permutation_importance(bundle["model"], sample, bench.loc[sample.index, "fire_next_day"],
                                  scoring="average_precision", n_repeats=3, random_state=42, n_jobs=1)
    imp = {}
    for name, val in zip(sample.columns, perm.importances_mean):
        imp[LABELS.get(name, name)] = imp.get(LABELS.get(name, name), 0.0) + float(val)
    log(f"permutation importance on {len(sample):,} rows in {time.perf_counter() - t1:.0f} s")

    day_rows = bench[bench.target_date == bench.target_date.iloc[-1]]
    reps = []
    for _ in range(20):
        t = time.perf_counter()
        blend(bundle["model"], features(day_rows, history, cols), bundle["weather_weight"])
        reps.append((time.perf_counter() - t) * 1000)

    days = bench.assign(hit=bench.top & (bench.fire_next_day == 1)).groupby("target_date").agg(
        fires=("recorded_fires", "sum"), fire_cells=("fire_next_day", "sum"), max_score=("score", "max"), caught=("hit", "sum"))
    totals = bench.groupby("cell_id").recorded_fires.sum()
    fires = pd.read_csv(args.data / "fires.csv")
    if CAUSES[args.cause]:
        fires = fires[fires.nwcg_cause_classification == CAUSES[args.cause]]
    fires["target_date"] = pd.to_datetime(fires.discovery_date, unit="ms").dt.normalize()
    (w, south, e, n), step = meta["bounds"], meta["step"]
    fires["cell_id"] = [f"r{int((la - south) // step)}c{int((lo - w) // step)}" for la, lo in zip(fires.latitude, fires.longitude)]
    fires = fires[(fires.target_date > cutoff) & fires.cell_id.isin(index)]
    fire_points = {d.strftime("%Y-%m-%d"): [[round(a, 4), round(b, 4), index[c]] for a, b, c in zip(g.latitude, g.longitude, g.cell_id)]
                   for d, g in fires.groupby("target_date")}

    model = bundle["model"]
    overview = {
        "region": {"name": meta["name"], "bounds": meta["bounds"], "step": step},
        "cause": args.cause,
        "model": {"name": bundle["selected_name"], "version": bundle["version"], "learner": type(model).__name__,
                  "trees": int(model.n_iter_), "features": len(cols), "weather_weight": bundle["weather_weight"],
                  "trained_through": str(history["fitted_through"])[:10],
                  "size_bytes": (model_dir / "model.joblib").stat().st_size},
        "dates": {"first": bench.target_date.min().strftime("%Y-%m-%d"), "last": bench.target_date.max().strftime("%Y-%m-%d")},
        "cells": [{"id": c["id"], "name": names[c["id"]], "bounds": c["bounds"], "land": c["land"],
                   "fires_total": int(totals.get(c["id"], 0))} for c in meta["cells"]],
        "data": {"fire_records_total": int(data.recorded_fires.sum()), "cells": len(order),
                 "training_rows": int(((data.target_date.dt.year >= HISTORY_START) & (data.target_date <= cutoff)).sum()),
                 "replay_rows": int(len(bench)), "replay_fires": int(bench.recorded_fires.sum()),
                 "first_year": int(data.target_date.dt.year.min()), "last_year": int(data.target_date.dt.year.max())},
        "metrics": {
            "rows": int(len(y)), "positives": int(y.sum()), "prevalence": r(prevalence),
            "average_precision": r(average_precision_score(y, s)), "roc_auc": r(roc_auc_score(y, s)),
            "baseline_average_precision": r(average_precision_score(y, bench.historical_prior)),
            "baseline_review_recall": r(bundle["metrics"]["history_only_top3"]),
            "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
            "accuracy": r((tp + tn) / len(y)), "always_no_accuracy": r(1 - prevalence),
            "review_k": k, "review_caught": tp, "review_recall": r(tp / y.sum()), "review_random": r(k / len(order)),
            "badge_k": BADGES, "watch_k": WATCH, "heat_scale": r(scale),
            "pr_curve": [[r(recall[i]), r(precision[i])] for i in idx], "years": years,
        },
        "importance": sorted(({"feature": f, "importance": round(v, 5)} for f, v in imp.items()), key=lambda d: -d["importance"])[:12],
        "timeline": {"dates": [d.strftime("%Y-%m-%d") for d in days.index], "fires": days.fires.astype(int).tolist(),
                     "fire_cells": days.fire_cells.astype(int).tolist(),
                     "max_score": [round(float(v), 4) for v in days.max_score], "caught": days.caught.astype(int).tolist()},
        "runtime": {"snapshot": True, "machine": platform.machine(), "system": f"{platform.system()} {platform.release()}",
                    "python": platform.python_version(), "sklearn": sklearn.__version__,
                    "score_all_ms": round(score_ms, 1), "score_day_ms": round(float(np.median(reps)), 2)},
    }

    experiment = args.data / "cause_experiment.json"
    if args.cause != "all" and experiment.exists():
        keep = ("type", "score", "fires", "prevalence", "ap", "ap_lift", "top_k_recall", "within_cell_auc")
        overview["experiment"] = [{f: (r(v) if isinstance(v, float) else v) for f, v in row.items() if f in keep}
                                  for row in json.loads(experiment.read_text())["rows"]]

    shutil.rmtree(OUT, ignore_errors=True)
    write(OUT / "overview.json", overview)
    prior_high = 0.3 / scale
    for day, g in bench.groupby("target_date"):
        g = g.set_index("cell_id").reindex(order)
        ranked = g.sort_values("rank")
        date = day.strftime("%Y-%m-%d")
        write(OUT / f"day/{date}.json", {
            "date": date, "issued": (day - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            "score": [r(v) for v in g.score], "fires_by_cell": {index[c]: int(v) for c, v in g.recorded_fires.items() if v},
            "weather": {"tmax": [r(v, 1) for v in g.tmax_lag1], "rh": [r(v, 0) for v in g.rhmin_lag1],
                        "wind": [r(v, 1) for v in g.windmax_lag1], "rain30": [r(v, 1) for v in g.rain_sum30],
                        "dry_days": [int(v) for v in g.dry_days], "past_rate": [r(v) for v in g.historical_prior]},
            "anomaly": {"heat": [r(v, 2) for v in g.tmax_lag1_anomaly], "humidity": [r(v, 2) for v in g.rhmin_lag1_anomaly],
                        "wind": [r(v, 2) for v in g.windmax_lag1_anomaly], "rain30": [r(v, 2) for v in g.rain_sum30_anomaly]},
            "reasons": {index[c]: reasons(row, prior_high, args.cause == "lightning") for c, row in ranked.head(DETAILED).iterrows()},
            "fires": fire_points.get(date, []),
        })
    size = sum(f.stat().st_size for f in OUT.rglob("*.json"))
    log(f"AP {overview['metrics']['average_precision']}, top-{k} recall {overview['metrics']['review_recall']}; "
        f"wrote {len(days)} days to {OUT} ({size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
