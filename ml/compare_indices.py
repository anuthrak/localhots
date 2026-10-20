"""Compare the model with the US fire danger indices on the 2022-2024 test years.

    python ml/compare_indices.py --region norcal --data private/norcal --model private/norcal/model.joblib
    python ml/compare_indices.py --region norcal24 --data private/wildfire-outlook \
        --model private/wildfire-outlook/model/model.joblib --gridmet private/norcal/gridmet

Indices come from gridMET (NFDRS fuel model G): Energy Release Component (ERC) and Burning Index (BI),
averaged over each cell, taken from the day before each target day like every other input. Scores:

  erc, bi              the index as is
  erc_percentile       ERC as a percentile of that cell's own 2012-2021 values (how agencies read it)
  past_fire_rate       the model's history baseline: how often the cell burns in that month
  history_plus_erc     logistic regression on past fire rate and ERC percentile, fitted on 2013-2021
  model                the trained model as deployed
  model_plus_indices   the same model refitted with ERC and BI added as inputs

Two views of skill: average precision and top-k recall reward knowing *where* fires start (ranking cells
against each other), while within_cell_auc rewards knowing *when* (each cell's fire days vs its quiet
days, averaged over cells). The indices were built for the second.

Differences in average precision come with a 90% interval from resampling test days (500 draws), so a
gap smaller than the noise shows up as an interval that crosses zero. Writes <data>/index_comparison.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from build_dataset import gridmet_cells, make_grid
from regions import REGIONS
from train import (CONFIGS, FIT_END, FIT_START, HISTORY_START, blend, expanding_features, features, fit_history, load,
                   make_model, top_k_recall)

INDICES = {"erc": "erc", "bi": "bi"}
DRAWS = 500


def log(msg: str) -> None:
    print(f"[compare] {msg}", flush=True)


def index_inputs(folder: Path, region: dict, cells: list[str]) -> pd.DataFrame:
    """Each index per cell, from the day before each target day (long format: target_date, cell_id, erc, bi)."""
    grid = make_grid(region)
    grid = grid[grid.cell_id.isin(cells)].reset_index(drop=True)
    parts = []
    for name, var in INDICES.items():
        frame, _ = gridmet_cells(folder, var, grid, region)
        frame.index = frame.index + pd.Timedelta(days=1)  # value of day t-1 lands on target day t
        parts.append(frame.stack().rename(name))
    out = pd.concat(parts, axis=1).reset_index()
    return out.rename(columns={"level_0": "target_date", "level_1": "cell_id"})


def percentile_of_own_history(data: pd.DataFrame, col: str) -> pd.Series:
    """Percentile of each value within its cell's 2012-2021 values (fixed reference, no test years)."""
    out = pd.Series(np.nan, index=data.index)
    for cell, g in data.groupby("cell_id"):
        ref = np.sort(g.loc[g.year.between(HISTORY_START, FIT_END), col].dropna().to_numpy())
        out[g.index] = np.searchsorted(ref, g[col].to_numpy(), side="right") / len(ref)
    return out


def within_cell_auc(test: pd.DataFrame, s: np.ndarray) -> float:
    """ROC AUC of each cell's days against each other, averaged over cells with both fire and quiet days."""
    aucs = [roc_auc_score(g.fire_next_day, s[g.index]) for _, g in test.groupby("cell_id") if 0 < g.fire_next_day.sum() < len(g)]
    return float(np.mean(aucs))


def bootstrap_ap_gap(test: pd.DataFrame, a: np.ndarray, b: np.ndarray, rng) -> tuple[float, float]:
    """90% interval of AP(a) - AP(b), resampling whole test days."""
    days = test.target_date.to_numpy()
    uniq, inverse = np.unique(days, return_inverse=True)
    rows_by_day = np.split(np.argsort(inverse, kind="stable"), np.cumsum(np.bincount(inverse))[:-1])
    y = test.fire_next_day.to_numpy()
    gaps = []
    for _ in range(DRAWS):
        idx = np.concatenate([rows_by_day[i] for i in rng.integers(0, len(uniq), len(uniq))])
        gaps.append(average_precision_score(y[idx], a[idx]) - average_precision_score(y[idx], b[idx]))
    return float(np.quantile(gaps, 0.05)), float(np.quantile(gaps, 0.95))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--region", choices=REGIONS, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--gridmet", type=Path, help="folder of gridMET files (default <data>/gridmet)")
    args = parser.parse_args()
    region = REGIONS[args.region]

    data = load(args.data)
    bundle = joblib.load(args.model)
    cols, history, w = bundle["features"], bundle["history"], bundle["weather_weight"]
    cells = sorted(data.cell_id.unique())
    k = bundle.get("review_k", 3)
    cell_codes = history["cell_codes"]

    idx = index_inputs(args.gridmet or args.data / "gridmet", region, cells)
    data = data.merge(idx, on=["target_date", "cell_id"], how="left")
    missing = data.loc[data.year >= FIT_START, list(INDICES)].isna().mean()
    log(f"{len(cells)} cells; index values missing on {missing.max():.2%} of rows from {FIT_START}")
    data["erc_percentile"] = percentile_of_own_history(data, "erc")
    test = data[data.year > FIT_END].reset_index(drop=True)

    # The model as deployed, and its history baseline.
    X_test = features(test, history, cols)
    scores = {"erc": test.erc.to_numpy(), "bi": test.bi.to_numpy(), "erc_percentile": test.erc_percentile.to_numpy(),
              "past_fire_rate": X_test.historical_prior.to_numpy(), "model": blend(bundle["model"], X_test, w)}

    # History + ERC: a two-input logistic regression, trained on 2013-2021 with history built from earlier years only.
    fit_years = range(FIT_START, FIT_END + 1)
    X_fit = expanding_features(data, fit_years, cell_codes, cols)
    fit = data.loc[X_fit.index]
    simple = lambda prior, pct: np.column_stack([np.log(prior), pct])
    ok = fit.erc_percentile.notna().to_numpy()
    lr = LogisticRegression().fit(simple(X_fit.historical_prior.to_numpy()[ok], fit.erc_percentile.to_numpy()[ok]), fit.fire_next_day[ok])
    scores["history_plus_erc"] = lr.predict_proba(simple(scores["past_fire_rate"], np.nan_to_num(test.erc_percentile.to_numpy(), nan=0.5)))[:, 1]

    # The same model refitted with the indices as extra inputs.
    config = bundle["selected_name"].split("__")[0]
    extra = list(INDICES) + ["erc_percentile"]
    model_plus = make_model(CONFIGS[config], cols + extra).fit(pd.concat([X_fit, fit[extra]], axis=1), fit.fire_next_day)
    scores["model_plus_indices"] = blend(model_plus, pd.concat([X_test, test[extra]], axis=1), w)

    y = test.fire_next_day.to_numpy()
    rng = np.random.default_rng(42)
    rows = []
    for name, s in scores.items():
        s = np.nan_to_num(s, nan=np.nanmin(s))
        scores[name] = s
        lo, hi = bootstrap_ap_gap(test, s, scores["past_fire_rate"], rng) if name != "past_fire_rate" else (0.0, 0.0)
        rows.append({"score": name, "ap": average_precision_score(y, s), "roc_auc": roc_auc_score(y, s),
                     "top_k_recall": top_k_recall(test, s, k), "within_cell_auc": within_cell_auc(test, s),
                     "ap_vs_history_90ci": [lo, hi]})
    table = pd.DataFrame(rows)
    log(f"{args.region}, test 2022-2024: {len(test):,} cell-days, fire rate {y.mean():.4f}, crews check the top {k} of {len(cells)}\n"
        + table.assign(ap_vs_history_90ci=table.ap_vs_history_90ci.map(lambda v: f"[{v[0]:+.4f}, {v[1]:+.4f}]")).round(4).to_string(index=False))
    result = {"region": args.region, "cells": len(cells), "review_k": k, "prevalence": float(y.mean()),
              "random_top_k": k / len(cells), "scores": rows,
              "history_plus_erc_coefficients": {"log_past_fire_rate": float(lr.coef_[0][0]), "erc_percentile": float(lr.coef_[0][1])}}
    out = args.data / ("index_comparison.json" if args.region != "norcal24" else "index_comparison_norcal24.json")
    out.write_text(json.dumps(result, indent=2))
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
