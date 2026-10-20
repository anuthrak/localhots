"""Does weather matter more for some fires than others? Train one model per fire type and compare each with
its own fire-history baseline and with ERC, on the 2022-2024 test years.

    python ml/cause_experiment.py --region norcal --data private/norcal --model private/norcal/model.joblib

Fire types come from FPA FOD: human-caused, lightning (NWCG "Natural"), cause unknown, and large fires
(10+ acres, size class C or bigger, any cause). Each gets its own label (a fire of that type discovered
in the cell that day), its own past fire rate, and its own model with the deployed model's settings. The
question is the weather gain: how much the model beats past fire rate, for where (average precision,
top-k recall) and for when (within-cell AUC). Writes <data>/cause_experiment.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

import compare_indices
from build_dataset import fire_days, make_grid
from compare_indices import bootstrap_ap_gap, index_inputs, within_cell_auc
from regions import REGIONS
from train import CONFIGS, FIT_END, FIT_START, blend, expanding_features, features, fit_history, load, make_model, top_k_recall

TYPES = {
    "human": lambda f: f.nwcg_cause_classification == "Human",
    "lightning": lambda f: f.nwcg_cause_classification == "Natural",
    "unknown": lambda f: ~f.nwcg_cause_classification.isin(["Human", "Natural"]),
    "large": lambda f: f.fire_size_class.isin(list("CDEFG")),
}
compare_indices.DRAWS = 200


def log(msg: str) -> None:
    print(f"[causes] {msg}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--region", choices=REGIONS, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    region = REGIONS[args.region]

    data = load(args.data)
    bundle = joblib.load(args.model)
    cols, w, k = bundle["features"], bundle["weather_weight"], bundle["review_k"]
    config = CONFIGS[bundle["selected_name"].split("__")[0]]
    cells = sorted(data.cell_id.unique())
    cell_codes = {c: i for i, c in enumerate(cells)}
    grid = make_grid(region)
    grid = grid[grid.cell_id.isin(cells)]

    fires = pd.read_csv(args.data / "fires.csv")
    data = data.merge(index_inputs(args.data / "gridmet", region, cells), on=["target_date", "cell_id"], how="left")
    rng = np.random.default_rng(42)
    rows = []
    for name, pick in TYPES.items():
        counts = fire_days(fires[pick(fires)], region, grid).rename(columns={"recorded_fires": "n"})
        d = data.drop(columns="fire_next_day").merge(counts, on=["cell_id", "target_date"], how="left")
        d["fire_next_day"] = (d.n.fillna(0) > 0).astype(int)  # train.py's label column, now for this fire type
        fit_years = range(FIT_START, FIT_END + 1)
        X = expanding_features(d, fit_years, cell_codes, cols)
        model = make_model(config, cols).fit(X, d.loc[X.index, "fire_next_day"])
        history = fit_history(d[d.year <= FIT_END], cell_codes)
        test = d[d.year > FIT_END].reset_index(drop=True)
        X_test = features(test, history, cols)
        y = test.fire_next_day.to_numpy()
        scores = {"erc": np.nan_to_num(test.erc.to_numpy()), "past_fire_rate": X_test.historical_prior.to_numpy(),
                  "model": blend(model, X_test, w)}
        for score, s in scores.items():
            gap = bootstrap_ap_gap(test, s, scores["past_fire_rate"], rng) if score == "model" else None
            rows.append({"type": name, "score": score, "fires": int(test.n.fillna(0).sum()), "prevalence": float(y.mean()),
                         "ap": average_precision_score(y, s), "ap_lift": average_precision_score(y, s) / y.mean(),
                         "top_k_recall": top_k_recall(test, s, k), "within_cell_auc": within_cell_auc(test, s),
                         "ap_gain_vs_history_90ci": gap})
        log(f"{name}: {int(y.sum()):,} test cell-days with a fire ({y.mean():.4f})")

    table = pd.DataFrame(rows)
    show = table.drop(columns="ap_gain_vs_history_90ci").round(4)
    log(f"{args.region}, test 2022-2024, crews check the top {k} of {len(cells)}:\n{show.to_string(index=False)}")
    gains = table.pivot(index="type", columns="score", values=["within_cell_auc", "top_k_recall", "ap_lift"])
    summary = pd.DataFrame({
        "when: model - history (AUC)": gains["within_cell_auc"]["model"] - gains["within_cell_auc"]["past_fire_rate"],
        "when: ERC - history (AUC)": gains["within_cell_auc"]["erc"] - gains["within_cell_auc"]["past_fire_rate"],
        "where: model - history (top-k)": gains["top_k_recall"]["model"] - gains["top_k_recall"]["past_fire_rate"],
        "AP gain 90% interval": table[table.score == "model"].set_index("type").ap_gain_vs_history_90ci.map(lambda v: f"[{v[0]:+.4f}, {v[1]:+.4f}]"),
    })
    log(f"weather gain by fire type:\n{summary.round(4).to_string()}")
    (args.data / "cause_experiment.json").write_text(json.dumps({"region": args.region, "review_k": k, "rows": rows}, indent=2))
    log(f"wrote {args.data / 'cause_experiment.json'}")


if __name__ == "__main__":
    main()
