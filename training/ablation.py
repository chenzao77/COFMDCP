"""Strict nested-Stacking feature ablation.

Geometry, geometry plus categorical descriptors, and the full feature set are
each trained and evaluated independently with the same nested GroupKFold
protocol. The script does not import predictions from the main model run.

Example:

    python -m training.ablation --outer-folds 5 --inner-folds 4 \
        --trials 30 --output-dir results/ablation
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge

from core.features import (SEED, build_dataset, load_cv_groups,
                           make_group_folds, regression_metrics)
from training.nested_stacking import (
    _assert_group_disjoint, _jsonable, fixed_lgbm, fixed_xgb, fit_mlp_fixed,
    inner_meta_features, predict_mlp, tune_lgbm, tune_xgb,
)


METRICS = ["MAE", "RMSE", "R2", "Pearson", "Spearman"]
FEATURE_SET_ORDER = [
    "Geometry only",
    "Geometry + Categorical",
    "Full (+ MACCS)",
]


def fold_metrics(feature_set, y, pred, groups, fold_id):
    rows = []
    for fold in np.unique(fold_id):
        valid = np.flatnonzero(fold_id == fold)
        train = np.flatnonzero(fold_id != fold)
        rows.append({
            "feature_set": feature_set,
            "fold": int(fold) + 1,
            "n": len(valid),
            "train_groups": len(np.unique(groups[train])),
            "valid_groups": len(np.unique(groups[valid])),
            **regression_metrics(y[valid], pred[valid]),
        })
    return rows


def evaluate_nested_stack(X, y, groups, feature_set, args):
    outer = make_group_folds(groups, n_splits=args.outer_folds)
    predictions = {
        name: np.zeros(len(y), dtype=float)
        for name in ["XGBoost", "LightGBM", "MLP", "Stacking"]
    }
    fold_id = np.full(len(y), -1, dtype=int)
    configs = []

    for fold, (train, valid) in enumerate(outer):
        _assert_group_disjoint(groups, train, valid, f"{feature_set} outer")
        seed = SEED + fold * 100
        print(f"\n[{feature_set}] outer fold {fold + 1}/{args.outer_folds}")

        xgb_params, xgb_score = tune_xgb(
            X[train], y[train], groups[train], args.inner_folds,
            args.trials, seed,
        )
        lgbm_params, lgbm_score = tune_lgbm(
            X[train], y[train], groups[train], args.inner_folds,
            args.trials, seed,
        )
        meta, xgb_trees, lgbm_trees, mlp_epochs = inner_meta_features(
            X[train], y[train], groups[train], xgb_params, lgbm_params,
            args.inner_folds, seed, args.mlp_max_epochs, args.mlp_patience,
        )
        ridge = Ridge(alpha=1.0).fit(meta, y[train])

        xgb_model = fixed_xgb(xgb_params, xgb_trees)
        lgbm_model = fixed_lgbm(lgbm_params, lgbm_trees)
        xgb_model.fit(X[train], y[train], verbose=False)
        lgbm_model.fit(X[train], y[train])
        mlp_model, scaler = fit_mlp_fixed(
            X[train], y[train], mlp_epochs, seed,
        )

        predictions["XGBoost"][valid] = xgb_model.predict(X[valid])
        predictions["LightGBM"][valid] = lgbm_model.predict(X[valid])
        predictions["MLP"][valid] = predict_mlp(mlp_model, scaler, X[valid])
        outer_meta = np.column_stack([
            predictions[name][valid]
            for name in ["XGBoost", "LightGBM", "MLP"]
        ])
        predictions["Stacking"][valid] = ridge.predict(outer_meta)
        fold_id[valid] = fold

        configs.append({
            "fold": fold + 1,
            "xgb_params": xgb_params,
            "lgbm_params": lgbm_params,
            "xgb_inner_rmse": xgb_score,
            "lgbm_inner_rmse": lgbm_score,
            "xgb_trees": xgb_trees,
            "lgbm_trees": lgbm_trees,
            "mlp_epochs": mlp_epochs,
            "ridge_coef": ridge.coef_.tolist(),
            "ridge_intercept": float(ridge.intercept_),
        })
        print("  Stacking:", regression_metrics(
            y[valid], predictions["Stacking"][valid],
        ))

        del meta, ridge, xgb_model, lgbm_model, mlp_model, scaler, outer_meta
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if np.any(fold_id < 0):
        raise RuntimeError(f"{feature_set}: incomplete outer OOF coverage")

    stack = predictions["Stacking"]
    return {
        "metrics": regression_metrics(y, stack),
        "stack_oof": stack,
        "fold_id": fold_id,
        "fold_metrics": fold_metrics(
            feature_set, y, stack, groups, fold_id,
        ),
        "outer_configs": configs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--mlp-max-epochs", type=int, default=150)
    parser.add_argument("--mlp-patience", type=int, default=20)
    parser.add_argument("--output-dir", default=".")
    parser.add_argument(
        "--max-groups", type=int,
        help="Small deterministic subset for smoke tests only.",
    )
    args = parser.parse_args()

    Xdf, y, blocks = build_dataset(return_groups=True)
    groups = load_cv_groups()
    if args.max_groups:
        rng = np.random.default_rng(SEED)
        chosen = rng.choice(
            np.unique(groups),
            min(args.max_groups, len(np.unique(groups))),
            replace=False,
        )
        keep = np.isin(groups, chosen)
        Xdf = Xdf.loc[keep].reset_index(drop=True)
        y = y[keep]
        groups = groups[keep]
        print(
            f"Smoke-test subset: {len(y)} rows / "
            f"{len(np.unique(groups))} linker groups"
        )

    feature_sets = {
        "Geometry only": blocks["geometry"],
        "Geometry + Categorical": blocks["geometry"] + blocks["categorical"],
        "Full (+ MACCS)": (
            blocks["geometry"] + blocks["categorical"] + blocks["maccs"]
        ),
    }

    result_rows = {}
    oof_frames = []
    all_fold_rows = []
    metadata = {
        "protocol": "independent strict nested Stacking feature ablation",
        "outer_folds": args.outer_folds,
        "inner_folds": args.inner_folds,
        "optuna_trials_per_base_learner_per_outer_fold": args.trials,
        "mlp_max_epochs": args.mlp_max_epochs,
        "mlp_patience": args.mlp_patience,
        "grouping": "unordered linker pair",
        "n_samples": len(y),
        "n_groups": len(np.unique(groups)),
        "feature_sets": {},
    }

    for feature_set in FEATURE_SET_ORDER:
        columns = feature_sets[feature_set]
        result = evaluate_nested_stack(
            Xdf[columns].to_numpy(np.float32), y, groups, feature_set, args,
        )
        result_rows[feature_set] = {
            **result["metrics"], "n_features": len(columns),
        }
        oof_frames.append(pd.DataFrame({
            "feature_set": feature_set,
            "y_true": y,
            "y_pred": result["stack_oof"],
            "group": groups,
            "outer_fold": result["fold_id"],
        }))
        all_fold_rows.extend(result["fold_metrics"])
        metadata["feature_sets"][feature_set] = {
            "n_features": len(columns),
            "outer_configs": result["outer_configs"],
        }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(result_rows).T.loc[
        FEATURE_SET_ORDER, METRICS + ["n_features"]
    ]
    table["n_features"] = table["n_features"].astype(int)
    table.to_csv(output_dir / "ablation_results.csv")
    pd.concat(oof_frames, ignore_index=True).to_csv(
        output_dir / "ablation_oof.csv", index=False,
    )
    pd.DataFrame(all_fold_rows).to_csv(
        output_dir / "ablation_fold_metrics.csv", index=False,
    )
    with (output_dir / "ablation_evaluation.json").open(
        "w", encoding="utf-8",
    ) as handle:
        json.dump(_jsonable(metadata), handle, indent=2)

    print("\n=== Independent nested-Stacking feature ablation ===")
    print(table.round(4).to_string())
    print(f"\nSaved outputs to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
