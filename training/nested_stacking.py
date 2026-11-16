# -*- coding: utf-8 -*-
"""Strict nested GroupKFold evaluation and deployable stacking model.

Outer folds are used exactly once for reporting.  Optuna tuning, early
stopping, scaling, MLP epoch selection and Ridge meta-training happen only
inside the corresponding outer-training partition.

Formal paper run (expensive):
    python -m training.nested_stacking --outer-folds 5 --inner-folds 4 \
        --trials 30 --final-trials 30 --bootstrap 10000
"""
import argparse
import copy
import json
import os

import joblib
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from core.features import (SEED, build_dataset, load_cv_groups,
                           make_group_folds, regression_metrics)
from core.models import COF_MLP

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _assert_group_disjoint(groups, tr, va, label):
    overlap = set(groups[tr]) & set(groups[va])
    if overlap:
        raise RuntimeError(f'{label} group leakage: {len(overlap)} groups')


def tune_xgb(X, y, groups, n_splits, n_trials, seed):
    folds = make_group_folds(groups, n_splits=n_splits)

    def objective(t):
        params = dict(
            n_estimators=t.suggest_int('n_estimators', 400, 2000, step=100),
            learning_rate=t.suggest_float('learning_rate', .01, .2, log=True),
            max_depth=t.suggest_int('max_depth', 4, 10),
            subsample=t.suggest_float('subsample', .6, 1.0),
            colsample_bytree=t.suggest_float('colsample_bytree', .6, 1.0),
            min_child_weight=t.suggest_int('min_child_weight', 1, 10),
            reg_lambda=t.suggest_float('reg_lambda', 1e-3, 10, log=True),
            reg_alpha=t.suggest_float('reg_alpha', 1e-3, 10, log=True),
            random_state=seed, n_jobs=-1, early_stopping_rounds=50)
        scores = []
        for tr, va in folds:
            _assert_group_disjoint(groups, tr, va, 'XGB inner')
            m = xgb.XGBRegressor(**params)
            m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
            scores.append(_rmse(y[va], m.predict(X[va])))
        return float(np.mean(scores))

    study = optuna.create_study(
        direction='minimize', sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    p = dict(study.best_params)
    p.update(random_state=seed, n_jobs=-1, early_stopping_rounds=50)
    return p, float(study.best_value)


def tune_lgbm(X, y, groups, n_splits, n_trials, seed):
    folds = make_group_folds(groups, n_splits=n_splits)

    def objective(t):
        params = dict(
            n_estimators=t.suggest_int('n_estimators', 400, 2000, step=100),
            learning_rate=t.suggest_float('learning_rate', .01, .2, log=True),
            num_leaves=t.suggest_int('num_leaves', 31, 255),
            max_depth=t.suggest_int('max_depth', 4, 12),
            subsample=t.suggest_float('subsample', .6, 1.0),
            colsample_bytree=t.suggest_float('colsample_bytree', .6, 1.0),
            min_child_samples=t.suggest_int('min_child_samples', 5, 100),
            reg_lambda=t.suggest_float('reg_lambda', 1e-3, 10, log=True),
            reg_alpha=t.suggest_float('reg_alpha', 1e-3, 10, log=True),
            random_state=seed, n_jobs=-1, verbose=-1)
        scores = []
        for tr, va in folds:
            _assert_group_disjoint(groups, tr, va, 'LGBM inner')
            m = LGBMRegressor(**params)
            m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
            scores.append(_rmse(y[va], m.predict(X[va])))
        return float(np.mean(scores))

    study = optuna.create_study(
        direction='minimize', sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    p = dict(study.best_params)
    p.update(random_state=seed, n_jobs=-1, verbose=-1)
    return p, float(study.best_value)


def fit_mlp_with_validation(Xtr, ytr, Xva, yva, seed, max_epochs, patience):
    torch.manual_seed(seed)
    scaler = StandardScaler().fit(Xtr)
    a = scaler.transform(Xtr).astype(np.float32)
    b = scaler.transform(Xva).astype(np.float32)
    tr_loader = DataLoader(TensorDataset(
        torch.tensor(a), torch.tensor(ytr, dtype=torch.float32)[:, None]),
        batch_size=512, shuffle=True)
    va_loader = DataLoader(TensorDataset(
        torch.tensor(b), torch.tensor(yva, dtype=torch.float32)[:, None]),
        batch_size=512, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = COF_MLP(a.shape[1]).to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_loss, best_state, best_epoch, stale = np.inf, None, 1, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        for bx, by in tr_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad(); loss = loss_fn(model(bx), by)
            loss.backward(); optimizer.step()
        model.eval(); total = 0.0
        with torch.no_grad():
            for bx, by in va_loader:
                bx, by = bx.to(device), by.to(device)
                total += loss_fn(model(bx), by).item() * len(bx)
        total /= len(va_loader.dataset)
        if total < best_loss:
            best_loss, best_epoch, stale = total, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state); model.eval()
    pred = []
    with torch.no_grad():
        for bx, _ in va_loader:
            pred.extend(model(bx.to(device)).cpu().numpy().ravel())
    return np.asarray(pred), int(best_epoch)


def fit_mlp_fixed(X, y, epochs, seed):
    torch.manual_seed(seed)
    scaler = StandardScaler().fit(X)
    a = scaler.transform(X).astype(np.float32)
    loader = DataLoader(TensorDataset(
        torch.tensor(a), torch.tensor(y, dtype=torch.float32)[:, None]),
        batch_size=512, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = COF_MLP(a.shape[1]).to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(max(1, int(epochs))):
        model.train()
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad(); loss = loss_fn(model(bx), by)
            loss.backward(); optimizer.step()
    model.eval()
    return model, scaler


def predict_mlp(model, scaler, X):
    device = next(model.parameters()).device
    a = scaler.transform(X).astype(np.float32)
    loader = DataLoader(torch.tensor(a), batch_size=1024, shuffle=False)
    pred = []
    with torch.no_grad():
        for bx in loader:
            pred.extend(model(bx.to(device)).cpu().numpy().ravel())
    return np.asarray(pred)


def inner_meta_features(X, y, groups, xparams, lparams, n_splits,
                        seed, mlp_max_epochs, mlp_patience):
    folds = make_group_folds(groups, n_splits=n_splits)
    meta = np.zeros((len(y), 3), dtype=float)
    xi, li, epochs = [], [], []
    for k, (tr, va) in enumerate(folds):
        _assert_group_disjoint(groups, tr, va, 'meta inner')
        xm = xgb.XGBRegressor(**xparams)
        xm.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
        meta[va, 0] = xm.predict(X[va])
        xi.append((xm.best_iteration or 0) + 1)
        lm = LGBMRegressor(**lparams)
        lm.fit(X[tr], y[tr], eval_set=[(X[va], y[va])],
               callbacks=[lgb.early_stopping(50, verbose=False)])
        meta[va, 1] = lm.predict(X[va])
        li.append(lm.best_iteration_ or lparams['n_estimators'])
        meta[va, 2], ep = fit_mlp_with_validation(
            X[tr], y[tr], X[va], y[va], seed + k,
            mlp_max_epochs, mlp_patience)
        epochs.append(ep)
    return (meta, int(np.median(xi)), int(np.median(li)),
            int(np.median(epochs)))


def fixed_xgb(params, n_estimators):
    p = {k: v for k, v in params.items() if k != 'early_stopping_rounds'}
    p['n_estimators'] = int(n_estimators)
    return xgb.XGBRegressor(**p)


def fixed_lgbm(params, n_estimators):
    p = dict(params); p['n_estimators'] = int(n_estimators)
    return LGBMRegressor(**p)


def grouped_bootstrap(y, stack, lgbm_pred, groups, n_boot, seed):
    """CI for MAE(LGBM)-MAE(Stack); positive means Stack is better."""
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    by_group = {g: np.flatnonzero(groups == g) for g in unique}
    delta = []
    for _ in range(n_boot):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([by_group[g] for g in sampled])
        delta.append(mean_absolute_error(y[idx], lgbm_pred[idx]) -
                     mean_absolute_error(y[idx], stack[idx]))
    q = np.percentile(delta, [2.5, 50, 97.5])
    return {'metric': 'MAE_lgbm_minus_MAE_stack', 'low': float(q[0]),
            'median': float(q[1]), 'high': float(q[2]),
            'stack_significantly_better': bool(q[0] > 0)}


def save_parity(path, y, predictions):
    fig, axes = plt.subplots(1, len(predictions),
                             figsize=(4.5 * len(predictions), 4.2))
    axes = np.atleast_1d(axes)
    for ax, (name, pred) in zip(axes, predictions.items()):
        ax.hexbin(y, pred, gridsize=55, mincnt=1, cmap='viridis')
        lo, hi = float(y.min()), float(y.max())
        ax.plot([lo, hi], [lo, hi], 'r--', lw=1)
        m = regression_metrics(y, pred)
        ax.set_title(f'{name}: R²={m["R2"]:.3f}, MAE={m["MAE"]:.2f}')
        ax.set_xlabel('GCMC true [v STP/v]'); ax.set_ylabel('OOF prediction')
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outer-folds', type=int, default=5)
    ap.add_argument('--inner-folds', type=int, default=4)
    ap.add_argument('--trials', type=int, default=30)
    ap.add_argument('--final-trials', type=int, default=30)
    ap.add_argument('--mlp-max-epochs', type=int, default=150)
    ap.add_argument('--mlp-patience', type=int, default=20)
    ap.add_argument('--bootstrap', type=int, default=10000)
    ap.add_argument('--prefix', default='nested')
    ap.add_argument('--max-groups', type=int,
                    help='仅供烟雾测试；正式结果禁止设置')
    ap.add_argument('--no-final', action='store_true')
    args = ap.parse_args()

    Xdf, y = build_dataset(); groups = load_cv_groups()
    X = Xdf.values.astype(np.float32)
    if args.max_groups:
        rng = np.random.default_rng(SEED)
        chosen = rng.choice(np.unique(groups),
                            min(args.max_groups, len(np.unique(groups))),
                            replace=False)
        keep = np.isin(groups, chosen)
        Xdf, X, y, groups = Xdf.loc[keep].reset_index(drop=True), X[keep], \
            y[keep], groups[keep]
        print(f'烟雾测试子集: {len(y)} rows / {len(np.unique(groups))} groups')

    outer = make_group_folds(groups, n_splits=args.outer_folds)
    preds = {k: np.zeros(len(y)) for k in ['XGBoost', 'LightGBM', 'MLP', 'Stacking']}
    fold_id = np.full(len(y), -1, dtype=int)
    fold_rows, configs = [], []
    for fold, (tr, va) in enumerate(outer):
        _assert_group_disjoint(groups, tr, va, 'outer')
        seed = SEED + fold * 100
        print(f'\n=== Outer {fold + 1}/{args.outer_folds} ===')
        xp, xs = tune_xgb(X[tr], y[tr], groups[tr], args.inner_folds,
                          args.trials, seed)
        lp, ls = tune_lgbm(X[tr], y[tr], groups[tr], args.inner_folds,
                           args.trials, seed)
        meta, nx, nl, ne = inner_meta_features(
            X[tr], y[tr], groups[tr], xp, lp, args.inner_folds, seed,
            args.mlp_max_epochs, args.mlp_patience)
        ridge = Ridge(alpha=1.0).fit(meta, y[tr])
        xm, lm = fixed_xgb(xp, nx), fixed_lgbm(lp, nl)
        xm.fit(X[tr], y[tr], verbose=False); lm.fit(X[tr], y[tr])
        mm, scaler = fit_mlp_fixed(X[tr], y[tr], ne, seed)
        preds['XGBoost'][va] = xm.predict(X[va])
        preds['LightGBM'][va] = lm.predict(X[va])
        preds['MLP'][va] = predict_mlp(mm, scaler, X[va])
        outer_meta = np.column_stack([preds[k][va]
                                      for k in ['XGBoost', 'LightGBM', 'MLP']])
        preds['Stacking'][va] = ridge.predict(outer_meta)
        fold_id[va] = fold
        fm = regression_metrics(y[va], preds['Stacking'][va])
        fold_rows.append({'fold': fold + 1, 'n': len(va),
                          'train_groups': len(np.unique(groups[tr])),
                          'valid_groups': len(np.unique(groups[va])), **fm})
        configs.append({'fold': fold + 1, 'xgb_params': xp,
                        'lgbm_params': lp, 'xgb_inner_rmse': xs,
                        'lgbm_inner_rmse': ls, 'xgb_trees': nx,
                        'lgbm_trees': nl, 'mlp_epochs': ne,
                        'ridge_coef': ridge.coef_.tolist(),
                        'ridge_intercept': float(ridge.intercept_)})
        print('Outer stack:', fm)

    if np.any(fold_id < 0):
        raise RuntimeError('OOF coverage failure')
    table = pd.DataFrame({k: regression_metrics(y, v)
                          for k, v in preds.items()}).T
    table.to_csv(f'{args.prefix}_model_comparison.csv')
    pd.DataFrame(fold_rows).to_csv(f'{args.prefix}_fold_metrics.csv', index=False)
    for name, pred in preds.items():
        stem = {'XGBoost': 'xgb', 'LightGBM': 'lgbm',
                'MLP': 'mlp', 'Stacking': 'stack'}[name]
        pd.DataFrame({'y_true': y, 'y_pred': pred, 'group': groups,
                      'outer_fold': fold_id}).to_csv(
            f'{args.prefix}_{stem}_oof.csv', index=False)
    boot = grouped_bootstrap(y, preds['Stacking'], preds['LightGBM'],
                             groups, args.bootstrap, SEED)
    decision = 'Stacking' if boot['stack_significantly_better'] else 'LightGBM'
    json.dump(_jsonable({'bootstrap': boot, 'paper_primary_model': decision,
                         'outer_configs': configs}),
              open(f'{args.prefix}_evaluation.json', 'w'), indent=2)
    save_parity(f'fig_{args.prefix}_parity.png', y,
                {'LightGBM': preds['LightGBM'], 'Stacking': preds['Stacking']})
    print('\nNested comparison:\n', table.round(4).to_string())
    print('Grouped bootstrap:', boot, '| paper primary:', decision)

    if args.no_final:
        return

    print('\n=== Final deployable fit (all groups; no reported score) ===')
    xp, _ = tune_xgb(X, y, groups, args.inner_folds,
                     args.final_trials, SEED + 9000)
    lp, _ = tune_lgbm(X, y, groups, args.inner_folds,
                      args.final_trials, SEED + 9000)
    meta, nx, nl, ne = inner_meta_features(
        X, y, groups, xp, lp, args.inner_folds, SEED + 9000,
        args.mlp_max_epochs, args.mlp_patience)
    ridge = Ridge(alpha=1.0).fit(meta, y)
    xm, lm = fixed_xgb(xp, nx), fixed_lgbm(lp, nl)
    xm.fit(X, y, verbose=False); lm.fit(X, y)
    mm, scaler = fit_mlp_fixed(X, y, ne, SEED + 9000)
    xm.save_model(f'{args.prefix}_xgb_final.json')
    lm.booster_.save_model(f'{args.prefix}_lgbm_final.txt')
    torch.save({'state_dict': mm.cpu().state_dict(),
                'input_dim': X.shape[1]}, f'{args.prefix}_mlp_final.pt')
    joblib.dump(scaler, f'{args.prefix}_mlp_scaler.joblib')
    pd.Series(Xdf.columns, name='feature').to_csv(
        f'{args.prefix}_feature_columns.csv', index=False)
    model_cfg = {
        'base_order': ['xgb', 'lgbm', 'mlp'],
        'ridge_coef': ridge.coef_.tolist(),
        'ridge_intercept': float(ridge.intercept_),
        'xgb_params': xp, 'lgbm_params': lp,
        'xgb_n_estimators': nx, 'lgbm_n_estimators': nl,
        'mlp_epochs': ne, 'input_dim': X.shape[1],
        'feature_file': f'{args.prefix}_feature_columns.csv',
        'paper_primary_model': decision, 'seed': SEED,
    }
    json.dump(_jsonable(model_cfg), open(f'{args.prefix}_stack_model.json', 'w'),
              indent=2)
    print(f'已保存 {args.prefix}_* 完整部署模型')


if __name__ == '__main__':
    main()
