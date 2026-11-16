# -*- coding: utf-8 -*-
"""Phase C — leakage-safe cross-database transfer model.

Only six intensive pore descriptors are transferable across CIF databases.
Raw atom counts and ``dimensions`` are deliberately excluded because they
depend on cell representation or unavailable metadata.

The reported OOF score uses a strict nested stack: each outer validation fold
is untouched while four inner group folds create the Ridge meta-training
features.  Legacy ``geom_*`` artifacts are never overwritten.
"""
import argparse
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge

from core.features import (
    SEED, TRANSFER_FEATURES, build_dataset, load_cv_groups, make_group_folds,
    regression_metrics,
)

XGB_PARAMS = dict(n_estimators=1000, learning_rate=0.05, max_depth=6,
                  subsample=0.8, colsample_bytree=0.8,
                  random_state=SEED, n_jobs=-1, early_stopping_rounds=50)
LGBM_PARAMS = dict(n_estimators=1000, learning_rate=0.05, max_depth=6,
                   num_leaves=63, subsample=0.8, colsample_bytree=0.8,
                   random_state=SEED, n_jobs=-1, verbose=-1)


def _fixed_xgb(n_estimators):
    p = {k: v for k, v in XGB_PARAMS.items()
         if k != 'early_stopping_rounds'}
    p['n_estimators'] = int(n_estimators)
    return xgb.XGBRegressor(**p)


def _fixed_lgbm(n_estimators):
    p = dict(LGBM_PARAMS)
    p['n_estimators'] = int(n_estimators)
    return LGBMRegressor(**p)


def _inner_meta(X, y, groups, n_splits):
    """Inner OOF base predictions and robust iteration counts."""
    meta = np.zeros((len(y), 2), dtype=float)
    x_iters, l_iters = [], []
    for tr, va in make_group_folds(groups, n_splits=n_splits):
        xm = xgb.XGBRegressor(**XGB_PARAMS)
        xm.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
        meta[va, 0] = xm.predict(X[va])
        x_iters.append((xm.best_iteration or 0) + 1)

        lm = LGBMRegressor(**LGBM_PARAMS)
        lm.fit(X[tr], y[tr], eval_set=[(X[va], y[va])],
               callbacks=[lgb.early_stopping(50, verbose=False)])
        meta[va, 1] = lm.predict(X[va])
        l_iters.append(lm.best_iteration_ or LGBM_PARAMS['n_estimators'])
    return meta, int(np.median(x_iters)), int(np.median(l_iters))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outer-folds', type=int, default=5)
    ap.add_argument('--inner-folds', type=int, default=4)
    args = ap.parse_args()

    X, Y = build_dataset()
    missing = [f for f in TRANSFER_FEATURES if f not in X.columns]
    if missing:
        raise SystemExit(f'缺少迁移特征: {missing}')
    Xg = X[TRANSFER_FEATURES]
    Xa = Xg.values
    groups = load_cv_groups()
    outer = make_group_folds(groups, n_splits=args.outer_folds)
    print(f'跨库不变量 ({len(TRANSFER_FEATURES)}):', TRANSFER_FEATURES)

    ox = np.zeros(len(Y)); ol = np.zeros(len(Y)); outer_stack = np.zeros(len(Y))
    fold_id = np.full(len(Y), -1, dtype=int)
    fold_rows, outer_x_iters, outer_l_iters = [], [], []
    for fold, (tr, va) in enumerate(outer):
        if set(groups[tr]) & set(groups[va]):
            raise RuntimeError('外层分组泄漏')
        inner_meta, nx, nl = _inner_meta(
            Xa[tr], Y[tr], groups[tr], args.inner_folds)
        ridge = Ridge(alpha=1.0).fit(inner_meta, Y[tr])
        xm, lm = _fixed_xgb(nx), _fixed_lgbm(nl)
        xm.fit(Xa[tr], Y[tr], verbose=False)
        lm.fit(Xa[tr], Y[tr])
        ox[va], ol[va] = xm.predict(Xa[va]), lm.predict(Xa[va])
        outer_stack[va] = ridge.predict(np.column_stack([ox[va], ol[va]]))
        fold_id[va] = fold
        outer_x_iters.append(nx); outer_l_iters.append(nl)
        m = regression_metrics(Y[va], outer_stack[va])
        fold_rows.append({'fold': fold + 1, 'n': len(va),
                          'n_train_groups': len(np.unique(groups[tr])),
                          'n_valid_groups': len(np.unique(groups[va])), **m})
        print(f'Outer fold {fold + 1}: XGB={nx} trees, LGBM={nl} trees, {m}')

    if np.any(fold_id < 0):
        raise RuntimeError('并非每个样本都恰好获得一次外层 OOF 预测')
    print('transfer XGB ', regression_metrics(Y, ox))
    print('transfer LGBM', regression_metrics(Y, ol))
    print('transfer STACK', regression_metrics(Y, outer_stack))
    # The final deployable Ridge is fitted on cross-fitted base predictions.
    final_ridge = Ridge(alpha=1.0).fit(np.column_stack([ox, ol]), Y)

    # final refit on all data + final Ridge; use median inner-selected tree counts
    nx, nl = int(np.median(outer_x_iters)), int(np.median(outer_l_iters))
    fx = _fixed_xgb(nx)
    fx.fit(Xa, Y, verbose=False)
    fx.save_model('transfer_xgb_final.json')
    fl = _fixed_lgbm(nl)
    fl.fit(Xa, Y)
    fl.booster_.save_model('transfer_lgbm_final.txt')

    json.dump({
        'base': ['xgb', 'lgbm'],
        'ridge_coef': list(map(float, final_ridge.coef_)),
        'ridge_intercept': float(final_ridge.intercept_),
        'transfer_features': TRANSFER_FEATURES,
        'xgb_n_estimators': nx,
        'lgbm_n_estimators': nl,
        'outer_folds': args.outer_folds,
        'inner_folds': args.inner_folds,
        'seed': SEED,
    }, open('transfer_stack.json', 'w'), indent=2)

    # applicability-domain ranges
    pd.DataFrame({'feature': TRANSFER_FEATURES,
                  'train_min': Xg.min().values,
                  'train_max': Xg.max().values,
                  'train_q005': Xg.quantile(0.005).values,
                  'train_q995': Xg.quantile(0.995).values}).to_csv(
        'transfer_feature_ranges.csv', index=False)
    pd.DataFrame({'y_true': Y, 'xgb_pred': ox, 'lgbm_pred': ol,
                  'outer_fold': fold_id, 'group': groups}).to_csv(
        'transfer_base_oof.csv', index=False)
    pd.DataFrame({'y_true': Y, 'y_pred': outer_stack,
                  'outer_fold': fold_id, 'group': groups}).to_csv(
        'transfer_stack_oof.csv', index=False)
    pd.DataFrame(fold_rows).to_csv('transfer_fold_metrics.csv', index=False)
    print('已保存 transfer_* 模型、OOF、逐折指标与适用域；legacy geom_* 未覆盖')


if __name__ == '__main__':
    main()
