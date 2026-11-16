# -*- coding: utf-8 -*-
"""Phase C — screen CURATED-COFs with the six-descriptor transfer stack.

The legacy 14-feature ``curated_topk.csv`` is retained for audit.  This script
writes independent ``curated_transfer_ranked.csv`` and
``fig_curated_transfer_screen.png`` artifacts.
"""
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TOPK = 30
plt.rcParams.update({'figure.dpi': 150, 'savefig.bbox': 'tight'})


def predict_geom_stack(Xg):
    cfg = json.load(open('transfer_stack.json'))
    xm = xgb.XGBRegressor(); xm.load_model('transfer_xgb_final.json')
    lm = lgb.Booster(model_file='transfer_lgbm_final.txt')
    px = xm.predict(Xg)
    pl = lm.predict(Xg)
    c = cfg['ridge_coef']; b = cfg['ridge_intercept']
    return c[0] * px + c[1] * pl + b, cfg['transfer_features']


def applicability(feat_df, ranges_csv='transfer_feature_ranges.csv'):
    r = pd.read_csv(ranges_csv).set_index('feature')
    inside = pd.DataFrame(index=feat_df.index)
    central = pd.DataFrame(index=feat_df.index)
    for f in feat_df.columns:
        if f in r.index:
            inside[f] = (feat_df[f] >= r.loc[f, 'train_min']) & \
                        (feat_df[f] <= r.loc[f, 'train_max'])
            if {'train_q005', 'train_q995'}.issubset(r.columns):
                central[f] = (feat_df[f] >= r.loc[f, 'train_q005']) & \
                             (feat_df[f] <= r.loc[f, 'train_q995'])
    return inside.mean(axis=1), central.mean(axis=1)


def main():
    df = pd.read_csv('curated_features.csv', index_col=0)
    cfg = json.load(open('transfer_stack.json'))
    geom = cfg['transfer_features']

    missing = [c for c in geom if c not in df.columns]
    if missing:
        raise SystemExit(f'curated_features.csv 缺列: {missing}')

    feat = df[geom].copy()
    n0 = len(feat)
    feat = feat.dropna()
    if len(feat) < n0:
        print(f'丢弃含缺失值的 {n0 - len(feat)} 个 COF（Zeo++ 失败）')

    pred, _ = predict_geom_stack(feat.values.astype(np.float32))
    frac_in, central_frac = applicability(feat)

    out = pd.DataFrame({
        'pred_deliverable_capacity': pred,
        'in_domain_frac': frac_in.values,
        # Six features: >=0.999 means every descriptor is within train min/max.
        'in_domain': (frac_in.values >= 0.999),
        'central_domain_frac': central_frac.values,
    }, index=feat.index)
    out = pd.concat([out, feat], axis=1).sort_values(
        'pred_deliverable_capacity', ascending=False)
    out.insert(0, 'screen_rank', np.arange(1, len(out) + 1))
    out.to_csv('curated_transfer_ranked.csv')

    print(f'CURATED 预测完成: {len(out)} 个 COF')
    print(f'严格域内(6/6 特征在训练范围)占比: {out["in_domain"].mean():.1%}')
    print('\n=== Top 15（仅域内）===')
    print(out[out['in_domain']].head(15)[
        ['pred_deliverable_capacity', 'in_domain_frac',
         'void_fraction__widom', 'density__kg/m^3', 'surface_area__m^2/g']
    ].to_string())

    top = out[out['in_domain']].head(TOPK)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].hist(out['pred_deliverable_capacity'], bins=40, color='steelblue')
    ax[0].set_xlabel('Predicted deliverable capacity [v STP/v]')
    ax[0].set_ylabel('Count'); ax[0].set_title('CURATED-COFs prediction distribution')
    ax[1].barh(range(len(top))[::-1], top['pred_deliverable_capacity'],
               color='#268')
    ax[1].set_yticks(range(len(top))[::-1])
    ax[1].set_yticklabels(top.index, fontsize=6)
    ax[1].set_xlabel('Predicted [v STP/v]')
    ax[1].set_title(f'Top {TOPK} in-domain real COFs')
    fig.tight_layout(); fig.savefig('fig_curated_transfer_screen.png'); plt.close(fig)
    print('已保存 curated_transfer_ranked.csv / fig_curated_transfer_screen.png')


if __name__ == '__main__':
    main()
