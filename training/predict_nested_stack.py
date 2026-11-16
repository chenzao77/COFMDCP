# -*- coding: utf-8 -*-
"""Load a saved nested stack and make reproducible predictions."""
import argparse
import json

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import xgboost as xgb

from core.features import build_dataset
from core.models import COF_MLP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prefix', default='nested')
    ap.add_argument('--input', help='CSV containing all saved feature columns')
    ap.add_argument('--use-training-dataset', action='store_true',
                    help='rebuild properties.csv features for reproducibility check')
    ap.add_argument('--output', default='nested_predictions.csv')
    args = ap.parse_args()
    if bool(args.input) == bool(args.use_training_dataset):
        raise SystemExit('必须且只能指定 --input 或 --use-training-dataset')

    cfg = json.load(open(f'{args.prefix}_stack_model.json'))
    cols = pd.read_csv(cfg['feature_file'])['feature'].tolist()
    if args.use_training_dataset:
        frame, _ = build_dataset()
        index = frame.index
    else:
        frame = pd.read_csv(args.input, index_col=0)
        index = frame.index
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise SystemExit(f'输入缺少 {len(missing)} 列，前10列: {missing[:10]}')
    X = frame[cols].values.astype(np.float32)

    xm = xgb.XGBRegressor(); xm.load_model(f'{args.prefix}_xgb_final.json')
    lm = lgb.Booster(model_file=f'{args.prefix}_lgbm_final.txt')
    scaler = joblib.load(f'{args.prefix}_mlp_scaler.joblib')
    saved = torch.load(f'{args.prefix}_mlp_final.pt', map_location='cpu')
    mm = COF_MLP(saved['input_dim']); mm.load_state_dict(saved['state_dict'])
    mm.eval()
    scaled = scaler.transform(X).astype(np.float32)
    with torch.no_grad():
        pm = mm(torch.tensor(scaled)).numpy().ravel()
    base = np.column_stack([xm.predict(X), lm.predict(X), pm])
    stack = base @ np.asarray(cfg['ridge_coef']) + cfg['ridge_intercept']
    primary = stack if cfg['paper_primary_model'] == 'Stacking' else base[:, 1]
    out = pd.DataFrame({'pred_xgb': base[:, 0], 'pred_lgbm': base[:, 1],
                        'pred_mlp': base[:, 2], 'pred_stack': stack,
                        'prediction_primary': primary}, index=index)
    out.to_csv(args.output)
    print(f'已加载完整 {args.prefix} 模型并保存 {len(out)} 条预测 -> {args.output}')


if __name__ == '__main__':
    main()
