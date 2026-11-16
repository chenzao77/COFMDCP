# -*- coding: utf-8 -*-
"""
Phase C (C0) — descriptor calibration against the Mercado training set.

The training-set descriptors were computed by Mercado et al. (void fraction =
helium Widom insertion; surface area = methane-sized probe). To make our
self-computed Zeo++ descriptors for CURATED-COFs comparable, we recompute
descriptors for a SAMPLE of Mercado structures with our own Zeo++ pipeline and
compare against the ground-truth values in properties.csv.

For each Zeo++-derived descriptor we report R²/bias/slope and fit a linear
correction (truth ≈ a·zeo + b). Output: descriptor_calibration.json (+ plot).
If a descriptor already matches 1:1 it gets a≈1,b≈0 (harmless).

Usage:
  python -m transfer.calibrate_descriptors \
      --zeo mercado_zeo --cifs mercado_cifs
"""
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

from transfer.build_curated_features import CALIBRATABLE, build_table

# feature column -> column index in properties.csv (ground truth)
TRUTH_COL = {
    'void_fraction__widom': 3,
    'density__kg/m^3': 5,
    'surface_area__m^2/g': 26,
    'largest_included_sphere_diameter__A': 48,
    'largest_free_sphere_diameter__A': 49,
    'largest_included_sphere_along_free_sphere_path_diameter__A': 50,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zeo', required=True, help='Mercado 样本的 Zeo++ 输出目录')
    ap.add_argument('--cifs', required=True, help='Mercado 样本 CIF 目录')
    ap.add_argument('--props', default='properties.csv')
    ap.add_argument('--out', default='descriptor_calibration.json')
    args = ap.parse_args()

    comp = build_table(args.zeo, args.cifs)          # self-computed, index=name
    if len(comp) == 0:
        raise SystemExit('未解析到任何结构，检查 --zeo/--cifs')
    print(f'自算样本: {len(comp)} 个结构')

    props = pd.read_csv(args.props)
    truth = props.set_index(props.iloc[:, 2].astype(str).str.strip())

    common = [c for c in comp.index if c in truth.index]
    print(f'与 properties.csv 匹配: {len(common)} 个')
    if len(common) < 20:
        print('警告: 匹配过少，校准不可靠（检查 CIF 名是否与 properties.csv 一致）')

    calib, summary = {}, []
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    for ax, feat in zip(axes.ravel(), CALIBRATABLE):
        x = comp.loc[common, feat].astype(float).values            # zeo self-computed
        y = truth.iloc[:, TRUTH_COL[feat]].loc[common].astype(float).values  # truth
        m = ~(np.isnan(x) | np.isnan(y))
        x, y = x[m], y[m]
        a, b = np.polyfit(x, y, 1)
        r2_raw = r2_score(y, x)            # agreement before correction
        r2_fit = r2_score(y, a * x + b)    # after linear correction
        calib[feat] = {'a': float(a), 'b': float(b),
                       'r2_raw': float(r2_raw), 'r2_fit': float(r2_fit)}
        summary.append((feat, r2_raw, a, b, r2_fit))

        ax.scatter(x, y, s=8, alpha=0.4)
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1, label='1:1')
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, a * xs + b, 'r-', lw=1, label=f'fit r²={r2_fit:.3f}')
        ax.set_title(feat.replace('__', ' '), fontsize=8)
        ax.set_xlabel('Zeo++ self-computed'); ax.set_ylabel('Mercado truth')
        ax.legend(fontsize=7)
    fig.suptitle('Descriptor calibration: self-computed Zeo++ vs Mercado truth')
    fig.tight_layout()
    fig.savefig('fig_calibration.png', dpi=150)
    plt.close(fig)

    json.dump(calib, open(args.out, 'w'), indent=2)
    print('\n=== 校准结果 ===')
    print(f'{"descriptor":<52}{"r2_raw":>9}{"a":>9}{"b":>10}{"r2_fit":>9}')
    for feat, r2r, a, b, r2f in summary:
        print(f'{feat:<52}{r2r:>9.3f}{a:>9.3f}{b:>10.3f}{r2f:>9.3f}')
    print(f'\n已保存 {args.out} + fig_calibration.png')
    print('解读: r2_raw 高(≈1:1, a≈1,b≈0) → 探针口径已对齐；'
          'r2_raw 低但 r2_fit 高 → 有系统偏差，线性校正可纠正；'
          'r2_fit 也低 → 该描述符不可靠，需换探针/方法。')


if __name__ == '__main__':
    main()
