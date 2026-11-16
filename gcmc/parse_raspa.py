# -*- coding: utf-8 -*-
"""
Phase D — parse RASPA outputs into deliverable capacities.

Scans <runs>/<name>__<P>bar/Output/System_0/*.data, extracts the absolute
loading in cm^3 (STP)/cm^3 framework (= v STP/v, same unit as the ML target),
and computes deliverable capacity = loading(high P) - loading(low P).

Outputs: raspa_loadings.csv (long: name,pressure,loading) and
         raspa_deliverable.csv (name, loading_low, loading_high, deliverable_raspa).

Usage: python -m gcmc.parse_raspa --runs raspa_runs --low 5.8 --high 65
"""
import argparse
import glob
import os
import re
import numpy as np
import pandas as pd

# RASPA prints: "Average loading absolute [cm^3 (STP)/cm^3 framework]  <v> +/- <e>"
PAT = re.compile(
    r'Average loading absolute \[cm\^3 \(STP\)/cm\^3 framework\]\s+'
    r'([-\d.eE]+)\s+\+/-\s+([-\d.eE]+)')


def loading_from_dir(rundir):
    files = glob.glob(os.path.join(rundir, 'Output', 'System_0', '*.data'))
    if not files:
        return None, None, False, None
    txt = open(files[0]).read()
    m = PAT.search(txt)
    finished = 'Simulation finished' in txt
    wm = re.search(r'Simulation finished,\s+(\d+) warnings', txt)
    warnings = int(wm.group(1)) if wm else None
    if not m:
        return None, None, finished, warnings
    return float(m.group(1)), float(m.group(2)), finished, warnings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default='raspa_runs')
    ap.add_argument('--low', type=float, default=5.8)
    ap.add_argument('--high', type=float, default=65.0)
    ap.add_argument('--pred', default='curated_topk.csv',
                    help='可选：并入 ML 预测列对比')
    ap.add_argument('--pred-output-col', default='pred_ml')
    ap.add_argument('--long-out', default='raspa_loadings.csv')
    ap.add_argument('--out', default='raspa_deliverable.csv')
    args = ap.parse_args()

    rows = []
    for d in sorted(glob.glob(os.path.join(args.runs, '*__*bar'))):
        base = os.path.basename(d)
        m = re.match(r'(.+)__([\d.]+)bar$', base)
        if not m:
            continue
        name, p = m.group(1), float(m.group(2))
        load, err, finished, warnings = loading_from_dir(d)
        rows.append(dict(name=name, pressure=p, loading=load,
                         loading_error=err, finished=finished,
                         warnings=warnings))
        if load is None:
            print('无输出/未完成:', base)

    long = pd.DataFrame(rows)
    long.to_csv(args.long_out, index=False)

    wide = long.pivot_table(index='name', columns='pressure', values='loading')
    res = pd.DataFrame(index=wide.index)
    if args.low in wide.columns and args.high in wide.columns:
        res['loading_low'] = wide[args.low]
        res['loading_high'] = wide[args.high]
        res['deliverable_raspa'] = wide[args.high] - wide[args.low]
        ew = long.pivot_table(index='name', columns='pressure',
                              values='loading_error')
        if args.low in ew.columns and args.high in ew.columns:
            res['loading_low_error'] = ew[args.low]
            res['loading_high_error'] = ew[args.high]
            res['deliverable_error'] = np.sqrt(
                ew[args.low] ** 2 + ew[args.high] ** 2)
    if os.path.exists(args.pred):
        pred = pd.read_csv(args.pred, index_col=0)
        if 'pred_deliverable_capacity' in pred.columns:
            res[args.pred_output_col] = pred[
                'pred_deliverable_capacity'].reindex(res.index)
        for col in ['screen_rank', 'in_domain', 'in_domain_frac',
                    'central_domain_frac']:
            if col in pred.columns:
                res[col] = pred[col].reindex(res.index)
    res = res.sort_values('deliverable_raspa', ascending=False) \
        if 'deliverable_raspa' in res.columns else res
    res.to_csv(args.out)
    print(f'已保存 {args.long_out} / {args.out}')
    print(res.round(2).to_string())


if __name__ == '__main__':
    main()
