# -*- coding: utf-8 -*-
"""
Phase D — generate RASPA GCMC input dirs for selected COFs × pressures.

For each (framework, pressure) it creates  <out>/<name>__<P>bar/  containing:
  simulation.input (framework name, UnitCells, pressure in Pa),
  the framework CIF, and the local force-field def files.
Also writes <out>/run_all.sh to launch every run.

Candidates: top in-domain rows of curated_topk.csv (--n), plus any --names
(e.g. 1–2 Mercado structures for setup validation).

Usage:
  python -m gcmc.gen_raspa \
      --cifs ./curated_cifs --topk curated_transfer_ranked.csv --n 8
  python -m gcmc.gen_raspa \
      --cifs ./mercado_cifs --names linker91_C_linker91_C_lcs-f_relaxed \
      --out raspa_validate
"""
import argparse
import os
import shutil
import numpy as np
import pandas as pd

CUTOFF = 12.8
FF_FILES = ['force_field_mixing_rules.def', 'pseudo_atoms.def', 'methane.def']


def unit_cells(cif, cutoff=CUTOFF):
    from ase.io import read

    cell = read(cif).cell.array
    V = abs(np.linalg.det(cell))
    widths = [V / np.linalg.norm(np.cross(cell[(i + 1) % 3], cell[(i + 2) % 3]))
              for i in range(3)]
    return [max(1, int(np.ceil(2 * cutoff / w))) for w in widths]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cifs', required=True)
    ap.add_argument('--topk', help='curated_topk.csv')
    ap.add_argument('--n', type=int, default=8, help='取域内 Top-n 候选')
    ap.add_argument('--names', nargs='*', default=[], help='额外显式结构名(校验用)')
    ap.add_argument('--pressures', default='5.8,65', help='bar，逗号分隔')
    ap.add_argument('--ffdir', default='raspa_inputs')
    ap.add_argument('--template', default='raspa_inputs/simulation.input.template')
    ap.add_argument('--out', default='raspa_runs')
    args = ap.parse_args()

    names = list(args.names)
    if args.topk and os.path.exists(args.topk):
        t = pd.read_csv(args.topk, index_col=0)
        if 'in_domain' in t.columns:
            t = t[t['in_domain']]
        names += list(t.head(args.n).index)
    names = list(dict.fromkeys(names))          # dedup, keep order
    if not names:
        raise SystemExit('没有候选：给 --topk 或 --names')

    pressures = [float(p) for p in args.pressures.split(',')]
    template = open(args.template).read()
    os.makedirs(args.out, exist_ok=True)

    made, rundirs = 0, []
    for name in names:
        cif = os.path.join(args.cifs, name + '.cif')
        if not os.path.exists(cif):
            print('缺 CIF，跳过:', name); continue
        uc = ' '.join(map(str, unit_cells(cif)))
        for p in pressures:
            d = os.path.join(args.out, f'{name}__{p}bar')
            os.makedirs(d, exist_ok=True)
            inp = (template.replace('__FRAMEWORK__', name)
                           .replace('__UC__', uc)
                           .replace('__PRESSURE_PA__', f'{p * 1e5:.1f}'))
            open(os.path.join(d, 'simulation.input'), 'w').write(inp)
            shutil.copy(cif, d)
            for f in FF_FILES:
                shutil.copy(os.path.join(args.ffdir, f), d)
            rundirs.append(os.path.basename(d))
            made += 1

    # parallel launcher: each RASPA run is single-threaded → run dirs concurrently.
    # Resumable: skips any dir that already has an Output/ (completed run).
    sh = os.path.join(args.out, 'run_all.sh')
    script = [
        '#!/usr/bin/env bash',
        '# 并行运行所有 RASPA 任务。用法: [JOBS=12] [RASPA_BIN=simulate] bash run_all.sh',
        'cd "$(dirname "$0")"',
        ': "${RASPA_BIN:=simulate}"',
        ': "${JOBS:=$(nproc 2>/dev/null || echo 4)}"',
        'export RASPA_BIN',
        'run_one(){ [ -e "$1/Output" ] && { echo "skip $1 (已有结果)"; return; }; '
        'cd "$1" && "$RASPA_BIN" simulation.input > raspa.log 2>&1 '
        '&& echo "done $1" || echo "FAIL $1"; }',
        'export -f run_one',
        "cat <<'DIRS' | xargs -P \"$JOBS\" -I {} bash -c 'run_one \"$@\"' _ {}",
    ]
    script += rundirs
    script += ['DIRS', 'echo "全部完成 (JOBS=$JOBS)"']
    open(sh, 'w').write('\n'.join(script) + '\n')
    print(f'生成 {made} 个运行目录于 {args.out}/（{len(names)} 结构 × {len(pressures)} 压力）')
    print(f'启动(默认吃满核数): bash {sh}    # 或 JOBS=12 bash {sh}')


if __name__ == '__main__':
    main()
