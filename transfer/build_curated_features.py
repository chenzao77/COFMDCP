# -*- coding: utf-8 -*-
"""
Phase C — assemble CURATED-COFs feature table aligned to the training-set
geometry descriptors (14 columns), from Zeo++ outputs + CIF parsing.

Inputs:
  --zeo  dir with <name>.res / <name>.sa / <name>.vol  (from run_zeopp.sh)
  --cifs dir with <name>.cif                            (for atom counts)
Output: curated_features.csv  (index = COF name; 14 columns matching
        features.build_dataset geometry group names).

Field mapping (Zeo++ -> training):
  .res   -> largest_included_sphere_diameter__A (Di),
            largest_free_sphere_diameter__A (Df),
            largest_included_sphere_along_free_sphere_path_diameter__A (Dif)
  .sa    -> surface_area__m^2/g (ASA_m^2/g), density__kg/m^3 (Density g/cm^3 *1000)
  .vol   -> void_fraction__widom (AV_Volume_fraction, He probe)
  CIF    -> num_carbon..num_silicon, density cross-check
  dimensions: default 3 (low SHAP importance); override via --dims-csv if known.
"""
import argparse
import glob
import os
import re
import numpy as np
import pandas as pd

ELEMENTS = {'num_carbon': 'C', 'num_fluorine': 'F', 'num_hydrogen': 'H',
            'num_nitrogen': 'N', 'num_oxygen': 'O', 'num_sulfur': 'S',
            'num_silicon': 'Si'}
GEOM_COLS = (['dimensions', 'void_fraction__widom', 'density__kg/m^3',
              'surface_area__m^2/g'] + list(ELEMENTS) +
             ['largest_included_sphere_diameter__A',
              'largest_free_sphere_diameter__A',
              'largest_included_sphere_along_free_sphere_path_diameter__A'])


def parse_res(path):
    # "<name>   Di  Df  Dif"
    toks = open(path).read().split()
    nums = [float(t) for t in toks if re.fullmatch(r'-?\d+(\.\d+)?', t)]
    return nums[:3] if len(nums) >= 3 else [np.nan] * 3


def _grab(text, key):
    m = re.search(rf'{re.escape(key)}[:\s]+([-\d.eE]+)', text)
    return float(m.group(1)) if m else np.nan


def parse_sa(path):
    t = open(path).read()
    asa = _grab(t, 'ASA_m^2/g')
    dens = _grab(t, 'Density')          # g/cm^3
    return asa, dens


def parse_vol(path):
    t = open(path).read()
    vf = _grab(t, 'AV_Volume_fraction')
    if np.isnan(vf):
        vf = _grab(t, 'POAV_Volume_fraction')
    return vf


def atom_counts(cif):
    from ase.io import read
    atoms = read(cif)
    syms = atoms.get_chemical_symbols()
    return {col: syms.count(el) for col, el in ELEMENTS.items()}


# Zeo++-derived continuous descriptors eligible for calibration (atom counts /
# dimensions are exact from the CIF and are NOT calibrated).
CALIBRATABLE = ['void_fraction__widom', 'density__kg/m^3', 'surface_area__m^2/g',
                'largest_included_sphere_diameter__A',
                'largest_free_sphere_diameter__A',
                'largest_included_sphere_along_free_sphere_path_diameter__A']


def build_table(zeo_dir, cifs_dir, dims=None, default_dim=3, verbose=True):
    """Assemble the 14-col geometry feature table (index=COF name) from Zeo++
    outputs + CIF parsing. Reused by calibrate_descriptors.py."""
    dims = dims or {}
    rows = {}
    for res in sorted(glob.glob(os.path.join(zeo_dir, '*.res'))):
        name = os.path.basename(res)[:-4]
        sa = os.path.join(zeo_dir, name + '.sa')
        vol = os.path.join(zeo_dir, name + '.vol')
        cif = os.path.join(cifs_dir, name + '.cif')
        if not (os.path.exists(sa) and os.path.exists(vol) and os.path.exists(cif)):
            if verbose:
                print('跳过(缺文件):', name)
            continue
        di, df, dif = parse_res(res)
        asa, dens = parse_sa(sa)
        vf = parse_vol(vol)
        try:
            counts = atom_counts(cif)
        except Exception as e:
            if verbose:
                print('CIF 解析失败', name, e)
            continue
        row = {
            'dimensions': dims.get(name, default_dim),
            'void_fraction__widom': vf,
            'density__kg/m^3': dens * 1000.0 if not np.isnan(dens) else np.nan,
            'surface_area__m^2/g': asa,
            'largest_included_sphere_diameter__A': di,
            'largest_free_sphere_diameter__A': df,
            'largest_included_sphere_along_free_sphere_path_diameter__A': dif,
        }
        row.update(counts)
        rows[name] = row
    out = pd.DataFrame.from_dict(rows, orient='index')
    out = out[GEOM_COLS] if len(out) else pd.DataFrame(columns=GEOM_COLS)
    out.index.name = 'cof'
    return out


def apply_calibration(df, calib_path):
    """Apply per-descriptor linear correction y = a*x + b from a calibration
    json (produced by calibrate_descriptors.py) to align Zeo++ self-computed
    values onto the training-set scale."""
    import json
    calib = json.load(open(calib_path))
    df = df.copy()
    for feat, c in calib.items():
        if feat in df.columns:
            df[feat] = c['a'] * df[feat] + c['b']
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zeo', required=True)
    ap.add_argument('--cifs', required=True)
    ap.add_argument('--dims-csv', help='可选 CSV: name,dimensions(2/3)')
    ap.add_argument('--default-dim', type=int, default=3)
    ap.add_argument('--calib', help='descriptor_calibration.json，对齐训练口径')
    ap.add_argument('--out', default='curated_features.csv')
    args = ap.parse_args()

    dims = {}
    if args.dims_csv and os.path.exists(args.dims_csv):
        d = pd.read_csv(args.dims_csv)
        dims = dict(zip(d.iloc[:, 0].astype(str), d.iloc[:, 1]))

    out = build_table(args.zeo, args.cifs, dims, args.default_dim)
    if args.calib and os.path.exists(args.calib):
        out = apply_calibration(out, args.calib)
        print(f'已应用校准 {args.calib}')
    out.to_csv(args.out)
    print(f'已保存 {args.out}: {out.shape[0]} 个 COF × {out.shape[1]} 特征')
    miss = out.isna().sum()
    if miss.any():
        print('缺失值统计:\n', miss[miss > 0].to_string())


if __name__ == '__main__':
    main()
