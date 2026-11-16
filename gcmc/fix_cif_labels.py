# -*- coding: utf-8 -*-
"""
Phase D fix — rewrite CIF _atom_site_label = _atom_site_type_symbol (element).

Mercado / many COF CIFs put a pure number (1,2,3...) in the _atom_site_label
column and the element in _atom_site_type_symbol. RASPA types framework atoms
by the LABEL column, so numeric labels match no force-field entry and every
framework atom gets ZERO Lennard-Jones (Host/Adsorbate VDW = 0). Replacing the
label with the element symbol makes them match pseudo_atoms.def / force field.

Usage:
  python -m gcmc.fix_cif_labels \
      --cifs ./mercado_cifs --out ./mercado_cifs_raspa
  python -m gcmc.fix_cif_labels \
      --cifs ./curated_cifs --out ./curated_cifs_raspa
"""
import argparse
import glob
import os


def fix_one(src, dst):
    lines = open(src, encoding='utf-8', errors='ignore').read().splitlines()
    cols, hdr_start, hdr_end = [], None, None
    for i, ln in enumerate(lines):
        if ln.strip().startswith('_atom_site_'):
            if hdr_start is None:
                hdr_start = i
            cols.append(ln.strip())
            hdr_end = i
    if hdr_start is None or '_atom_site_label' not in cols \
            or '_atom_site_type_symbol' not in cols:
        return False
    li, ti, ncol = (cols.index('_atom_site_label'),
                    cols.index('_atom_site_type_symbol'), len(cols))
    out, j, changed = lines[:], hdr_end + 1, 0
    while j < len(lines):
        toks = lines[j].split()
        if len(toks) < ncol:
            break
        if toks[li] != toks[ti]:
            toks[li] = toks[ti]
            out[j] = '  '.join(toks)
            changed += 1
        j += 1
    open(dst, 'w', encoding='utf-8').write('\n'.join(out) + '\n')
    return changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cifs', required=True)
    ap.add_argument('--out', required=True, help='输出目录（勿与输入相同以防覆盖原始）')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    n_ok = n_skip = 0
    for src in sorted(glob.glob(os.path.join(args.cifs, '*.cif'))):
        dst = os.path.join(args.out, os.path.basename(src))
        r = fix_one(src, dst)
        if r is False:
            n_skip += 1
            print('跳过(无 label/type_symbol 列):', os.path.basename(src))
        else:
            n_ok += 1
    print(f'完成: {n_ok} 个 CIF 已改 label→元素 → {args.out}/ ；跳过 {n_skip}')


if __name__ == '__main__':
    main()
