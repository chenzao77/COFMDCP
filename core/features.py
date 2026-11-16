# -*- coding: utf-8 -*-
"""
Shared feature-engineering for the COF methane deliverable-capacity model.

Single source of truth used by the training, prediction, and transfer modules
so every model builds an identical feature matrix.

Y  = column 27, deliverable capacity [v STP/v]
X  = geometric/textural descriptors  [0,1,3,5,26,30,38-44,48-50]
     + 2 x 167-bit MACCS fingerprints (linkerA, linkerB)
"""
import re
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

# --- shared CV config so every base learner + stacking use identical folds ---
SEED = 42
N_FOLDS = 5


def make_folds(n_samples, n_splits=N_FOLDS, seed=SEED):
    """Deterministic random 5-fold (train_idx, val_idx). Kept for reference;
    the pipeline now uses make_group_folds to avoid polymorph leakage."""
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(kf.split(np.arange(n_samples)))


def load_cv_groups(props_csv='properties.csv'):
    """Group id = unordered linker pair, aligned 1:1 with build_dataset() rows.

    ~27% of rows share the same (linker pair, net) as a polymorph variant;
    grouping by linker pair keeps all variants of a pair within a single fold,
    preventing optimistic train/val leakage and giving an honest estimate of
    generalization to unseen linker combinations (the Phase C transfer setting).
    """
    df = pd.read_csv(props_csv)
    Y = df.iloc[:, TARGET_COL].values.astype(np.float64)
    mask = ~np.isnan(Y) & ~np.isinf(Y)
    a = df.iloc[:, LINKER_A_COL].apply(extract_linker_id)
    b = df.iloc[:, LINKER_B_COL].apply(extract_linker_id)
    pair = pd.Series(['|'.join(sorted([x, y])) for x, y in zip(a, b)])
    pair = pair[mask].reset_index(drop=True)
    return pair.astype('category').cat.codes.values


def make_group_folds(groups, n_splits=N_FOLDS):
    """Deterministic GroupKFold (train_idx, val_idx): all rows sharing a group
    (unordered linker pair) stay in the same fold. Shared by every base learner
    and stacking so OOF rows align."""
    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=n_splits)
    return list(gkf.split(np.arange(len(groups)), groups=groups))


def regression_metrics(y_true, y_pred):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from scipy.stats import pearsonr, spearmanr
    return dict(
        MAE=float(mean_absolute_error(y_true, y_pred)),
        RMSE=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        R2=float(r2_score(y_true, y_pred)),
        Pearson=float(pearsonr(y_true, y_pred)[0]),
        Spearman=float(spearmanr(y_true, y_pred)[0]),
    )


def save_oof(name, y_true, y_pred):
    """Persist out-of-fold predictions as <name>_oof.csv for stacking/eval."""
    pd.DataFrame({'y_true': y_true, 'y_pred': y_pred}).to_csv(
        f'{name}_oof.csv', index=False)
    return f'{name}_oof.csv'

# --- column indices in properties.csv -------------------------------------
TARGET_COL = 27                                  # deliverable capacity [v STP/v]
BASE_COLS = [0, 1, 3, 5, 26, 30, 38, 39, 40, 41, 42, 43, 44, 48, 49, 50]
LINKER_A_COL = 28
LINKER_B_COL = 29
CATEGORICAL = ['bond_type', 'net']               # one-hot encoded base columns
N_MACCS = 167

# Cross-database transfer must only use intensive, cell-representation-invariant
# pore descriptors.  Raw atom counts change when the same crystal is written as
# a primitive/conventional/supercell CIF, and ``dimensions`` is not available
# consistently in CURATED-COFs.  Keep this list as the single source of truth
# for Phase C/D transfer modelling.
TRANSFER_FEATURES = [
    'void_fraction__widom',
    'density__kg/m^3',
    'surface_area__m^2/g',
    'largest_included_sphere_diameter__A',
    'largest_free_sphere_diameter__A',
    'largest_included_sphere_along_free_sphere_path_diameter__A',
]


def sanitize(name):
    """Strip whitespace and replace XGBoost-forbidden chars '[', ']', '<'."""
    s = str(name).strip()
    for ch in '[]<':
        s = s.replace(ch, '_')
    return s.replace(' ', '_').strip('_')


def get_maccs_fingerprint(smiles):
    """167-bit MACCS key list; all-zero fallback for missing/invalid SMILES."""
    if pd.isna(smiles) or str(smiles).strip() == '':
        return [0] * N_MACCS
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is not None:
            return list(MACCSkeys.GenMACCSKeys(mol))
    except Exception:
        pass
    return [0] * N_MACCS


def extract_linker_id(linker_str):
    """'linker91_CO' -> '91'; first integer found, else stripped string."""
    match = re.search(r'\d+', str(linker_str))
    return str(match.group()) if match else str(linker_str).strip()


def load_smiles_dict(smiles_csv='linker.csv'):
    df = pd.read_csv(smiles_csv)
    df['Linker_ID_str'] = df['Linker_ID'].apply(extract_linker_id)
    return dict(zip(df['Linker_ID_str'], df['SMILES']))


def build_maccs_block(linker_a_raw, linker_b_raw, smiles_dict):
    """Return a (n, 334) DataFrame of MACCS-A | MACCS-B features.

    Fingerprints are cached per unique SMILES (only ~110 distinct linkers),
    so we parse each molecule once instead of once per row.
    """
    fp_cache = {smi: get_maccs_fingerprint(smi)
                for smi in set(smiles_dict.values())}

    def fp_for(linker):
        smi = smiles_dict.get(extract_linker_id(linker))
        return fp_cache.get(smi) if smi is not None else [0] * N_MACCS

    maccs_a = np.array([fp_for(x) for x in linker_a_raw], dtype=np.int8)
    maccs_b = np.array([fp_for(x) for x in linker_b_raw], dtype=np.int8)
    cols = ([f'maccsA_{i}' for i in range(N_MACCS)] +
            [f'maccsB_{i}' for i in range(N_MACCS)])
    return pd.DataFrame(np.hstack((maccs_a, maccs_b)), columns=cols)


def _clean(X):
    """Numeric coercion -> float32 (surfaces inf) -> inf->nan -> mean-fill."""
    X = X.apply(pd.to_numeric, errors='coerce').astype(np.float32)
    X[np.isinf(X)] = np.nan
    return X.fillna(X.mean())


def build_dataset(props_csv='properties.csv', smiles_csv='linker.csv',
                  return_groups=False):
    """
    Build the full (X, Y) modelling dataset.

    Returns X (DataFrame, float32, no NaN/inf), Y (1-D float array).
    If return_groups=True also returns a dict mapping
    'geometry' / 'categorical' / 'maccs' -> list[column names] for ablation.
    """
    df = pd.read_csv(props_csv)
    Y = df.iloc[:, TARGET_COL].values.astype(np.float64)

    X_base = df.iloc[:, BASE_COLS].copy()
    col_bond = X_base.columns[1]   # ' bond type'
    col_net = X_base.columns[5]    # ' net'
    geom_cols = [c for c in X_base.columns if c not in (col_bond, col_net)]
    X_base = pd.get_dummies(X_base, columns=[col_bond, col_net], drop_first=True)
    cat_cols = [c for c in X_base.columns if c not in geom_cols]

    smiles_dict = load_smiles_dict(smiles_csv)
    df_maccs = build_maccs_block(df.iloc[:, LINKER_A_COL],
                                 df.iloc[:, LINKER_B_COL], smiles_dict)

    X = pd.concat([X_base.reset_index(drop=True), df_maccs], axis=1)
    X = _clean(X)

    mask = ~np.isnan(Y) & ~np.isinf(Y)
    if (~mask).sum() > 0:
        X = X[mask].reset_index(drop=True)
        Y = Y[mask]

    # sanitize names: XGBoost/SHAP DMatrix forbids '[', ']', '<' (unit strings)
    X.columns = [sanitize(c) for c in X.columns]

    if return_groups:
        groups = {
            'geometry': [sanitize(c) for c in geom_cols if sanitize(c) in X.columns],
            'categorical': [sanitize(c) for c in cat_cols if sanitize(c) in X.columns],
            'maccs': [c for c in X.columns if c.startswith('maccs')],
        }
        return X, Y, groups
    return X, Y


if __name__ == '__main__':
    X, Y, g = build_dataset(return_groups=True)
    print('X', X.shape, '| Y', Y.shape)
    print('geometry:', len(g['geometry']),
          '| categorical:', len(g['categorical']),
          '| maccs:', len(g['maccs']))
    print('any nan/inf in X:', bool(np.isinf(X.values).any() or np.isnan(X.values).any()))
