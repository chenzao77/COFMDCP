# COFMDCP

COFMDCP is a machine-learning model for predicting methane deliverable capacity in covalent organic frameworks (COFs). It combines nested Stacking, feature ablation, six-descriptor cross-database transfer screening, and RASPA validation analysis.

## Repository structure

- `core/`: shared feature construction and the MLP architecture.
- `training/`: nested Stacking, saved-model prediction, and feature ablation.
- `transfer/`: six-descriptor transfer modelling and CURATED-COF screening.
- `gcmc/`: RASPA input preparation and output parsing.
- `scripts/`: Zeo++ batch launcher.
- `raspa_inputs/`: methane GCMC force-field and input templates.

Datasets, CIF collections, trained weights, simulation outputs, and manuscript figures are not redistributed. Before training, place `properties.csv` and `linker.csv` in the repository root. `linker.csv` must contain at least the `Linker_ID` and `SMILES` columns.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Zeo++ and RASPA 2.0.50 are external executables required only for structure characterization and GCMC validation.

## Main training

```bash
python -m training.nested_stacking \
  --outer-folds 5 --inner-folds 4 \
  --trials 30 --final-trials 30 --bootstrap 10000 --prefix nested
```

XGBoost, LightGBM, and MLP are trained as the three base learners, and Ridge is used as the meta-learner.

## Feature ablation

Each feature set is independently trained with the complete nested Stacking procedure.

```bash
python -m training.ablation \
  --outer-folds 5 --inner-folds 4 --trials 30 \
  --mlp-max-epochs 150 --mlp-patience 20 \
  --output-dir results/ablation
```

## Transfer screening

```bash
python -m transfer.train_geometry_model --outer-folds 5 --inner-folds 4
bash scripts/run_zeopp.sh ./curated_cifs /path/to/zeo++/network ./zeo_out 12
python -m transfer.build_curated_features \
  --zeo ./zeo_out --cifs ./curated_cifs
python -m transfer.screen_curated
```

Cross-database prediction uses only void fraction, density, surface area, and the three pore diameters (Di, Df, and Dif). Raw atom counts and default dimensionality are excluded.

## GCMC preparation

The `raspa_inputs/` directory contains the methane GCMC templates used with RASPA 2.0.50 at 298 K.

