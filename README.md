# ML

Python workspace for machine learning experiments and reusable utilities.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .[dev]
```

## Run Tests

```bash
pytest
```

## Train Load Prediction Models

The training pipeline reads a profiling CSV, trains practical baseline models, selects the
best model by validation RMSE, and saves the full scikit-learn pipeline for inference.

```bash
python -m ml_project.train \
  --data data/profiling_dataset.csv \
  --output-dir artifacts
```

For a quick smoke test with the 3-row template only:

```bash
python -m ml_project.train \
  --data data/profiling_dataset_template.csv \
  --output-dir artifacts \
  --allow-small-dataset \
  --models dummy_mean ridge
```

Outputs:

```text
artifacts/load_predictor.joblib
artifacts/metrics.json
```

The template file is only for checking the CSV shape. Real training should use a larger
`data/profiling_dataset.csv` file following `docs/dataset_schema.md`.

## Project Layout

```text
src/ml_project/     Python package code
tests/             Test suite
```
