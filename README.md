# ML

Python workspace for machine learning experiments and reusable utilities.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .[dev]
```

To run the Streamlit dashboard, install the UI extras:

```bash
python -m pip install -e .[ui]
```

## Run Tests

```bash
pytest
```

## Streamlit Dashboard

Run the bilingual dashboard locally:

```bash
streamlit run src/ml_project/streamlit_app.py
```

The dashboard opens in your browser and supports English/Chinese switching from the sidebar. Use it to upload a profiling CSV, validate the dataset, preview the data, train models, view metrics, and download the saved model and metrics JSON. When demonstrating with the small 3-row template, enable **Allow small dataset (smoke/demo mode)** before training.

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

## Deploy on Zeabur

This Streamlit app can be deployed to Zeabur with a managed PostgreSQL database
and a persistent volume for artifacts.

1. Create a project on Zeabur and link this repository.
2. Add a PostgreSQL service and copy its connection URI.
3. In **Variables**, set:
   - `DATABASE_URL` = `postgresql+psycopg://...` (use the service URI)
   - `ARTIFACT_ROOT` = `/data/training_runs`
4. Mount a persistent volume at `/data` so models and datasets survive restarts.
5. Deploy. Zeabur reads `zbpack.json`, uses Python 3.11, and starts Streamlit on
   the assigned `$PORT`.

If you prefer to pin the Python version explicitly, also set
`ZBPACK_PYTHON_VERSION=3.11` in the service variables.

See `.env.example` for local configuration.

## Project Layout

```text
src/ml_project/     Python package code
tests/             Test suite
zbpack.json        Zeabur build and start configuration
.env.example       Environment variable template
```
