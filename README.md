# Mini AutoML

**A leakage-safe AutoML platform for tabular classification and regression.**

Upload a CSV. The system profiles the data, builds family-aware sklearn pipelines, compares models inside cross-validation, and exports one serializable pipeline that can score raw data.

This is an **educational / research AutoML platform**. It is not a replacement for production ML governance, model risk review, or domain-specific feature design.

[![Python 3.10–3.12](https://img.shields.io/badge/python-3.10--3.12-blue.svg)](#installation)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#license)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](#usage)

Live demo (previous release): https://miniautoml-t9mph2natiawvuawdrdehs.streamlit.app/

---

## Problem statement

Building a trustworthy tabular model is mostly pipeline work: missing values, outliers, encoding, leakage, class imbalance, honest comparison, and a model that still works on raw rows after you close the notebook.

Mini AutoML automates that loop **without fitting transformers on the full dataset before CV**. The goal is to show how a strong ML engineer would structure AutoML — not to hide a giant grid search behind a single accuracy number.

---

## Key features

- Structured **data profiling** (types, missingness, IDs, imbalance, leakage suspects)
- **Family-aware preprocessing** (trees are not forced through the same scaler as SVM/ANN)
- Modular **feature engineering** and **feature selection inside CV**
- Model zoo: linear, trees, boosting, SVM, XGBoost, LightGBM, optional Keras ANN
- Staged search: baselines → promising models → tune top models → optional ensemble
- Honest **baseline vs advanced** comparison (no fabricated lift)
- Classification and regression diagnostics, SHAP, model card, local experiment log
- One `.joblib` artifact: preprocess + engineering + selection + model

---

## Architecture

```
DATA INPUT
    ↓
DATA VALIDATION
    ↓
DATA PROFILING / EDA          ← exploratory only
    ↓
TRAIN / TEST SPLIT            ← happens first
    ↓
LEAKAGE-SAFE ML PIPELINE
    Missing values
    Outliers
    Encoding
    Scaling (model-family aware)
    Feature engineering
    Feature selection
    Optional sampling
    Model
    ↓
CROSS VALIDATION
    ↓
HYPERPARAMETER SEARCH
    ↓
MODEL EVALUATION + COMPARISON
    ↓
EXPLAINABLE AI
    ↓
FINAL PIPELINE → SERIALIZATION → INFERENCE
```

See [docs/architecture.md](docs/architecture.md) for module-level detail.

---

## Pipeline diagram

```
Data → Validation → EDA → Preprocessing → Feature Engineering
    → Feature Selection → Models → Tuning → Evaluation → Best Model
```

Every box that **learns parameters** is an sklearn (or imblearn) step. `cross_validate` refits the whole chain on each fold.

---

## Data validation

The loader rejects empty frames, single-column files, duplicate names, all-null targets, and targets with a single class. Profiling **reports** constant, ID-like, high-missing, and potential leakage columns. Nothing is deleted without a recorded reason.

---

## EDA

The EDA dashboard covers dataset overview, univariate numeric/categorical plots, bivariate relationships, correlation (Pearson / Spearman), and optional pairplots on small samples. Charts are sampled on large frames.

**EDA correlation is not CV feature selection.** The UI states that distinction explicitly.

---

## Preprocessing

| Family | Imputation | Encoding | Scaling |
|---|---|---|---|
| Linear / SVM / ANN | yes | yes | yes (Standard / Robust / …) |
| Tree / boosting | yes | yes | generally off |

Categorical encoding: one-hot for low cardinality, target encoding for high cardinality. Numeric imputation: mean / median / constant / KNN / iterative. Missing indicators are optional.

Outlier methods: IQR, z-score, modified z-score, percentile/winsorize. Default strategy is **clip**, not row deletion. Bounds are learned on training folds only.

---

## Feature engineering

Supported (all optional, explosion-guarded):

- Skewness transforms, log1p, Yeo-Johnson, Box-Cox when valid
- Restricted polynomials and interactions
- Ratio features and quantile binning
- Datetime parts + cyclical sin/cos encodings

Generated columns have readable names (`salary_log1p`, `date_month_sin`).

---

## Feature selection

Strategies: `none`, `automatic`, `light`, `aggressive`.

Automatic adapts to width:

| Features | Default behavior |
|---|---|
| < 30 | light (variance + correlation) |
| 30–100 | correlation + mutual information |
| 100–500 | variance + correlation + SelectKBest |
| 500+ | variance + correlation + model-based |

Selection happens **inside** the CV pipeline.

---

## Model zoo

**Classification:** Logistic Regression, Ridge Classifier, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, HistGradientBoosting, SVM, XGBoost, LightGBM, ANN.

**Regression:** Linear Regression, Ridge, Lasso, ElasticNet, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, HistGradientBoosting, SVR, XGBoost, LightGBM, ANN.

Each spec declares family, scaling need, search space, and whether probabilities exist.

---

## ANN

SciKeras wrappers keep the network inside GridSearchCV / Optuna.

```
Input → Dense → BatchNorm → Activation → Dropout
     → Dense → BatchNorm → Dropout → Output
```

EarlyStopping and ReduceLROnPlateau monitor **`val_loss`**. TensorFlow is optional; the rest of AutoML still runs if it is missing.

---

## Hyperparameter optimization

| Mode | Behavior |
|---|---|
| FAST | small CV, defaults, no ANN, no Optuna |
| STANDARD | short GridSearch + limited Optuna on top models |
| DEEP | Optuna on the top models with pruning and a seeded sampler |

The system does not waste a full Optuna budget on every model.

---

## Ensembling

Optional voting (and stacking helpers) run only when at least two models succeed. The ensemble is compared to the best individual model and kept only if it wins on the held-out split.

---

## Evaluation

Regression: MAE, MSE, RMSE, R², explained variance, MAPE when defined.

Classification: accuracy, balanced accuracy, precision, recall, F1 (macro/weighted), ROC-AUC, PR-AUC, log loss.

Leaderboard columns: model, family, CV score ± std, test score, train time, inference time, status. One failed model is marked `failed` and the run continues.

---

## Explainable AI

TreeExplainer for tree/boosting models, LinearExplainer for linear models, KernelExplainer fallback. SHAP failures are logged and **do not** abort training. Large frames are sampled.

---

## Experiment tracking

Each run appends a JSONL record and a SQLite row under `experiments/` with timestamp, task, configuration, best model, and metrics. No external MLflow/W&B dependency.

---

## Model serialization

```python
from automl.serialization import load_model

model = load_model("best_model.joblib")
preds = model.predict(raw_dataframe)          # raw columns, no manual preprocess
proba = model.predict_proba(raw_dataframe)    # if supported
```

The saved object includes preprocessing, feature engineering, feature selection, and the estimator.

---

## Deployment

The app is a stateful Streamlit process with a heavy ML stack. Serverless hosts (Vercel, Netlify) are not suitable.

**Local**

```bash
streamlit run app/streamlit_app.py
```

**Streamlit Community Cloud**

1. Push this repository.
2. Set the main file to `app/streamlit_app.py`.
3. In Advanced settings, pin **Python 3.12**.
4. Deploy.

**Docker**

```bash
docker build -t mini-automl .
docker run -p 8501:8501 mini-automl
```

---

## Installation

Python 3.10–3.12. Python 3.13+ is not supported at the pinned TensorFlow / SHAP versions.

```bash
git clone https://github.com/Ravana01m/mini_automl.git
cd mini_automl
# if you cloned the nested layout, cd mini-automl
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
pip install -e .
```

ANN extras if you install from `setup.py`:

```bash
pip install -e ".[ann]"
```

---

## Usage

```python
from automl import AutoMLPipeline, ExperimentConfig

config = ExperimentConfig(tuning_mode="fast", skip_ann=True, cv_folds=3)
automl = AutoMLPipeline(config=config)
result = automl.run(df, target_col="target")
print(result["best_model_name"], result["test_metrics"])
automl.export_pipeline("models/best_model.joblib")
```

---

## Screenshots

Run the Streamlit app and capture:

1. Dataset + quality profile
2. EDA correlation view
3. Leaderboard with baseline vs advanced
4. SHAP global importance
5. Download / model card

---

## Project structure

```
mini-automl/
├── app/                    # Streamlit UI
├── src/automl/             # Leakage-safe AutoML package
├── tests/                  # Synthetic-data unit + integration tests
├── data/examples/          # Demo CSVs generated by automl.datasets
├── docs/architecture.md
├── requirements.txt
└── Dockerfile
```

---

## Testing

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

All fixtures are synthetic. Tests do not download external datasets.

---

## Example results

Results vary by seed and hardware. These are **illustrative** numbers from the bundled synthetic sets in FAST mode — they are not a claim of SOTA.

| Dataset | Task | Typical winner | Notes |
|---|---|---|---|
| `data/examples/regression.csv` | regression | RandomForest / Ridge | Advanced may or may not beat baseline |
| `data/examples/binary.csv` | binary | Logistic / trees | Report F1, not only accuracy |
| `data/examples/imbalanced.csv` | binary | class-weighted linear | Prefer balanced accuracy / PR-AUC |

If the advanced pipeline is worse, the UI shows a negative change percentage. That is expected behavior.

---

## Limitations

- Tabular data only; no images, audio, or raw documents
- Text-like columns are profiled, not turned into embeddings
- ANN quality depends on TensorFlow being installed and on runtime hardware
- Small datasets make held-out metrics noisy
- Automatic FE cannot replace domain features
- Not a governed production training service

---

## Future roadmap

- Partial-dependence and calibration-aware model selection
- Time-based splits for temporal tables
- Optional MLflow export
- Narrower Streamlit Cloud dependency extra without TensorFlow

---

## License

MIT
