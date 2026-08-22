# Architecture

Mini AutoML is a leakage-safe research platform. Every learned transform lives inside an sklearn-compatible pipeline and is refit on each cross-validation fold.

```
Raw CSV
  → Validation
  → Profiling / EDA (exploratory only)
  → Train / test split
  → Per-model pipeline
        Column pruning (train stats)
        Datetime extraction
        Outlier bounds (train stats)
        Family-aware preprocess
        Feature engineering
        Feature selection
        Optional oversampling
        Estimator
  → Staged model search + CV
  → Optional ensemble
  → Held-out evaluation
  → SHAP / model card / serialization
```

## Module responsibilities

| Module | Responsibility |
|---|---|
| `config.py` | Experiment configuration and global seeds |
| `profiling.py` | Structured `DataProfile` |
| `quality.py` | Outliers and uninformative-column pruning |
| `preprocessing.py` | Family-aware ColumnTransformer |
| `feature_engineering.py` | Log, datetime, polynomial, interactions, bins |
| `feature_selection.py` | Variance, correlation, MI, model-based selection |
| `sampling.py` | CV-safe SMOTE / random oversampling |
| `model_registry.py` | Model zoo with family metadata |
| `ann_builder.py` | SciKeras ANN with `val_loss` early stopping |
| `trainer.py` | Staged search, GridSearch, Optuna |
| `evaluation.py` / `diagnostics.py` | Metrics and residual / class plots |
| `explainer.py` | SHAP with failure isolation |
| `pipeline_builder.py` | Orchestrator |
| `serialization.py` | Save / load / predict on raw frames |
| `experiment.py` | Local JSON + SQLite history |

## Leakage rules

- Split before fitting anything that learns parameters.
- EDA correlation is never used as the CV selection decision.
- Oversampling is never applied to validation or test folds.
- ANN early stopping monitors validation loss, not training loss.
