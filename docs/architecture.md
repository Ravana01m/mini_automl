# Architecture

See the Mermaid diagram in the main [README.md](../README.md#architecture).

## Module Responsibilities

| Module | Responsibility |
|---|---|
| `detector.py` | Auto-detect classification vs regression from target column |
| `eda.py` | Generate automated EDA report (distributions, correlations, missing values) |
| `preprocessing.py` | Build dynamic ColumnTransformer (imputation, encoding, scaling) |
| `feature_engineering.py` | Auto feature construction (polynomial, datetime, log, interactions) |
| `feature_selection.py` | Variance threshold + correlation filter + SelectKBest |
| `model_registry.py` | Registry of 6 classification + 6 regression model configs |
| `ann_builder.py` | Keras model factory for ANN classifier/regressor |
| `trainer.py` | Cross-validated training, GridSearchCV, Optuna tuning |
| `explainer.py` | SHAP global + local explanations |
| `pipeline_builder.py` | Orchestrator: assembles full sklearn Pipeline |
| `utils.py` | Validation, logging, error handling helpers |

## Data Flow

```
Raw CSV → detect_task_type() → generate_eda_report()
       → build_preprocessor() → build_feature_engineer()
       → build_feature_selector() → get_models()
       → train_and_evaluate() → tune_top_models()
       → explain_model() → export_pipeline()
```
