# 🤖 Mini AutoML Pipeline

> A "mini DataRobot" — upload any CSV and get automated data cleaning, feature engineering, model training, hyperparameter tuning, explainability, and a downloadable production-ready model.

![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-deployed-red.svg)

---

## 🎯 Problem Statement

Building ML models involves repetitive steps: data cleaning, feature engineering, model selection, hyperparameter tuning, and explainability. This project automates the entire pipeline, reducing model development time from hours to minutes.

**Key Capabilities:**
- Upload **any** CSV — the pipeline auto-detects classification vs regression
- Automated preprocessing: missing values, outliers, encoding, scaling
- Automated feature engineering: polynomial features, datetime extraction, log transforms
- Trains & compares **6 models** including a TensorFlow/Keras neural network
- Two-stage tuning: GridSearchCV → Optuna (top 2 models)
- SHAP explainability: global feature importance + per-prediction explanations
- Download a **self-contained model bundle** (preprocessing + model in one `.joblib` file)

## 🏗️ Architecture

*Architecture diagram will be added after implementation*

## 🚀 Quick Start

### Prerequisites
- Python 3.13+
- pip

### Installation
```bash
git clone https://github.com/YOUR_USERNAME/mini-automl.git
cd mini-automl
pip install -r requirements.txt
pip install -e .
```

### Run the App
```bash
streamlit run app/streamlit_app.py
```

### Run Tests
```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

### Docker
```bash
docker build -t mini-automl .
docker run -p 8501:8501 mini-automl
```

## 📁 Project Structure

```
mini-automl/
├── src/automl/          # Core ML pipeline modules
├── app/                 # Streamlit web application
│   └── components/      # UI components (upload, charts, SHAP, download)
├── tests/               # Pytest suite (2+ dataset generalization tests)
├── models/              # Fitted models (runtime, gitignored)
├── data/                # Sample datasets
└── docs/                # Architecture documentation
```

## 🔬 Design Decisions

*Will be expanded after implementation*

| Decision | Choice | Rationale |
|---|---|---|
| Single sklearn Pipeline | ✅ | One `.joblib` = preprocessing + model, works on raw data |
| IQR clipping over row deletion | Clipping | Composable in Pipeline, works at prediction time |
| Keras ANN via scikeras | ✅ | Real TensorFlow on resume, but fits sklearn Pipeline |
| Optuna over Bayesian sklearn | Optuna | Better pruning, visualization, flexible search spaces |
| Plotly for charts | ✅ | Interactive, polished for portfolio demos |

## 📊 Demo

*Screenshots and GIF will be added after implementation*

## 🌐 Live Demo

*Deployment link will be added*

## 📄 License

MIT
