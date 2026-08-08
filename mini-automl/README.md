# 🤖 Mini AutoML Pipeline

> A "mini DataRobot" — upload any CSV and get automated data cleaning, feature engineering, model training, hyperparameter tuning, explainability, and a downloadable production-ready model.

![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)
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
- Python 3.12 (3.10-3.12 supported; 3.13+ is not, since tensorflow-cpu and shap don't ship 3.13 wheels at the pinned versions)
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

See **Deployment** below for running via Docker or hosting it live.

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

## 🌐 Deployment

This is a stateful Streamlit server app with a heavy ML backend (TensorFlow, XGBoost,
LightGBM, SHAP, Optuna) — it needs a host that keeps a persistent Python process
running, so **serverless platforms like Vercel or Netlify cannot run it** (no
persistent process, and the dependency set alone exceeds typical serverless
deployment-size limits). Use one of the following instead:

### Option A — Streamlit Community Cloud (free, simplest)
1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → pick the repo.
3. Set **Main file path** to `app/streamlit_app.py`.
4. Open **Advanced settings** and explicitly select **Python 3.12**.
   Do this even though a `runtime.txt` is included — Community Cloud has repeatedly
   ignored `runtime.txt` and defaulted to Python 3.13/3.14 for other users, which
   breaks the TensorFlow install (no 3.13+ wheels at the pinned version). The
   Advanced settings dropdown is the reliable way to pin it.
5. Deploy.

### Option B — Hugging Face Spaces (free, Docker-based)
1. Create a new Space → SDK: **Docker**.
2. Push this repo to the Space's git remote (the included `Dockerfile` is used as-is).
3. Set the Space's port to `8501` if prompted.

Docker-based hosts (HF Spaces, Render, Railway, Fly.io) sidestep the Python-version
issue entirely, since the Dockerfile pins Python 3.12 directly — this makes them the
more reliable option if Option A's Python version keeps getting overridden.

### Local Docker
```bash
docker build -t mini-automl .
docker run -p 8501:8501 mini-automl
```

## 📄 License

MIT
