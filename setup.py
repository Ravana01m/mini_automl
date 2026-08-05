"""Setup script for mini-automl package."""
from setuptools import setup, find_packages

setup(
    name="mini-automl",
    version="1.0.0",
    description="A mini AutoML pipeline that automates ML model development",
    author="Ravin",
    python_requires=">=3.13",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "scikit-learn>=1.5.0",
        "xgboost>=2.1.0",
        "lightgbm>=4.4.0",
        "tensorflow-cpu>=2.19.0",
        "scikeras>=0.13.0",
        "optuna>=3.6.0",
        "shap>=0.45.0",
        "streamlit>=1.35.0",
        "plotly>=5.22.0",
        "pandas>=2.2.0",
        "numpy>=1.26.0",
        "category-encoders>=2.6.0",
        "joblib>=1.4.0",
        "matplotlib>=3.9.0",
        "seaborn>=0.13.0",
    ],
)
