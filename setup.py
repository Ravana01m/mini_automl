"""Setup script for mini-automl package."""
from setuptools import setup, find_packages

setup(
    name="mini-automl",
    version="2.0.0",
    description="Leakage-safe AutoML platform for classification and regression",
    author="Ravin",
    python_requires=">=3.10,<3.13",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "scikit-learn>=1.5.0,<1.6",
        "xgboost>=2.1.0",
        "lightgbm>=4.4.0",
        "optuna>=3.6.0",
        "shap>=0.45.0",
        "imbalanced-learn>=0.12.0",
        "streamlit>=1.35.0",
        "plotly>=5.22.0",
        "pandas>=2.2.0",
        "numpy>=1.26.0",
        "scipy>=1.11.0",
        "category-encoders>=2.6.0,<2.8",
        "joblib>=1.4.0",
        "matplotlib>=3.9.0",
        "seaborn>=0.13.0",
    ],
    extras_require={
        "ann": ["tensorflow-cpu>=2.16.0,<3.0", "scikeras>=0.13.0"],
        "dev": ["pytest>=8.2.0,<9.0", "pytest-cov>=5.0.0,<6.0", "ruff>=0.4.0,<1.0"],
    },
)
