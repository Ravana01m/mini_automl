from __future__ import annotations
import pandas as pd
import logging

class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass

def validate_csv(df: pd.DataFrame, filename: str) -> None:
    """Validates the structure of the input CSV dataset."""
    MAX_ROWS = 1000000
    MAX_COLS = 500
    
    if df.empty:
        raise ValidationError(f"Dataset in {filename} is empty.")
    if len(df.columns) < 2:
        raise ValidationError(f"Dataset in {filename} must have at least 2 columns.")
    if len(df) > MAX_ROWS:
        raise ValidationError(f"Dataset in {filename} exceeds maximum allowed rows ({MAX_ROWS}).")
    if len(df.columns) > MAX_COLS:
        raise ValidationError(f"Dataset in {filename} exceeds maximum allowed columns ({MAX_COLS}).")

def validate_target_column(df: pd.DataFrame, target_col: str) -> None:
    """Validates the target column for modeling."""
    if target_col not in df.columns:
        raise ValidationError(f"Target column '{target_col}' not found in dataset.")
    
    if df[target_col].isnull().all():
        raise ValidationError(f"Target column '{target_col}' contains only null values.")
    
    if df[target_col].nunique(dropna=True) < 2:
        raise ValidationError(f"Target column '{target_col}' must have at least 2 unique values.")

def get_column_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Separates columns into numeric and categorical types, ignoring all-null columns."""
    numeric_cols = []
    categorical_cols = []
    
    for col in df.columns:
        if df[col].isnull().all():
            continue
        
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        elif (pd.api.types.is_object_dtype(df[col]) or 
              isinstance(df[col].dtype, pd.CategoricalDtype) or 
              pd.api.types.is_bool_dtype(df[col])):
            categorical_cols.append(col)
            
    return numeric_cols, categorical_cols

def setup_logging(level: int = logging.INFO) -> None:
    """Sets up standard logging for the automl package."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
