import pandas as pd

MAX_UNIQUE_FOR_CLASSIFICATION = 20

def detect_task_type(target_series: pd.Series) -> str:
    """
    Detects whether the machine learning task is classification or regression
    based on the target series characteristics.
    """
    if target_series.empty or target_series.isnull().all():
        raise ValueError("Target series is empty or contains all nulls.")
    
    if (pd.api.types.is_object_dtype(target_series) or 
        isinstance(target_series.dtype, pd.CategoricalDtype) or 
        pd.api.types.is_bool_dtype(target_series)):
        return 'classification'
    
    if pd.api.types.is_numeric_dtype(target_series):
        if target_series.nunique(dropna=True) <= MAX_UNIQUE_FOR_CLASSIFICATION:
            return 'classification'
        else:
            return 'regression'
            
    return 'regression'
