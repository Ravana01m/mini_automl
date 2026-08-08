from automl.detector import detect_task_type
import pandas as pd
import pytest
import numpy as np

class TestDetectTaskType:
    def test_categorical_target_is_classification(self):
        s = pd.Series(['cat', 'dog', 'bird', 'cat', 'dog'])
        assert detect_task_type(s) == 'classification'
    
    def test_boolean_target_is_classification(self):
        s = pd.Series([True, False, True, False, True])
        assert detect_task_type(s) == 'classification'
    
    def test_few_unique_numeric_is_classification(self):
        s = pd.Series([0, 1, 2, 0, 1, 2, 0, 1])
        assert detect_task_type(s) == 'classification'
    
    def test_many_unique_numeric_is_regression(self):
        s = pd.Series(np.random.uniform(0, 100, 1000))
        assert detect_task_type(s) == 'regression'
    
    def test_empty_series_raises(self):
        with pytest.raises(ValueError):
            detect_task_type(pd.Series(dtype=float))
    
    def test_all_null_raises(self):
        with pytest.raises(ValueError):
            detect_task_type(pd.Series([np.nan, np.nan, np.nan]))
