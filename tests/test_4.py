import pytest
import pandas as pd
from definition_839ad43d02c74547b70b37a49f89a6c7 import display_metrics

@pytest.fixture
def mock_metrics_data():
    # Create a sample DataFrame for testing
    data = {'processing_time_ms': [10, 20, 15, 25, 30],
            'execution_accuracy': [0.9, 0.8, 0.95, 0.75, 0.85],
            'task_type': ['arithmetic', 'logical', 'arithmetic', 'query', 'logical']}
    return pd.DataFrame(data)

def test_display_metrics_valid_data(mock_metrics_data, monkeypatch):
    # Test that the function runs without errors with valid data
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)  # Suppress plot display
    try:
        display_metrics(mock_metrics_data)
    except Exception as e:
        pytest.fail(f"display_metrics raised an exception: {e}")

def test_display_metrics_empty_dataframe():
    # Test the function handles an empty DataFrame gracefully
    empty_df = pd.DataFrame()
    try:
        display_metrics(empty_df) #Should not raise an error
    except Exception as e:
        pytest.fail(f"display_metrics raised an exception with empty DataFrame: {e}")

def test_display_metrics_missing_columns(mock_metrics_data):
    # Test the function raises an error when required columns are missing
    del mock_metrics_data['execution_accuracy']
    with pytest.raises(KeyError):
        display_metrics(mock_metrics_data)

def test_display_metrics_non_numeric_data(monkeypatch):
    # Test when processing_time_ms or execution_accuracy are strings
    data = {'processing_time_ms': ['10', '20', '15', '25', '30'],
            'execution_accuracy': ['0.9', '0.8', '0.95', '0.75', '0.85'],
            'task_type': ['arithmetic', 'logical', 'arithmetic', 'query', 'logical']}
    df = pd.DataFrame(data)
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
    try:
        display_metrics(df)
    except TypeError as e:
         assert "unsupported operand type(s)" in str(e)