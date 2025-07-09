import pytest
import pandas as pd
import matplotlib.pyplot as plt
from definition_8d2e3e95b67d493487c8f82506796bfe import plot_parsing_metrics

@pytest.fixture
def sample_dataframe():
    data = {
        'natural_language_input': ['add 2 and 3', 'calculate (4 + 5) * 2', 'define a function', 'what is the capital of France', 'explain quantum entanglement'],
        'symbolic_output': ['2 + 3', '(4 + 5) * 2', 'def func(): pass', 'Paris', 'Complex Explanation'],
        'parsing_success': [True, True, True, True, False],
        'parsing_time_ms': [10, 20, 30, 15, 100],
        'complexity_level': ['Simple', 'Medium', 'Medium', 'Simple', 'Complex'],
        'input_length': [10, 20, 17, 25, 28]
    }
    return pd.DataFrame(data)

def test_plot_parsing_metrics_valid_dataframe(sample_dataframe, monkeypatch):
    # Mock plt.show to prevent actual plot display during testing
    monkeypatch.setattr(plt, 'show', lambda: None)
    try:
        plot_parsing_metrics(sample_dataframe)
    except Exception as e:
        pytest.fail(f"plot_parsing_metrics raised an exception: {e}")

def test_plot_parsing_metrics_empty_dataframe():
    empty_df = pd.DataFrame()
    try:
        plot_parsing_metrics(empty_df)
    except Exception as e:
        assert isinstance(e, KeyError), "Expected KeyError due to missing columns"

def test_plot_parsing_metrics_missing_columns(sample_dataframe):
    # Create a DataFrame missing a required column ('parsing_time_ms')
    missing_col_df = sample_dataframe.drop('parsing_time_ms', axis=1)
    with pytest.raises(KeyError):
        plot_parsing_metrics(missing_col_df)

def test_plot_parsing_metrics_non_dataframe_input():
    with pytest.raises(AttributeError):
        plot_parsing_metrics("not a dataframe")

def test_plot_parsing_metrics_with_nan_values(sample_dataframe, monkeypatch):
    # Introduce NaN values into the DataFrame
    sample_dataframe.loc[0, 'parsing_time_ms'] = float('nan')
    monkeypatch.setattr(plt, 'show', lambda: None)
    try:
        plot_parsing_metrics(sample_dataframe)
    except Exception as e:
        pytest.fail(f"plot_parsing_metrics raised an exception with NaN values: {e}")
