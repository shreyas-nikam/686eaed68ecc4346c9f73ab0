import pytest
import pandas as pd
import matplotlib.pyplot as plt
from definition_7d3cf63fd1644f0ca00680182381ffdf import plot_parsing_metrics

@pytest.fixture
def sample_dataframe():
    data = {
        'natural_language_input': ['add 1 and 2', 'is it raining?', 'complex query'],
        'symbolic_output': ['1 + 2', 'unknown', 'very complex expression'],
        'parsing_success': [True, False, True],
        'parsing_time_ms': [10, 50, 150],
        'complexity_level': ['Simple', 'Medium', 'Complex'],
        'input_length': [10, 12, 13]
    }
    return pd.DataFrame(data)

def test_plot_parsing_metrics_runs(sample_dataframe, monkeypatch):
    monkeypatch.setattr(plt, 'show', lambda: None)
    plot_parsing_metrics(sample_dataframe)

def test_plot_parsing_metrics_empty_dataframe():
    df = pd.DataFrame()
    try:
        plot_parsing_metrics(df)
    except Exception as e:
        assert isinstance(e, (ValueError, TypeError, KeyError))

def test_plot_parsing_metrics_no_display(sample_dataframe, monkeypatch):
    monkeypatch.delattr(plt, 'show', raising=False)
    plot_parsing_metrics(sample_dataframe)
    
def test_plot_parsing_metrics_invalid_input_type():
    with pytest.raises(TypeError):
        plot_parsing_metrics("not a dataframe")

def test_plot_parsing_metrics_with_missing_columns(sample_dataframe):
    df = sample_dataframe.drop(columns=['parsing_time_ms'])
    try:
        plot_parsing_metrics(df)
    except Exception as e:
        assert isinstance(e, KeyError)
