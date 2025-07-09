import pytest
from definition_343431bfe39f4375a9b1394f927d7a3c import generate_benchmark_comparison_plot
import pandas as pd
import matplotlib.pyplot as plt
from unittest.mock import patch

@pytest.fixture
def mock_data():
    data = pd.DataFrame({
        'model_name': ['ModelA', 'ModelA', 'ModelB', 'ModelB'],
        'benchmark_category': ['Logic', 'Associative Prediction', 'Logic', 'Associative Prediction'],
        'vertex_score': [0.8, 0.6, 0.7, 0.9]
    })
    return data

def test_generate_benchmark_comparison_plot_valid_data(mock_data):
    try:
        generate_benchmark_comparison_plot(mock_data)
    except Exception as e:
        pytest.fail(f"Unexpected exception: {e}")

def test_generate_benchmark_comparison_plot_empty_data():
    empty_data = pd.DataFrame({'model_name': [], 'benchmark_category': [], 'vertex_score': []})
    try:
        generate_benchmark_comparison_plot(empty_data)
    except Exception as e:
        pytest.fail(f"Unexpected exception: {e}")

def test_generate_benchmark_comparison_plot_missing_column():
    incomplete_data = pd.DataFrame({
        'model_name': ['ModelA', 'ModelB'],
        'vertex_score': [0.8, 0.7]
    })
    with pytest.raises(KeyError):
        generate_benchmark_comparison_plot(incomplete_data)

@patch("matplotlib.pyplot.show")
def test_generate_benchmark_comparison_plot_calls_show(mock_show, mock_data):
    generate_benchmark_comparison_plot(mock_data)
    mock_show.assert_called_once()

def test_generate_benchmark_comparison_plot_non_dataframe_input():
    with pytest.raises(AttributeError):
        generate_benchmark_comparison_plot([1, 2, 3])
