import pytest
from definition_9da4284ce6e44224a2e495d9509d2693 import generate_scatter_plot
import matplotlib.pyplot as plt

@pytest.fixture
def sample_data():
    x_data = [1, 2, 3, 4, 5]
    y_data = [2, 4, 1, 3, 5]
    return x_data, y_data

def test_generate_scatter_plot_valid_data(sample_data, monkeypatch):
    x_data, y_data = sample_data
    x_label = "Test Label"
    
    def mock_show():
        pass

    monkeypatch.setattr(plt, "show", mock_show)
    
    try:
        generate_scatter_plot(x_data, y_data, x_label)
    except Exception as e:
        pytest.fail(f"Unexpected exception: {e}")

def test_generate_scatter_plot_empty_data():
    x_data = []
    y_data = []
    x_label = "Test Label"
    with pytest.raises(Exception):
        generate_scatter_plot(x_data, y_data, x_label)

def test_generate_scatter_plot_mismatched_data_lengths():
    x_data = [1, 2, 3]
    y_data = [1, 2]
    x_label = "Test Label"
    with pytest.raises(Exception):
        generate_scatter_plot(x_data, y_data, x_label)

def test_generate_scatter_plot_non_numeric_data():
    x_data = ['a', 'b', 'c']
    y_data = [1, 2, 3]
    x_label = "Test Label"
    with pytest.raises(TypeError):
        generate_scatter_plot(x_data, y_data, x_label)

def test_generate_scatter_plot_none_data():
    with pytest.raises(TypeError):
         generate_scatter_plot(None, None, None)

