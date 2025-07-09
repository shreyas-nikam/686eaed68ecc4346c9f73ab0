import pytest
import pandas as pd
from definition_782c5f58bcb04926a08057cbb79f91fa import generate_synthetic_parsing_data

def is_dataframe_and_has_columns(df):
    if not isinstance(df, pd.DataFrame):
        return False
    required_columns = ['natural_language_input', 'symbolic_output', 'parsing_success', 'parsing_time_ms', 'complexity_level', 'input_length']
    return all(col in df.columns for col in required_columns)


def test_generate_synthetic_parsing_data_positive_samples():
    num_samples = 50
    df = generate_synthetic_parsing_data(num_samples)
    assert is_dataframe_and_has_columns(df)
    assert len(df) == num_samples
    assert df['parsing_success'].dtype == bool
    assert df['parsing_time_ms'].dtype == float
    assert df['input_length'].dtype == int
    assert all(level in ['Simple', 'Medium', 'Complex', 'Challenging'] for level in df['complexity_level'])


def test_generate_synthetic_parsing_data_zero_samples():
    num_samples = 0
    df = generate_synthetic_parsing_data(num_samples)
    assert is_dataframe_and_has_columns(df)
    assert len(df) == 0


def test_generate_synthetic_parsing_data_large_number_of_samples():
    num_samples = 1000
    df = generate_synthetic_parsing_data(num_samples)
    assert is_dataframe_and_has_columns(df)
    assert len(df) == num_samples


def test_generate_synthetic_parsing_data_negative_input():
    with pytest.raises(ValueError):
        generate_synthetic_parsing_data(-10)

def test_generate_synthetic_parsing_data_invalid_input_type():
    with pytest.raises(TypeError):
        generate_synthetic_parsing_data("invalid")
