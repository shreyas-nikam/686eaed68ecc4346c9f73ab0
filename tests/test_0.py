import pytest
import pandas as pd
from definition_c362292890fa4ffe8de68104bc9f451b import generate_synthetic_parsing_data

def is_dataframe_valid(df):
    required_columns = ['natural_language_input', 'symbolic_output', 'parsing_success', 'parsing_time_ms', 'complexity_level', 'input_length']
    if not all(col in df.columns for col in required_columns):
        return False
    if df['parsing_success'].dtype != bool:
        return False
    if df['parsing_time_ms'].dtype not in ['int64', 'float64']:
        return False
    if df['input_length'].dtype not in ['int64', 'float64']:
        return False
    return True

@pytest.mark.parametrize("num_samples, expected_valid_dataframe", [
    (5, True),
    (0, True),
    (10, True),
    
])
def test_generate_synthetic_parsing_data_valid_dataframe(num_samples, expected_valid_dataframe):
    df = generate_synthetic_parsing_data(num_samples)
    assert isinstance(df, pd.DataFrame)
    assert is_dataframe_valid(df) == expected_valid_dataframe
    
@pytest.mark.parametrize("num_samples", [
    (5),
    (0),
    (10),
    
])
def test_generate_synthetic_parsing_data_correct_length(num_samples):
    df = generate_synthetic_parsing_data(num_samples)
    assert len(df) == num_samples

def test_generate_synthetic_parsing_data_negative_samples():
    with pytest.raises(ValueError):
        generate_synthetic_parsing_data(-5)

def test_generate_synthetic_parsing_data_empty_strings():
    df = generate_synthetic_parsing_data(5)
    assert not any(df['natural_language_input'] == "")
    
def test_generate_synthetic_parsing_data_non_numeric_input():
    with pytest.raises(TypeError):
        generate_synthetic_parsing_data("abc")
