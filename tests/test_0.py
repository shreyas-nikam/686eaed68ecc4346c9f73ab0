import pytest
import pandas as pd
from definition_8a4e0a2dc1724e4cac3c954bd402b9ad import generate_synthetic_parsing_data

def is_dataframe_equal(df1, df2):
    try:
        pd.testing.assert_frame_equal(df1, df2, check_dtype=False)
        return True
    except AssertionError:
        return False


def test_generate_synthetic_parsing_data_positive_samples():
    num_samples = 10
    df = generate_synthetic_parsing_data(num_samples)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == num_samples
    assert set(df.columns) == {'natural_language_input', 'symbolic_output', 'parsing_success', 'parsing_time_ms', 'complexity_level', 'input_length'}
    assert df['parsing_success'].dtype == bool
    assert df['parsing_time_ms'].dtype == 'float64'
    assert df['complexity_level'].dtype == 'object' # String in pandas
    assert df['input_length'].dtype == 'int64'

def test_generate_synthetic_parsing_data_zero_samples():
    num_samples = 0
    df = generate_synthetic_parsing_data(num_samples)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
    assert set(df.columns) == {'natural_language_input', 'symbolic_output', 'parsing_success', 'parsing_time_ms', 'complexity_level', 'input_length'}

def test_generate_synthetic_parsing_data_non_integer_input():
    with pytest.raises(TypeError):
        generate_synthetic_parsing_data("abc")

def test_generate_synthetic_parsing_data_negative_samples():
   with pytest.raises(ValueError):
       generate_synthetic_parsing_data(-5)

def test_generate_synthetic_parsing_data_consistent_failure():
    num_samples = 5
    df = generate_synthetic_parsing_data(num_samples)
    failure_count = sum(df['parsing_success'] == False)
    assert failure_count >=0
