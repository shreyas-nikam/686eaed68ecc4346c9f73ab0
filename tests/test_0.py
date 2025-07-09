import pytest
import pandas as pd
from definition_fc267c1f4fb248a7bce054968ca76f82 import generate_synthetic_data

def is_dataframe(obj):
    try:
        import pandas as pd
        return isinstance(obj, pd.DataFrame)
    except ImportError:
        return False

@pytest.mark.parametrize("num_rows", [
    0,
    1,
    10,
])
def test_generate_synthetic_data_row_count(num_rows):
    df = generate_synthetic_data(num_rows)
    if num_rows > 0:
        assert is_dataframe(df)
        assert len(df) == num_rows
    else:
        assert is_dataframe(df)
        assert len(df) == 0


def test_generate_synthetic_data_valid_dataframe():
    df = generate_synthetic_data(10)
    required_columns = ['Model', 'BenchmarkCategory', 'TaskStep', 'ContextSize', 'FewShotExamples', 'SimulatedMMD2', 'VERTEXScore']
    assert all(col in df.columns for col in required_columns)

    # Check data types (basic check, more rigorous checks can be added)
    assert df['TaskStep'].dtype == 'int64'
    assert df['ContextSize'].dtype == 'int64'
    assert df['FewShotExamples'].dtype == 'int64'
    assert df['SimulatedMMD2'].dtype == 'float64'
    assert df['VERTEXScore'].dtype == 'float64'


def test_generate_synthetic_data_vertex_score_range():
    df = generate_synthetic_data(10)
    assert all(0 <= score <= 1 for score in df['VERTEXScore'])

def test_generate_synthetic_data_num_rows_negative():
    with pytest.raises(ValueError):
         generate_synthetic_data(-1)
