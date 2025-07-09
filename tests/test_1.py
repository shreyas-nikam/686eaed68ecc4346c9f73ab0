import pytest
import pandas as pd
from definition_3586f295759d4bc89ce2f73ac6f041bc import simulate_symbolicai_workflow

@pytest.mark.parametrize("natural_language_task, complexity_level, expected_columns", [
    ("add two numbers", "Simple", ['stage', 'input', 'output']),
    ("sort a list", "Complex", ['stage', 'input', 'output']),
    ("translate to spanish", "Medium", ['stage', 'input', 'output']),
])
def test_simulate_symbolicai_workflow_valid_input(natural_language_task, complexity_level, expected_columns):
    result = simulate_symbolicai_workflow(natural_language_task, complexity_level)
    assert isinstance(result, pd.DataFrame)
    assert all(col in result.columns for col in expected_columns)

def test_simulate_symbolicai_workflow_empty_task():
    result = simulate_symbolicai_workflow("", "Simple")
    assert isinstance(result, pd.DataFrame)

def test_simulate_symbolicai_workflow_none_task():
    with pytest.raises(TypeError):
        simulate_symbolicai_workflow(None, "Simple")

def test_simulate_symbolicai_workflow_invalid_complexity():
    result = simulate_symbolicai_workflow("add two numbers", "Invalid")
    assert isinstance(result, pd.DataFrame)
