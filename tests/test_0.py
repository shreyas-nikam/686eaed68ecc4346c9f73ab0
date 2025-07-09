import pytest
import pandas as pd
from definition_b56cf18d4e254d969cff48196a7f36f6 import generate_synthetic_workflow_data

@pytest.fixture
def sample_task_types():
    return ['arithmetic', 'logical', 'query']

def test_generate_synthetic_workflow_data_positive(sample_task_types):
    num_workflows = 3
    max_stages = 5
    df = generate_synthetic_workflow_data(num_workflows, max_stages, sample_task_types)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == num_workflows
    assert 'workflow_id' in df.columns
    assert 'stage_name' in df.columns
    assert 'task_type' in df.columns

def test_generate_synthetic_workflow_data_zero_workflows(sample_task_types):
    num_workflows = 0
    max_stages = 5
    df = generate_synthetic_workflow_data(num_workflows, max_stages, sample_task_types)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0

def test_generate_synthetic_workflow_data_large_numbers(sample_task_types):
    num_workflows = 2
    max_stages = 10
    df = generate_synthetic_workflow_data(num_workflows, max_stages, sample_task_types)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == num_workflows

def test_generate_synthetic_workflow_data_invalid_task_types():
    num_workflows = 1
    max_stages = 2
    with pytest.raises(TypeError):
        generate_synthetic_workflow_data(num_workflows, max_stages, 123)

def test_generate_synthetic_workflow_data_empty_task_types():
    num_workflows = 1
    max_stages = 2
    task_types = []
    df = generate_synthetic_workflow_data(num_workflows, max_stages, task_types)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == num_workflows
    assert all(df['task_type'] == None)
