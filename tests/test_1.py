import pytest
from definition_4bf00071351d43a9947f92807ff20dfc import simulate_llm_parsing
import pandas as pd

@pytest.fixture
def synthetic_data():
    data = {
        'natural_language_input': ['2 + 2', 'Is a cat an animal?', 'What is the capital of France?', 'Define quantum entanglement', 'Marvins is a cat given he has four paws and likes to meow when petted?'],
        'symbolic_output': ['4', 'True', 'Paris', 'Complex definition', 'True'],
        'parsing_success': [True, True, True, True, True],
        'parsing_time_ms': [10, 15, 20, 25, 30],
        'complexity_level': ['Simple', 'Medium', 'Medium', 'Complex', 'Complex'],
        'input_length': [5, 20, 28, 25, 78]
    }
    return pd.DataFrame(data)

def test_simulate_llm_parsing_simple(synthetic_data):
    result = simulate_llm_parsing('2 + 2', 'Simple', synthetic_data)
    assert result['parsed_symbolic_expression'] == '4'
    assert result['simulated_parsing_success'] == True

def test_simulate_llm_parsing_complex(synthetic_data):
    result = simulate_llm_parsing('Marvins is a cat given he has four paws and likes to meow when petted?', 'Complex', synthetic_data)
    assert result['parsed_symbolic_expression'] == 'True'
    assert result['simulated_parsing_success'] == True

def test_simulate_llm_parsing_no_match(synthetic_data):
    result = simulate_llm_parsing('unknown query', 'Medium', synthetic_data)
    #  Implementation details not known, but should handle no exact match.
    assert isinstance(result, dict)
    
def test_simulate_llm_parsing_empty_input(synthetic_data):
    with pytest.raises(TypeError):
        simulate_llm_parsing(None, 'Simple', synthetic_data)

def test_simulate_llm_parsing_invalid_complexity(synthetic_data):
     with pytest.raises(TypeError):
        simulate_llm_parsing('2+2', 123, synthetic_data) # complexity needs to be a string
