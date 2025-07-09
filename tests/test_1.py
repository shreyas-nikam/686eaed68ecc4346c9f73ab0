import pytest
from definition_13a500887d59470a81f9f253fe043ae9 import simulate_vertex_score_single

@pytest.mark.parametrize("model_name, context_size, few_shot_examples, task_step, benchmark_category, expected", [
    ("GPT-4 Turbo", 1024, 5, 3, "Associative Prediction", (0.0, 1.0)),
    ("LLaMA3-Chat", 512, 0, 1, "Program Synthesis", (0.0, 1.0)),
    ("Mistral 7B", 2048, 10, 5, "Logic", (0.0, 1.0)),
    ("GPT-4 Turbo", 0, 5, 3, "Associative Prediction", (0.0, 1.0)), # Edge case: zero context size
    ("GPT-4 Turbo", 1024, -1, 3, "Associative Prediction", (0.0, 1.0)), # Edge case: negative few shot examples

])
def test_simulate_vertex_score_single(model_name, context_size, few_shot_examples, task_step, benchmark_category, expected):
    result = simulate_vertex_score_single(model_name, context_size, few_shot_examples, task_step, benchmark_category)
    assert isinstance(result, float)
    assert expected[0] <= result <= expected[1]
