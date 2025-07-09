import pytest
from definition_f3b894299b1b4467a79e1d3947986174 import generate_computational_graph_visualization

@pytest.mark.parametrize("graph_data, expected_exception", [
    ({"nodes": ["A", "B"], "edges": [("A", "B")]}, None),  # Valid graph data
    ([], None), # Empty list graph_data
    ({}, None),  # Empty dict graph_data
    (None, TypeError),  # None input
    (123, TypeError), # Invalid data type
])
def test_generate_computational_graph_visualization(graph_data, expected_exception):
    if expected_exception:
        with pytest.raises(expected_exception):
            generate_computational_graph_visualization(graph_data)
    else:
        try:
            generate_computational_graph_visualization(graph_data)
        except Exception as e:
            pytest.fail(f"Unexpected exception: {e}")