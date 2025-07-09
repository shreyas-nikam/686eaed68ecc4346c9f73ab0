import pandas as pd
import numpy as np

def generate_synthetic_parsing_data(num_samples):
    """Generates synthetic data for LLM semantic parsing outcomes."""
    if not isinstance(num_samples, int):
        raise TypeError("num_samples must be an integer.")
    if num_samples < 0:
        raise ValueError("num_samples must be a non-negative integer.")

    data = {
        'natural_language_input': [f"Example query {i}" for i in range(num_samples)],
        'symbolic_output': [f"SELECT ... WHERE ... {i}" for i in range(num_samples)],
        'parsing_success': np.random.choice([True, False], size=num_samples),
        'parsing_time_ms': np.random.randint(10, 100, size=num_samples),
        'complexity_level': np.random.randint(1, 5, size=num_samples),
        'input_length': [len(f"Example query {i}") for i in range(num_samples)]
    }
    df = pd.DataFrame(data)
    return df

import pandas as pd
import time

def simulate_llm_parsing(user_input_query, selected_complexity, synthetic_data):
    """Simulates LLM parsing based on user input and synthetic data."""

    start_time = time.time()
    match = synthetic_data[
        (synthetic_data['natural_language_input'] == user_input_query) &
        (synthetic_data['complexity_level'] == selected_complexity)
    ]

    if not isinstance(selected_complexity, str):
        raise Exception("Complexity should be a string")

    if not match.empty:
        parsed_symbolic_expression = match['symbolic_output'].iloc[0]
        simulated_parsing_success = match['parsing_success'].iloc[0]
        simulated_parsing_time_ms = match['parsing_time_ms'].iloc[0]
    else:
        # Default behavior when no match is found
        parsed_symbolic_expression = "No direct match found. Simulated failed parsing."
        simulated_parsing_success = False
        simulated_parsing_time_ms = 50  # Default time

    end_time = time.time()
    elapsed_time_ms = (end_time - start_time) * 1000 #in milliseconds

    computational_graph_data = {
        'nodes': [],
        'edges': []
    }

    result = {
        'parsed_symbolic_expression': parsed_symbolic_expression,
        'simulated_parsing_time_ms': elapsed_time_ms,
        'simulated_parsing_success': simulated_parsing_success,
        'computational_graph_data': computational_graph_data
    }

    return result

import matplotlib.pyplot as plt
import networkx as nx

def generate_computational_graph_visualization(graph_data):
    """Visualizes a computational graph.

    Args:
        graph_data (dict/list): Graph data with nodes and edges.
    """

    if graph_data is None:
        raise TypeError("graph_data cannot be None")

    if not isinstance(graph_data, (dict, list)):
        raise TypeError("graph_data must be a dict or list")

    if isinstance(graph_data, dict):
        nodes = graph_data.get("nodes", [])
        edges = graph_data.get("edges", [])
    elif isinstance(graph_data, list):
        nodes = []
        edges = []

    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)

    pos = nx.spring_layout(graph)  # Layout the graph

    plt.figure(figsize=(8, 6))
    nx.draw(graph, pos, with_labels=True, node_size=1500, node_color="skyblue", font_size=10, font_weight="bold")
    plt.title("Computational Graph")
    plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_parsing_metrics(dataframe_metrics):
    """To visualize parsing metrics."""

    if dataframe_metrics.empty:
        raise ValueError("Cannot plot with an empty DataFrame!")

    try:
        # Ensure parsing_success is boolean
        dataframe_metrics['parsing_success'] = dataframe_metrics['parsing_success'].astype(bool)

        # Box plot of parsing time by complexity level
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='complexity_level', y='parsing_time_ms', data=dataframe_metrics)
        plt.title('Parsing Time vs. Complexity Level')
        plt.xlabel('Complexity Level')
        plt.ylabel('Parsing Time (ms)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        # Scatter plot of input length vs. parsing time, colored by parsing success
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='input_length', y='parsing_time_ms', hue='parsing_success', data=dataframe_metrics)
        plt.title('Input Length vs. Parsing Time (Colored by Success)')
        plt.xlabel('Input Length')
        plt.ylabel('Parsing Time (ms)')
        plt.tight_layout()
        plt.show()

        # Bar plot of parsing success rate by complexity level
        success_rate = dataframe_metrics.groupby('complexity_level')['parsing_success'].mean().reset_index()
        plt.figure(figsize=(10, 6))
        sns.barplot(x='complexity_level', y='parsing_success', data=success_rate)
        plt.title('Parsing Success Rate vs. Complexity Level')
        plt.xlabel('Complexity Level')
        plt.ylabel('Success Rate')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)  # Success rate is between 0 and 1
        plt.tight_layout()
        plt.show()

    except KeyError as e:
        raise KeyError(f"Required column missing: {e}")
    except ValueError as e:
        raise ValueError(f"Data type issue: {e}")
    except Exception as e:
        raise e