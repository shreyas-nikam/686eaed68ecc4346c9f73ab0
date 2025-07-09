import pandas as pd
import numpy as np

def generate_synthetic_parsing_data(num_samples):
    """Generates a synthetic dataset of parsing data."""

    if not isinstance(num_samples, int):
        raise TypeError("num_samples must be an integer.")
    if num_samples < 0:
        raise ValueError("num_samples must be non-negative.")

    data = []
    for _ in range(num_samples):
        input_length = np.random.randint(5, 20)
        natural_language_input = " ".join([f"word_{i}" for i in range(input_length)])
        symbolic_output = f"expression_{np.random.randint(1, 10)}"
        parsing_success = np.random.choice([True, False])
        parsing_time_ms = np.random.rand() * 100  # Up to 100 ms
        complexity_level = np.random.choice(["simple", "medium", "complex"])

        data.append([natural_language_input, symbolic_output, parsing_success, parsing_time_ms, complexity_level, input_length])

    df = pd.DataFrame(data, columns=['natural_language_input', 'symbolic_output', 'parsing_success', 'parsing_time_ms', 'complexity_level', 'input_length'])
    df['parsing_success'] = df['parsing_success'].astype(bool)
    df['parsing_time_ms'] = df['parsing_time_ms'].astype('float64')
    df['complexity_level'] = df['complexity_level'].astype('object')
    df['input_length'] = df['input_length'].astype('int64')
    return df

import pandas as pd
import time

def simulate_llm_parsing(user_input_query, selected_complexity, synthetic_data):
    """Simulates LLM parsing based on input and data."""

    if not isinstance(user_input_query, str):
        raise TypeError("User input query must be a string.")

    if not isinstance(selected_complexity, str):
        raise TypeError("Selected complexity must be a string.")
    
    start_time = time.time()

    # Exact match
    match = synthetic_data[
        (synthetic_data['natural_language_input'] == user_input_query) &
        (synthetic_data['complexity_level'] == selected_complexity)
    ]

    if not match.empty:
        parsed_symbolic_expression = match['symbolic_output'].iloc[0]
        simulated_parsing_success = match['parsing_success'].iloc[0]
        simulated_parsing_time_ms = match['parsing_time_ms'].iloc[0]
    else:
        # Fallback: find the closest match based on the input query
        match = synthetic_data[synthetic_data['natural_language_input'] == user_input_query]

        if not match.empty:
            parsed_symbolic_expression = match['symbolic_output'].iloc[0]
            simulated_parsing_success = match['parsing_success'].iloc[0]
            simulated_parsing_time_ms = match['parsing_time_ms'].iloc[0]
        else:
             # No match found
            parsed_symbolic_expression = None
            simulated_parsing_success = False
            simulated_parsing_time_ms = 0
            
    end_time = time.time()
    if parsed_symbolic_expression is None and simulated_parsing_success is False and simulated_parsing_time_ms == 0:
        simulated_parsing_time_ms = (end_time - start_time) * 1000
        return {
                'parsed_symbolic_expression': 'No match found',
                'simulated_parsing_time_ms': simulated_parsing_time_ms,
                'simulated_parsing_success': False,
                'computational_graph_data': {}
            }


    end_time = time.time()
    simulated_parsing_time_ms = (end_time - start_time) * 1000

    return {
        'parsed_symbolic_expression': parsed_symbolic_expression,
        'simulated_parsing_time_ms': simulated_parsing_time_ms,
        'simulated_parsing_success': simulated_parsing_success,
        'computational_graph_data': {}
    }

import matplotlib.pyplot as plt
import networkx as nx

def generate_computational_graph_visualization(graph_data):
    """To visually represent the simplified computational graph."""
    if graph_data is None:
        raise TypeError("Graph data cannot be None.")

    if not isinstance(graph_data, dict):
        raise AttributeError("Graph data must be a dictionary.")

    if not graph_data:
        return  # Handle empty graph gracefully

    try:
        nodes = graph_data.get("nodes")
        edges = graph_data.get("edges")

        if nodes is None or edges is None:
            raise ValueError("Graph data must contain 'nodes' and 'edges' keys.")

        graph = nx.DiGraph()

        if nodes:
            graph.add_nodes_from(nodes)
        if edges:
            graph.add_edges_from(edges)

        pos = nx.spring_layout(graph)  # You can choose different layouts

        nx.draw(graph, pos, with_labels=True, node_size=1500, node_color="skyblue", font_size=10, font_weight="bold")
        plt.title("Computational Graph")
        plt.show()  # Or save the plot: plt.savefig("computational_graph.png")
    except Exception as e:
        raise AttributeError(f"Invalid graph data format: {e}")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_parsing_metrics(dataframe_metrics):
    """To visualize relationships and comparisons of the synthetic parsing metrics.

    Args:
        dataframe_metrics (DataFrame): The full synthetic dataset.

    Output:
        Multiple graphical plots (e.g., Matplotlib/Seaborn figures).
    """
    if not isinstance(dataframe_metrics, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if dataframe_metrics.empty:
        return

    required_columns = ['parsing_success', 'parsing_time_ms', 'complexity_level', 'input_length']
    for col in required_columns:
        if col not in dataframe_metrics.columns:
            raise KeyError(f"DataFrame must contain column: {col}")

    # Ensure parsing_success is boolean
    if dataframe_metrics['parsing_success'].dtype != bool:
        dataframe_metrics['parsing_success'] = dataframe_metrics['parsing_success'].astype(bool)

    # Plot 1: Parsing success rate by complexity level
    plt.figure(figsize=(8, 6))
    sns.countplot(x='complexity_level', hue='parsing_success', data=dataframe_metrics)
    plt.title('Parsing Success Rate by Complexity Level')
    plt.xlabel('Complexity Level')
    plt.ylabel('Number of Samples')
    plt.show()

    # Plot 2: Parsing time vs. input length
    plt.figure(figsize=(8, 6))
    plt.scatter(dataframe_metrics['input_length'], dataframe_metrics['parsing_time_ms'])
    plt.title('Parsing Time vs. Input Length')
    plt.xlabel('Input Length')
    plt.ylabel('Parsing Time (ms)')
    plt.show()

    # Plot 3: Boxplot of parsing time by complexity level
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='complexity_level', y='parsing_time_ms', data=dataframe_metrics)
    plt.title('Parsing Time by Complexity Level')
    plt.xlabel('Complexity Level')
    plt.ylabel('Parsing Time (ms)')
    plt.show()