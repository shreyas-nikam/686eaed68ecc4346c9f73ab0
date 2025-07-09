import numpy as np
import pandas as pd

def simulate_vertex_score(model_name, context_size, few_shot_examples, benchmark_category, task_step):
    """Simulates vertex_score based on input parameters."""

    # Simulate a base score, making it slightly dependent on model and context
    base_score = 0.5 + (context_size / 10000) * 0.1  # Context influence
    if "GPT-4" in model_name:
        base_score += 0.2  # GPT-4 boost
    elif "LLaMA" in model_name:
        base_score += 0.1
    elif "Mistral" in model_name:
        base_score += 0.05
    base_score = max(0, min(base_score, 1))  # Ensure within [0, 1]

    # Adjust score based on few-shot examples
    few_shot_effect = few_shot_examples * 0.05
    base_score += few_shot_effect
    base_score = max(0, min(base_score, 1))

    # Introduce randomness
    random_variation = np.random.normal(0, 0.05)
    base_score += random_variation
    base_score = max(0, min(base_score, 1))

    return float(base_score)

import matplotlib.pyplot as plt
import numpy as np

def generate_trend_plot(model, context_size, few_shot_examples):
    """Generates a trend plot of VERTEX score vs task step."""

    if model is None:
        raise Exception("Model cannot be None")

    num_steps = 10  # Define the number of task steps

    # Generate dummy data for vertex scores (replace with actual model output)
    vertex_scores = np.random.rand(num_steps)
    task_steps = np.arange(1, num_steps + 1)

    # Create the plot
    fig, ax = plt.subplots()
    ax.plot(task_steps, vertex_scores)

    # Set labels and title
    ax.set_xlabel("Task Step")
    ax.set_ylabel("VERTEX Score")
    ax.set_title(f"VERTEX Score Trend (Context Size: {context_size}, Few-Shot: {few_shot_examples})")

    # Return the figure
    return fig

import pandas as pd
import matplotlib.pyplot as plt

def generate_benchmark_comparison_plot(data):
    """Generates a bar chart comparing aggregated VERTEX scores of models across benchmark categories."""

    if not isinstance(data, pd.DataFrame):
        raise AttributeError("Input must be a Pandas DataFrame.")

    if data.empty:
        print("Warning: Input DataFrame is empty. No plot will be generated.")
        return

    required_columns = ['model_name', 'benchmark_category', 'vertex_score']
    if not all(col in data.columns for col in required_columns):
        raise KeyError("DataFrame must contain 'model_name', 'benchmark_category', and 'vertex_score' columns.")

    aggregated_data = data.groupby(['model_name', 'benchmark_category'])['vertex_score'].mean().unstack()

    ax = aggregated_data.plot(kind='bar', figsize=(10, 6))
    ax.set_title('Benchmark Comparison')
    ax.set_xlabel('Benchmark Category')
    ax.set_ylabel('Average Vertex Score')
    ax.legend(title='Model')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt

def generate_scatter_plot(x_data, y_data, x_label):
    """Generates a scatter plot."""

    if not all(isinstance(item, (int, float)) for item in x_data) or not all(isinstance(item, (int, float)) for item in y_data):
        raise TypeError("x_data and y_data must contain numeric values.")
    
    if x_data is None or y_data is None or x_label is None:
        raise TypeError("Arguments cannot be None")
    
    if not x_data or not y_data:
        raise Exception("x_data and y_data cannot be empty.")
    
    if len(x_data) != len(y_data):
        raise Exception("x_data and y_data must have the same length.")

    plt.scatter(x_data, y_data)
    plt.xlabel(x_label)
    plt.ylabel("VERTEX Score")
    plt.title(f"Scatter Plot of VERTEX Score vs {x_label}")
    plt.show()