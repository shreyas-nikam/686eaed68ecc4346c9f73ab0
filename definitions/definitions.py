import pandas as pd
import numpy as np

def generate_synthetic_data(num_rows):
    """Generates synthetic LLM performance data."""

    if num_rows < 0:
        raise ValueError("Number of rows must be non-negative.")

    if num_rows == 0:
        return pd.DataFrame({
            'Model': [],
            'BenchmarkCategory': [],
            'TaskStep': [],
            'ContextSize': [],
            'FewShotExamples': [],
            'SimulatedMMD2': [],
            'VERTEXScore': []
        })

    models = ['ModelA', 'ModelB', 'ModelC']
    categories = ['Reasoning', 'Coding', 'Translation']

    data = {
        'Model': np.random.choice(models, num_rows),
        'BenchmarkCategory': np.random.choice(categories, num_rows),
        'TaskStep': np.random.randint(1, 11, num_rows),
        'ContextSize': np.random.choice([128, 256, 512, 1024], num_rows),
        'FewShotExamples': np.random.choice([0, 1, 5, 10], num_rows),
        'SimulatedMMD2': np.random.rand(num_rows),
        'VERTEXScore': np.random.rand(num_rows)
    }

    df = pd.DataFrame(data)
    return df

import random

def simulate_vertex_score_single(model_name, context_size, few_shot_examples, task_step, benchmark_category):
    """Simulates VERTEX score."""
    # This is a simplified simulation; replace with a more sophisticated model.
    # The simulation considers model_name, context_size, few_shot_examples, task_step, and benchmark_category.
    # The task_step introduces a trend (higher score for later steps).

    base_score = 0.5  # A base score

    # Adjust for model name (higher score for "better" models)
    if "GPT-4 Turbo" in model_name:
        base_score += 0.2
    elif "LLaMA3-Chat" in model_name:
        base_score += 0.1
    elif "Mistral" in model_name:
        base_score += 0.05

    # Adjust for context size
    base_score += context_size / 4096 * 0.05 #Scale context effect

    # Adjust for few-shot examples
    base_score += few_shot_examples * 0.01 #Scale few shot effect

    # Adjust for task step (trend)
    base_score += task_step * 0.02 #Scale Task Step effect

    # Adjust for benchmark category (slight variation)
    if benchmark_category == "Logic":
        base_score -= 0.03
    elif benchmark_category == "Program Synthesis":
        base_score += 0.02


    # Ensure the score stays within [0, 1]
    final_score = max(0.0, min(1.0, base_score + random.uniform(-0.1, 0.1)))
    return float(final_score)

import pandas as pd
import plotly.express as px

def plot_vertex_trend(simulated_df, selected_model, selected_context, selected_few_shot):
    """Plots the VERTEX score trend over task steps."""

    filtered_df = simulated_df[
        (simulated_df['Model'] == selected_model) &
        (simulated_df['ContextSize'] == selected_context) &
        (simulated_df['FewShotExamples'] == selected_few_shot)
    ]

    if filtered_df.empty:
        return None

    fig = px.line(filtered_df, x='TaskStep', y='VERTEXScore',
                  title=f'VERTEX Score Trend for {selected_model}, Context: {selected_context}, Few-Shot: {selected_few_shot}')
    fig.update_layout(xaxis_title='Task Step', yaxis_title='VERTEX Score')
    return fig

import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

def plot_benchmark_comparison(simulated_df):
    """Compares aggregated VERTEX scores across benchmark categories."""

    if not isinstance(simulated_df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if simulated_df.empty:
        return  # Handle empty DataFrame gracefully (no plot generated)

    required_columns = ['Model', 'BenchmarkCategory', 'TaskStep', 'ContextSize', 'FewShotExamples', 'SimulatedMMD2', 'VERTEXScore']
    for col in required_columns:
        if col not in simulated_df.columns:
            raise KeyError(f"DataFrame must contain column '{col}'.")

    aggregated_data = simulated_df.groupby('BenchmarkCategory')['VERTEXScore'].mean().sort_values()

    plt.figure(figsize=(10, 6))  # Adjust figure size for better readability
    aggregated_data.plot(kind='bar', color='skyblue')
    plt.title('Average VERTEX Score by Benchmark Category')
    plt.xlabel('Benchmark Category')
    plt.ylabel('Average VERTEX Score')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for readability
    plt.tight_layout()  # Adjust layout to prevent labels from overlapping

    # Convert plot to PNG image in memory
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
    img_data = base64.b64encode(img_buf.read()).decode('utf-8')
    plt.close()

    return # Or optionally return img_data for web display

import pandas as pd
import matplotlib.pyplot as plt

def plot_performance_scatter(simulated_df, x_axis_param):
    """Plots the relationship between x_axis_param and VERTEXScore."""
    try:
        plt.figure(figsize=(8, 6))
        plt.scatter(simulated_df[x_axis_param], simulated_df['VERTEXScore'])
        plt.xlabel(x_axis_param)
        plt.ylabel('VERTEXScore')
        plt.title(f'VERTEXScore vs {x_axis_param}')
        plt.grid(True)
        plt.show()
    except KeyError as e:
        raise KeyError(e)
    except ValueError as e:
        raise ValueError(e)