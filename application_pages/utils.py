
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import random
import streamlit as st

@st.cache_data
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

    models = ['GPT-4 Turbo', 'LLaMA3-Chat', 'Mistral 7B'] # Use models from spec for simulation
    categories = ['Associative Prediction', 'Multi-modal Binding', 'Program Synthesis', 'Logic', 'Computational Graphs'] # Renamed for consistent display
    context_sizes = [500, 1000, 2000, 3000, 4096] # Range from spec
    few_shot_examples_counts = [0, 1, 2, 5, 10] # Range from spec
    task_steps = list(range(1, 11)) # 1 to 10 steps

    data = []
    for _ in range(num_rows):
        model = np.random.choice(models)
        category = np.random.choice(categories)
        task_step = np.random.choice(task_steps)
        context_size = np.random.choice(context_sizes)
        few_shot_examples = np.random.choice(few_shot_examples_counts)
        simulated_mmd2 = np.random.rand() # Placeholder

        # Use the simulate_vertex_score_single logic for more meaningful VERTEXScore
        vertex_score = simulate_vertex_score_single(
            model_name=model,
            context_size=context_size,
            few_shot_examples=few_shot_examples,
            task_step=task_step,
            benchmark_category=category
        )
        data.append({
            'Model': model,
            'BenchmarkCategory': category,
            'TaskStep': task_step,
            'ContextSize': context_size,
            'FewShotExamples': few_shot_examples,
            'SimulatedMMD2': simulated_mmd2,
            'VERTEXScore': vertex_score
        })
    df = pd.DataFrame(data)
    # Ensure VERTEXScore is within [0, 1] as per the formula implication, even if simulate_vertex_score_single does it.
    df['VERTEXScore'] = df['VERTEXScore'].apply(lambda x: max(0.0, min(1.0, x)))
    return df

def simulate_vertex_score_single(model_name, context_size, few_shot_examples, task_step, benchmark_category):
    """Simulates VERTEX score."""
    base_score = 0.5

    if "GPT-4 Turbo" in model_name:
        base_score += 0.2
    elif "LLaMA3-Chat" in model_name:
        base_score += 0.1
    elif "Mistral" in model_name:
        base_score += 0.05

    base_score += context_size / 4096 * 0.05
    base_score += few_shot_examples * 0.01
    base_score += task_step * 0.02

    if benchmark_category == "Logic":
        base_score -= 0.03
    elif benchmark_category == "Program Synthesis":
        base_score += 0.02
    elif benchmark_category == "Associative Prediction":
        base_score += 0.01
    elif benchmark_category == "Multi-modal Binding":
        base_score -= 0.02
    elif benchmark_category == "Computational Graphs":
        base_score += 0.03

    final_score = max(0.0, min(1.0, base_score + random.uniform(-0.1, 0.1)))
    return float(final_score)

def plot_vertex_trend(simulated_df, selected_model, selected_context, selected_few_shot):
    """Plots the VERTEX score trend over task steps."""
    # To ensure a trend is visible and reproducible for the selected parameters,
    # generate a specific DataFrame for the exact selected combination of parameters
    # over all task steps (1 to 10).
    specific_plot_data = []
    for step in range(1, 11): # 10 steps
        score = simulate_vertex_score_single(selected_model, selected_context, selected_few_shot, step, 'Program Synthesis') # Using an arbitrary but consistent category for trend
        specific_plot_data.append({
            'Model': selected_model,
            'ContextSize': selected_context,
            'FewShotExamples': selected_few_shot,
            'TaskStep': step,
            'VERTEXScore': score
        })
    plot_df_for_selected_params = pd.DataFrame(specific_plot_data)

    if plot_df_for_selected_params.empty:
        st.info(f"No trend data generated for Model: {selected_model}, Context: {selected_context}, Few-Shot: {selected_few_shot}.")
        return None

    fig = px.line(plot_df_for_selected_params, x='TaskStep', y='VERTEXScore',
                  title=f'VERTEX Score Trend for {selected_model}, Context: {selected_context}, Few-Shot: {selected_few_shot}',
                  color_discrete_sequence=px.colors.qualitative.Plotly) # Ensure color-blind friendly palette
    fig.update_layout(xaxis_title='Task Step', yaxis_title='VERTEX Score', font_size=12)
    fig.update_traces(mode='lines+markers') # Show markers on the line
    return fig

def plot_benchmark_comparison(simulated_df):
    """Compares aggregated VERTEX scores across benchmark categories."""
    if simulated_df.empty:
        st.info("DataFrame is empty. No benchmark comparison plot generated.")
        return None

    aggregated_data = simulated_df.groupby('BenchmarkCategory')['VERTEXScore'].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6)) # Create a Matplotlib figure and axes
    aggregated_data.plot(kind='bar', ax=ax, color=plt.cm.Paired.colors) # Use a color-blind friendly colormap
    ax.set_title('Average VERTEX Score by Benchmark Category', fontsize=14)
    ax.set_xlabel('Benchmark Category', fontsize=12)
    ax.set_ylabel('Average VERTEX Score', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    return fig

def plot_performance_scatter(simulated_df, x_axis_variable):
    """Plots a scatter plot of VERTEX score vs. context size or few-shot examples."""
    if x_axis_variable not in ['ContextSize', 'FewShotExamples']:
        raise ValueError("x_axis_variable must be 'ContextSize' or 'FewShotExamples'.")

    if simulated_df.empty:
        st.info(f"DataFrame is empty. No scatter plot for {x_axis_variable} generated.")
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(simulated_df[x_axis_variable], simulated_df['VERTEXScore'], alpha=0.6, color='teal')
    ax.set_title(f'VERTEX Score vs. {x_axis_variable}', fontsize=14)
    ax.set_xlabel(x_axis_variable, fontsize=12)
    ax.set_ylabel('VERTEX Score', fontsize=12)
    ax.grid(True)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    return fig
