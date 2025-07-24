
import streamlit as st
from application_pages.utils import generate_synthetic_data, plot_vertex_trend

def run_page1():
    st.header("VERTEX Score Trend")
    st.markdown("""
    A line plot showing the simulated VERTEX score over a series of sequential task steps for a chosen model and parameters.
    The VERTEX score, \$s\$, is based on a similarity measure, which could be represented as:
    $$\displaystyle s_{\text{gen, ref}} := \frac{1}{T_{RV}} \int_{0}^{T_{RV}} \min{\big(}\max{\big(}0, 1-\frac{1}{MMD^2}(\mu_{g_{t}}, \mu_{r_{t}})-z_{\text{rand}}{\big)}, 1{\big)} dt$$
    Where \$MMD^2\$ represents the Maximum Mean Discrepancy squared, \$\mu_g\$ and \$\mu_r\$ are mean embeddings of generated and reference data, and \$z_{\text{rand}}\$ is a baseline for randomness [1, Section 2, Equation 4]. For simplicity, the lab will use a simplified linear transformation of simulated \$MMD^2\$ to derive the VERTEX score.
    """)

    num_data_points = 200
    simulated_df = generate_synthetic_data(num_data_points)

    selected_model = st.selectbox(
        "Select LLM Model",
        simulated_df['Model'].unique(),
        help="Choose a synthetic LLM model to analyze its simulated performance."
    )

    selected_context = st.slider(
        "Simulated Context Size (tokens)",
        min_value=500,
        max_value=4096,
        value=2000,
        step=100,
        help="Adjust the simulated context window size. Larger contexts typically allow LLMs to process more information."
    )

    selected_few_shot = st.slider(
        "Number of Few-Shot Examples",
        min_value=0,
        max_value=10,
        value=5,
        step=1,
        help="Set the number of simulated few-shot examples provided. More examples generally improve in-context learning."
    )

    trend_fig = plot_vertex_trend(simulated_df, selected_model, selected_context, selected_few_shot)
    if trend_fig:
        st.plotly_chart(trend_fig, use_container_width=True)
