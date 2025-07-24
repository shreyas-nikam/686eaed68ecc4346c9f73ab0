
import streamlit as st
from application_pages.utils import generate_synthetic_data, plot_benchmark_comparison

def run_page2():
    st.header("Benchmark Category Comparison")
    st.markdown("""
    A bar chart comparing the aggregated VERTEX scores of selected models across the five benchmark categories: associative prediction, multi-modal binding, program synthesis, logic, and computational graphs [1, Section 7].
    """)

    num_data_points = 200
    simulated_df = generate_synthetic_data(num_data_points)

    benchmark_fig = plot_benchmark_comparison(simulated_df)
    if benchmark_fig:
        st.pyplot(benchmark_fig, use_container_width=True)
