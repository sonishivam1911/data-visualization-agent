# frontend.py

import streamlit as st
import pandas as pd
from backend import generate_visualization

def main():
    st.title("Interactive Data Visualization with Natural Language")

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file:
        df = load_data(uploaded_file)
        st.write("### Data Preview")
        st.dataframe(df.head())

        user_input = st.text_input("Describe the visualization you want to create:")
        if user_input:
            with st.spinner("Generating visualization..."):
                fig, code = generate_visualization(user_input, df)
                if fig:
                    st.pyplot(fig)
                    with st.expander("Show generated code"):
                        st.code(code, language='python')
                else:
                    st.error("Could not generate the visualization. Please try refining your request.")

def load_data(uploaded_file) -> pd.DataFrame:
    """
    Loads the CSV data into a pandas DataFrame.
    """
    df = pd.read_csv(uploaded_file)
    return df

if __name__ == "__main__":
    main()
