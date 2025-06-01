import sys
import os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
import streamlit as st

st.set_page_config(page_title="Customer360AI", layout="wide")

st.title("Welcome to Customer360AI")
st.markdown("""
Upload customer data and explore modules:
- EDA Engine
- Segmentation
- Churn Prediction
- Forecasting
- Insights
""")