"""
Streamlit Dashboard for Credit Risk Probability Model
Bati Bank Credit Scoring System
"""

from PIL import Image
import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import json

# Page configuration
icon = Image.open("assets/Bati.png")
st.set_page_config(
    page_title="Bati Bank - Credit Risk Scoring", page_icon=icon, layout="wide"
)

# Styling
st.markdown(
    """
    <style>
    .main { padding: 0rem 1rem; }
    h1 { color: #1f77b4; text-align: center; }
    </style>
""",
    unsafe_allow_html=True,
)

# ==================== API Configuration ====================
API_BASE_URL = "http://localhost:8000"


def check_api_health() -> bool:
    """Check if the API is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False
