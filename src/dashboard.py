"""
Streamlit Dashboard for Credit Risk Probability Model
Bati Bank Credit Scoring System
"""

import os
from PIL import Image
import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import json

# Page configuration
icon = Image.open("src/assets/Bati.png")
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
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


def check_api_health() -> bool:
    """Check if the API is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def call_api_predict(customer_id: str, features: dict, is_raw: bool = True) -> dict:
    """Call the FastAPI endpoint for credit scoring."""
    try:
        url = f"{API_BASE_URL}/predict"
        params = {"customer_id": customer_id, "is_raw": is_raw}
        response = requests.post(url, json=features, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to API. Ensure the FastAPI server is running on port 8000.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"API Error: {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return None


# ==================== Main Dashboard ====================

# Title
st.title("Bati Bank - Credit Risk Scoring")

# Check API status
api_status = check_api_health()
if api_status:
    st.success("API Connected")
else:
    st.error("API Disconnected - Please start the FastAPI server on port 8000")

st.markdown("---")

# Input Method Selection
input_method = st.radio("Input Method:", ["Form", "JSON"], horizontal=True)

if input_method == "Form":
    col1, col2 = st.columns(2)
    
    with col1:
        customer_id = st.text_input("Customer ID", value="CUST-12345")
        recency = st.number_input("Recency (days since last transaction)", min_value=0, value=30, step=1)
        frequency = st.number_input("Frequency (number of transactions)", min_value=0, value=15, step=1)
    
    with col2:
        monetary_total = st.number_input("Monetary Total", min_value=0.0, value=50000.0, step=100.0)
        monetary_mean = st.number_input("Monetary Mean", min_value=0.0, value=2500.0, step=10.0)
        monetary_std = st.number_input("Monetary Std", min_value=0.0, value=750.0, step=5.0)
    
    features = {
        "Recency": float(recency),
        "Frequency": float(frequency),
        "Monetary_Total": float(monetary_total),
        "Monetary_Mean": float(monetary_mean),
        "Monetary_Std": float(monetary_std)
    }

else:  # JSON Input
    customer_id = st.text_input("Customer ID", value="CUST-12345")
    json_input = st.text_area(
        "Enter JSON formatted customer data:",
        value=json.dumps({
            "Recency": 30,
            "Frequency": 15,
            "Monetary_Total": 50000.0,
            "Monetary_Mean": 2500.0,
            "Monetary_Std": 750.0
        }, indent=2),
        height=200
    )
    try:
        features = json.loads(json_input)
    except json.JSONDecodeError:
        st.error("Invalid JSON format")
        features = None

# Score Button
st.markdown("---")
score_button = st.button("Calculate Credit Score", use_container_width=True, type="primary")

# Display Results
if score_button and features and api_status:
    with st.spinner("Calculating credit score..."):
        result = call_api_predict(customer_id, features, is_raw=True)
    
    if result:
        st.success("Score calculated!")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Credit Score", f"{result['credit_score']}/850") 
        with col2: 
            st.metric("Default Probability", f"{result['probability_of_default']*100:.4f}") 
        with col3: 
            st.metric("Risk Tier", result['risk_tier']) 
            # Debug output 
            st.info(f"Debug: PD={result['probability_of_default']}, Score={result['credit_score']}, Tier={result['risk_tier']}") 
            # Optionally log to console 
            import logging 
            logging.info(f"DASHBOARD DEBUG: PD={result['probability_of_default']}, Score={result['credit_score']}, Tier={result['risk_tier']}") 
        with col4: 
            status = "APPROVED" if result.get('approved', False) else "REJECTED" 
            st.metric("Decision", status) 
        
        # Gauge Chart
        st.markdown("---")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=result['credit_score'],
            title={'text': "Credit Score"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [300, 850]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [300, 580], 'color': "#d62728"},
                    {'range': [580, 670], 'color': "#ff7f0e"},
                    {'range': [670, 740], 'color': "#1f77b4"},
                    {'range': [740, 850], 'color': "#2ca02c"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 650
                }
            }
        ))
        fig_gauge.update_layout(height=350)
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Score Interpretation
        st.info(f"""
        **Score Interpretation:**
        - Credit Score: {result['credit_score']}/850 ({result['risk_tier']})
        - Default Probability: {result['probability_of_default']*100:.2f}%
        - Decision: {"Approved" if result['approved'] else "Rejected"} (threshold: 650)
        """)

