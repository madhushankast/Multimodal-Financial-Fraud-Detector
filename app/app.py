"""
Streamlit Web Application.
Interactive dashboard for real-time transaction fraud detection, risk scoring,
feature attribution / inspection, and threshold adjustment.
"""
import streamlit as st

st.set_page_config(
    page_title="Multimodal Financial Fraud Detector",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Multimodal Financial Fraud Detector")
st.markdown("Real-time transaction fraud scoring with categorical embeddings and deep neural networks.")
