import streamlit as st
from app.predictor import run_prediction  # You'll define this in predictor.py

st.title("Smart Prediction Wizard")
uploaded_file = st.file_uploader("Upload your CSV or text file")
model_type = st.selectbox("Choose model type", ["Time Series", "Deep Learning"])

if st.button("Predict") and uploaded_file:
    result = run_prediction(uploaded_file, model_type)
    st.write("Prediction Results:")
    st.write(result)
