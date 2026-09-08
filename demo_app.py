from src.model import Model

import streamlit as st
import requests

if "model" not in st.session_state:
    st.session_state.model = Model()

st.title("SMS Spam Classifier - Demo")

text = st.text_input("Enter text message", width="stretch")

if st.button("Predict", width="stretch"):
    prediction = st.session_state.model.predict(text)

    classes = {0: "Ham", 1: "Spam"}

    st.success(f"Prediction: {classes[prediction]}")
