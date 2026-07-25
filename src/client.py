import streamlit as st
import requests

API_URL = "http://localhost:8000/predict"

st.title("SMS Spam Classifier")

text = st.text_input("Enter text message", width="stretch")

if st.button("Predict", width="stretch"):
    payload = {
        "text": text,
    }

    response = requests.post(API_URL, json=payload)

    if response.status_code == 200:
        prediction = response.json()["prediction"]

        classes = {0: "Ham", 1: "Spam"}

        st.success(f"Prediction: {classes[prediction]}")
    else:
        st.error(response.text)
