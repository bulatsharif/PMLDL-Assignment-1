import os
from typing import Dict

import numpy as np
import pandas as pd
import requests
import streamlit as st


st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")
st.title("Titanic Survival Predictor")
st.caption("Enter passenger features and get survival prediction")


API_URL = os.getenv("API_URL", "http://api:8000")


def predict(features: Dict[str, float]) -> Dict:
    payload = {"instances": [features]}
    resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
    resp.raise_for_status()
    return resp.json()


with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        Pclass = st.selectbox("Pclass", options=[1, 2, 3], index=2)
        Age = st.number_input("Age", min_value=0.0, max_value=100.0, value=30.0)
        SibSp = st.number_input("SibSp", min_value=0, max_value=10, value=0)
        Parch = st.number_input("Parch", min_value=0, max_value=10, value=0)
        Fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=32.0)
    with col2:
        embarked = st.selectbox("Embarked", options=["S", "C", "Q"], index=0)
        gender = st.selectbox("Gender", options=["male", "female"], index=0)

    submitted = st.form_submit_button("Predict")

if submitted:
    C = 1 if embarked == "C" else 0
    Q = 1 if embarked == "Q" else 0
    S = 1 if embarked == "S" else 0
    female = 1 if gender == "female" else 0
    male = 1 if gender == "male" else 0

    features = {
        "Pclass": int(Pclass),
        "Age": float(Age),
        "SibSp": int(SibSp),
        "Parch": int(Parch),
        "Fare": float(Fare),
        "C": int(C),
        "Q": int(Q),
        "S": int(S),
        "female": int(female),
        "male": int(male),
    }

    try:
        result = predict(features)
        prob = result.get("probabilities", [0.0])[0]
        pred = result.get("predictions", [0])[0]
        st.success(f"Predicted: {'Survived' if pred == 1 else 'Did not survive'} (p={prob:.2f})")
    except Exception as e:
        st.error(f"Prediction failed: {e}")


