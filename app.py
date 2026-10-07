import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Medical Insurance Premium Predictor", layout="centered")

st.title("🏥 Medical Insurance Premium Predictor")
st.write("Enter your details below to estimate your medical insurance charges based on our trained neural network model.")

@st.cache_resource
def load_assets():
    model = joblib.load('insurance_model.joblib')
    scaler = joblib.load('scaler.joblib')
    return model, scaler

try:
    model, scaler = load_assets()
except Exception as e:
    st.error(f"Error loading model or scaler: {e}. Please ensure 'insurance_model.joblib' and 'scaler.joblib' are present in the directory.")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=30, step=1)
    bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
    children = st.number_input("Number of Children", min_value=0, max_value=10, value=1, step=1)

with col2:
    sex = st.selectbox("Sex", options=["male", "female"])
    smoker = st.selectbox("Are you a smoker?", options=["yes", "no"])
    region = st.selectbox("Region", options=["northeast", "northwest", "southeast", "southwest"])

if st.button("Estimate Insurance Charges", use_container_width=True):
    input_data = pd.DataFrame({
        'age': [age],
        'bmi': [bmi],
        'children': [children],
        'sex_male': [True if sex == 'male' else False],
        'smoker_yes': [True if smoker == 'yes' else False],
        'region_northwest': [True if region == 'northwest' else False],
        'region_southeast': [True if region == 'southeast' else False],
        'region_southwest': [True if region == 'southwest' else False]
    })
    
    try:
        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)
        predicted_value = max(0.0, float(prediction[0][0]))
        
        st.success(f"### Predicted Annual Insurance Premium: **${predicted_value:,.2f}**")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
