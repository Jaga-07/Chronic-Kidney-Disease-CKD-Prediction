import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model and scaler
model = joblib.load('tuned_decision_tree_model.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="CKD Prediction Dashboard", layout="centered")

# Streamlit App Title
st.title("Chronic Kidney Disease (CKD) Prediction")
st.write("Enter the patient's medical details below to predict CKD.")

# Sidebar for user input
st.sidebar.header("Enter Patient Details")

# Input features
sg = st.sidebar.slider('Specific Gravity (sg)', min_value=1.005, max_value=1.025, step=0.001, format="%.3f")
al = st.sidebar.slider('Albumin (al)', min_value=0, max_value=5, step=1)
hemo = st.sidebar.slider('Hemoglobin (hemo)', min_value=3.0, max_value=17.0, step=0.1)
htn = st.sidebar.selectbox('Hypertension (htn)', ['yes', 'no'])
dm = st.sidebar.selectbox('Diabetes Mellitus (dm)', ['yes', 'no'])
bgr = st.sidebar.slider('Blood Glucose Random (bgr)', min_value=22, max_value=490, step=1)
sc = st.sidebar.slider('Serum Creatinine (sc)', min_value=0.4, max_value=17.0, step=0.1)

# Prepare the input data for prediction
input_data = pd.DataFrame({
    'sg': [sg],
    'al': [al],
    'hemo': [hemo],
    'htn': [1 if htn == 'yes' else 0],
    'dm': [1 if dm == 'yes' else 0],
    'bgr': [bgr],
    'sc': [sc]
})

# Scale the input data
input_scaled = scaler.transform(input_data)

# Prediction
if st.sidebar.button("Predict CKD"):
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.success("Prediction: The patient is likely to have Chronic Kidney Disease (CKD).")
    else:
        st.success("Prediction: The patient is unlikely to have Chronic Kidney Disease (CKD).")

    # Optionally display input summary
    st.subheader("Input Summary")
    st.write(input_data)

# Optional: Add some visual explanation if needed
st.write("---")
st.write("Model: Tuned Decision Tree | Developed with ❤️ using Streamlit")
