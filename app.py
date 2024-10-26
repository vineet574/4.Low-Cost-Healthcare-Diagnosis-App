import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load the trained model (if already trained)
try:
    model = joblib.load("health_diagnosis_model.pkl")
except FileNotFoundError:
    st.error("Model not found. Please ensure the trained model file is available.")

# Feature columns for symptoms
symptoms = ["fever", "cough", "fatigue", "sore_throat", "body_aches"]

# Sidebar for user input
st.sidebar.title("Enter Symptoms")
user_symptoms = {symptom: st.sidebar.checkbox(symptom) for symptom in symptoms}

# Convert input to DataFrame format
input_data = pd.DataFrame([user_symptoms])

# Display results
st.title("Low-Cost Healthcare Diagnosis App")
st.write("This app provides a basic symptom-based diagnosis for common conditions.")

if st.sidebar.button("Get Diagnosis"):
    if model:
        prediction = model.predict(input_data)
        diagnosis = "Flu" if prediction[0] == 1 else "Cold"
        st.write(f"**Predicted Condition:** {diagnosis}")
    else:
        st.write("Model is not loaded.")

