import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib
import os

st.set_page_config(page_title="Healthcare Diagnosis App", layout="centered")

# Feature columns and possible conditions
symptoms = ["fever", "cough", "fatigue", "sore_throat", "body_aches"]
conditions = {0: "Cold", 1: "Flu", 2: "Allergy"}

# Function to train a simple model (for fallback/demo)
def train_model():
    # Sample training data
    data = pd.DataFrame({
        "fever": [1, 0, 1, 0, 1, 0],
        "cough": [1, 1, 1, 0, 0, 0],
        "fatigue": [1, 0, 1, 0, 0, 1],
        "sore_throat": [1, 0, 0, 1, 0, 1],
        "body_aches": [1, 0, 0, 1, 1, 0],
        "label": [1, 0, 2, 0, 1, 2]  # 0 = Cold, 1 = Flu, 2 = Allergy
    })
    X = data[symptoms]
    y = data["label"]
    clf = DecisionTreeClassifier()
    clf.fit(X, y)
    joblib.dump(clf, "health_diagnosis_model.pkl")
    return clf

# Try to load model
if os.path.exists("health_diagnosis_model.pkl"):
    model = joblib.load("health_diagnosis_model.pkl")
else:
    st.warning("Model file not found. Training a new model with sample data...")
    model = train_model()
    st.success("New model trained and loaded.")

# App UI
st.title("🩺 Low-Cost Healthcare Diagnosis App")
st.write("This app provides a basic symptom-based diagnosis for common conditions.")

# Sidebar input sliders for symptom severity
st.sidebar.title("Input Symptoms (Severity 0-1)")
user_input = {}
for symptom in symptoms:
    user_input[symptom] = st.sidebar.slider(symptom, 0, 1, 0)

input_df = pd.DataFrame([user_input])

if st.sidebar.button("Get Diagnosis"):
    prediction = model.predict(input_df)
    diagnosis = conditions.get(prediction[0], "Unknown")
    st.success(f"🧾 **Predicted Condition:** {diagnosis}")

# Optional retrain button
if st.sidebar.button("Retrain Model"):
    model = train_model()
    st.success("Model retrained with sample data.")
