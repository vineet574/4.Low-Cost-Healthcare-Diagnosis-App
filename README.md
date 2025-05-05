# Low-Cost Healthcare Diagnosis App

This project is a simple healthcare diagnosis app that predicts common conditions based on user-inputted symptoms. Built using Python and Streamlit, this app provides an easy-to-use interface for basic symptom-based health assessments.

## Project Overview
The app uses a basic machine learning model trained on sample data to distinguish between common conditions such as cold and flu. This is a prototype meant for educational purposes, demonstrating how machine learning can be applied to healthcare.

## Features
- Allows users to select symptoms and get a basic diagnosis.
- Utilizes a simple decision tree model for predictions.
- Interactive web interface built with Streamlit.

## File Descriptions
- **app.py**: The main file for running the diagnosis app.
- **train_model.py**: A script to train a basic decision tree model on sample data.
- **health_diagnosis_model.pkl**: Saved model file used for making predictions.
- **README.md**: Documentation file describing the project.

## Installation

### Prerequisites
1. Python 3.6 or higher
2. Install the required libraries by running:
   ```bash
   pip install -r requirements.txt

Example requirements.txt:
pandas
numpy
scikit-learn
streamlit
joblib


Running the Application
Open the terminal in the project directory.
Run the Streamlit app:
streamlit run app.py

Usage
Open the Streamlit web app.
Select the symptoms you’re experiencing.
Click "Get Diagnosis" to see the predicted condition based on the input symptoms.
Project Structure
app.py: Main application file for the Streamlit app.
train_model.py: Model training script.
health_diagnosis_model.pkl: Pre-trained decision tree model for diagnosis predictions.
Example Usage
User Input:

Symptoms selected: fever, cough
Model Output:

Predicted Condition: Cold or Flu (depending on symptoms selected).


Here's an enhanced version of your Streamlit app with new features, including:

Symptom severity sliders (instead of just checkboxes).

Option to retrain the model on sample data.

Expanded diagnosis with multiple possible outcomes.

Model training fallback if no model file is found.
