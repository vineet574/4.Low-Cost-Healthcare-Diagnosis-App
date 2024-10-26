import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

# Sample data (basic example)
data = {
    "fever": [1, 1, 0, 0],
    "cough": [1, 0, 1, 1],
    "fatigue": [1, 1, 0, 0],
    "sore_throat": [1, 0, 1, 0],
    "body_aches": [1, 1, 0, 0],
    "condition": [1, 1, 0, 0]  # 1 = Flu, 0 = Cold
}
df = pd.DataFrame(data)

# Separate features and target
X = df.drop("condition", axis=1)
y = df["condition"]

# Train a decision tree model
model = DecisionTreeClassifier()
model.fit(X, y)

# Save the model
joblib.dump(model, "health_diagnosis_model.pkl")
print("Model trained and saved.")
