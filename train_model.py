# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pickle
from imblearn.over_sampling import SMOTE
import json
import os

# Load data
df = pd.read_csv('data/bank-additional-full.csv', sep=';')

# Convert target to binary
y = df['y'].map({'no': 0, 'yes': 1})

# Drop target from features
X = df.drop('y', axis=1)

# One-hot encode
X = pd.get_dummies(X)

# Apply SMOTE
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X, y)

# Train/test split
x_train, x_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

# Predict and evaluate
y_pred = model.predict(x_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred))

# Create model directory if not exist
os.makedirs('model', exist_ok=True)

# Save model
with open('model/model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save feature names
with open('model/features.json', 'w') as f:
    json.dump(list(X.columns), f)
