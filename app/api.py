from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS 
import pickle 
import pandas as pd
import os
import json

app = Flask(__name__)
CORS(app)

# Load model
with open("model/model.pkl", 'rb') as f:
    model = pickle.load(f)

# Load feature names
with open("model/features.json", 'r') as f:
    feature_names = json.load(f)

@app.route('/')
def home():
    return send_from_directory('../frontend', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame([data])

    # Ensure consistent features
    df = pd.get_dummies(df)
    for col in feature_names:
        if col not in df:
            df[col] = 0
    df = df[feature_names]

    prediction = model.predict(df)[0]
    return jsonify({'prediction': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
