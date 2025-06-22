from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS 
import pickle 
import pandas as pd
import os
import json
import requests

app = Flask(__name__)
CORS(app)


MODEL_URL = "https://github.com/bismark-bakomora/term-deposit-prediction/releases/tag/v1.0/model.pkl"
MODEL_PATH = "model/model.pkl"

if not os.path.exists(MODEL_PATH):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    print("Downloading model...")
    r = requests.get(MODEL_URL)
    with open(MODEL_PATH, 'wb') as f:
        f.write(r.content)

# Loading model
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# Loading feature names
with open("model/features.json", 'r') as f:
    feature_names = json.load(f)

@app.route('/')
def home():
    return send_from_directory('../frontend', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame([data])

    # Ensuring consistent features
    df = pd.get_dummies(df)
    for col in feature_names:
        if col not in df:
            df[col] = 0
    df = df[feature_names]

    prediction = model.predict(df)[0]
    return jsonify({'prediction': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
