from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pickle
import numpy as np
import os

# import the training helper so we can create a model if missing
import train_model

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Load the trained model or create it if missing
MODEL_PATH = 'model.pkl'
model = None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
else:
    print("model.pkl not found — training model now (this may take a few seconds)...")
    model = train_model.train_and_save_model(MODEL_PATH)

@app.route('/')
def home():
    # Serve the local index.html from project root so Flask can host the frontend
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # Extract features based on the frontend inputs
        features = [
            float(data.get('attendance', 0)),
            float(data.get('study_hours', 0)),
            float(data.get('internal_marks', 0)),
            float(data.get('assignments', 0)),
            float(data.get('activities', 0))
        ]

        final_features = [np.array(features)]

        if model is None:
            return jsonify({'error': 'Model not available'}), 500

        prediction = model.predict(final_features)
        probability = model.predict_proba(final_features)

        result = "Pass" if prediction[0] == 1 else "Fail"
        confidence = f"{np.max(probability) * 100:.2f}%"

        return jsonify({'prediction': result, 'confidence': confidence})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    # Use explicit host and port for clarity
    app.run(host='127.0.0.1', port=5000, debug=True)