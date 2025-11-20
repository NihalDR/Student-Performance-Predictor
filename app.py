from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pickle
import numpy as np
import os
from datetime import datetime

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

# Optional: Initialize Google Sheets storage
try:
    from google_sheets_config import init_google_sheets
    google_sheets = init_google_sheets('credentials.json')
except ImportError:
    print("ℹ️  Google Sheets integration not available.")
    google_sheets = None
except Exception as e:
    print(f"ℹ️  Google Sheets initialization skipped: {type(e).__name__}")
    google_sheets = None

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

        # Store to Google Sheets if available
        if google_sheets:
            try:
                student_data = {
                    'name': data.get('name', 'Unknown'),
                    'year': data.get('year', ''),
                    'attendance': data.get('attendance', ''),
                    'study_hours': data.get('study_hours', ''),
                    'internal_marks': data.get('internal_marks', ''),
                    'assignments': data.get('assignments', ''),
                    'activities': data.get('activities', '')
                }
                
                # Debug logging
                print(f"DEBUG: Received data: {data}")
                print(f"DEBUG: Student data: {student_data}")
                
                prediction_result = {
                    'prediction': result,
                    'confidence': confidence.replace('%', '')
                }
                
                # Calculate risk category (same logic as frontend)
                risk_score = 0
                attendance = float(data.get('attendance', 0))
                internal_marks = float(data.get('internal_marks', 0))
                study_hours = float(data.get('study_hours', 0))
                assignments = float(data.get('assignments', 0))
                activities = float(data.get('activities', 0))
                
                if attendance < 60: risk_score += 30
                elif attendance < 75: risk_score += 20
                elif attendance < 90: risk_score += 10
                
                if internal_marks < 40: risk_score += 30
                elif internal_marks < 60: risk_score += 20
                elif internal_marks < 75: risk_score += 10
                
                if study_hours < 1.5: risk_score += 20
                elif study_hours < 2.5: risk_score += 10
                
                if assignments < 3: risk_score += 15
                elif assignments < 5: risk_score += 10
                
                if activities == 0: risk_score += 5
                
                if risk_score <= 20:
                    risk_category = 'Low Risk'
                elif risk_score <= 50:
                    risk_category = 'Medium Risk'
                else:
                    risk_category = 'High Risk'
                
                google_sheets.add_prediction(student_data, prediction_result, risk_category)
            except Exception as e:
                print(f"Error storing to Google Sheets: {e}")

        return jsonify({'prediction': result, 'confidence': confidence})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get statistics from Google Sheets"""
    try:
        if not google_sheets:
            return jsonify({'error': 'Google Sheets not configured'}), 500
        
        stats = google_sheets.get_statistics()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/student/<name>', methods=['GET'])
def get_student_data(name):
    """Get all predictions for a specific student"""
    try:
        if not google_sheets:
            return jsonify({'error': 'Google Sheets not configured'}), 500
        
        records = google_sheets.get_student_records(name)
        return jsonify({'student': name, 'records': records})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    # Use explicit host and port for clarity
    app.run(host='127.0.0.1', port=5000, debug=True)
