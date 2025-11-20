import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import google.generativeai as genai

# --- IMPORT TRAINING MODULE ---
# Ensure 'train_model_improved.py' is in your GitHub repository.
try:
    import train_model_improved
except ImportError:
    print("WARNING: 'train_model_improved.py' not found. If model.pkl is missing, the app will crash.")
    train_model_improved = None

app = Flask(__name__)
CORS(app)

# --- CONFIGURATION ---
# UPDATED: Securely load API Key from Render Environment Variables
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Paths to artifacts
MODEL_PATH = 'model.pkl'
SCALER_PATH = 'scaler.pkl'

model = None
scaler = None

def load_artifacts():
    global model, scaler
    # Check if model files exist
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        print("Loading existing model and scaler...")
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
    else:
        # If files are missing, try to retrain
        print("Artifacts missing. Attempting to train new model...")
        if train_model_improved:
            model, scaler = train_model_improved.train_and_save_model(MODEL_PATH, SCALER_PATH)
        else:
            # Fallback if training script is missing
            print("CRITICAL ERROR: Model files missing and 'train_model_improved.py' not found.")
            model, scaler = None, None

# Load model on startup
load_artifacts()

def generate_smart_advice(data, prediction):
    """
    Uses Google Gemini to generate personalized study advice.
    """
    # If no API key is set in Render, skip AI features gracefully
    if not GEMINI_API_KEY:
        print("Gemini API Key missing. Skipping AI advice.")
        return "AI advice unavailable (API Key missing)."

    try:
        genai.configure(api_key=GEMINI_API_KEY)
        ai_model = genai.GenerativeModel('gemini-pro')
        
        prompt = f"""
        Act as a supportive academic counselor. A student named {data.get('name', 'Student')} has these stats:
        - Attendance: {data.get('attendance')}%
        - Internal Marks: {data.get('internal_marks')}%
        - Study Hours: {data.get('study_hours')}/week
        - Assignments: {data.get('assignments')}/10
        - Predicted Result: {prediction}
        
        Give 2 short, specific, actionable sentences of advice to improve their result. 
        Address their weakest area directly. Be encouraging but firm. 
        Output ONLY the advice text.
        """
        
        response = ai_model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"AI Advice Error: {e}")
        return "Could not generate advice at this time."

@app.route('/')
def home():
    # Serves index.html from the current directory
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not scaler:
        return jsonify({'error': 'Model not loaded. Server configuration error.'}), 500

    try:
        data = request.json
        
        # 1. Validation
        required = ['attendance', 'study_hours', 'internal_marks', 'assignments', 'activities']
        if not all(k in data for k in required):
            return jsonify({'error': 'Missing fields'}), 400

        # 2. Extraction
        name = data.get('name', 'Student')
        student_id = data.get('student_id', 'N/A')
        
        features = np.array([[
            float(data['attendance']), 
            float(data['study_hours']), 
            float(data['internal_marks']), 
            float(data['assignments']), 
            float(data['activities'])
        ]])

        # 3. Prediction
        scaled_features = scaler.transform(features)
        pred = model.predict(scaled_features)[0]
        prob = model.predict_proba(scaled_features)
        
        result_text = "Pass" if pred == 1 else "Fail"
        conf_score = np.max(prob) * 100

        # 4. AI Advice Generation
        ai_suggestion = generate_smart_advice(data, result_text)

        return jsonify({
            'name': name,
            'student_id': student_id,
            'prediction': result_text,
            'confidence': f"{conf_score:.2f}%",
            'raw_probability': float(np.max(prob)),
            'ai_advice': ai_suggestion
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Use PORT environment variable if available (Render sets this)
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)