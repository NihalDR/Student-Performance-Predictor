from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pickle
import numpy as np
import os
import google.generativeai as genai
import train_model_improved

app = Flask(__name__)
CORS(app)

# --- CONFIGURATION ---
GEMINI_API_KEY = "AIzaSyA1Tc9odl_5k8wyn-3Q2TnoLswp2ycfh5k"

# Paths to artifacts
MODEL_PATH = 'model.pkl'
SCALER_PATH = 'scaler.pkl'

model = None
scaler = None

def load_artifacts():
    global model, scaler
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        print("Loading existing model and scaler...")
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
    else:
        print("Artifacts missing. Training new model...")
        model, scaler = train_model_improved.train_and_save_model(MODEL_PATH, SCALER_PATH)

load_artifacts()

def generate_smart_advice(data, prediction):
    """
    Uses Google Gemini to generate personalized study advice.
    """
    if not GEMINI_API_KEY:
        return None

    try:
        genai.configure(api_key=GEMINI_API_KEY)
        ai_model = genai.GenerativeModel('gemini-pro')
        
        prompt = f"""
        Act as a supportive academic counselor. A student named {data.get('name')} has these stats:
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
        return None

@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
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
    app.run(debug=True)