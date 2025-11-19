# Student-Performance-Predictor
Team:Orbit
Teammates:M Tanusree Reddy,Nihal DR,P Devesh Reddy,Rachana Naidu

![Student Performance Predictor](https://img.shields.io/badge/Status-Active-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Flask](https://img.shields.io/badge/Flask-2.0+-green)
![ML](https://img.shields.io/badge/ML-scikit--learn-orange)

## 📋 Overview

Student Performance Predictor is a machine learning web application that predicts student academic performance based on key metrics like attendance, study hours, internal marks, assignments submitted, and activities participation. The application provides:

- **Performance Prediction**: Pass/Fail classification
- **Confidence Score**: Model confidence in the prediction (0-100%)
- **Personalized Feedback**: AI-generated suggestions for improvement
- **Beautiful UI**: Modern, responsive web interface

## 🛠️ Tech Stack

### Backend
- **Python 3.8+** - Programming language
- **Flask 2.0+** - Web framework for REST API
- **scikit-learn** - Machine learning library
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing

### Frontend
- **HTML5** - Markup language
- **CSS3** - Styling with gradients and animations
- **JavaScript (Vanilla)** - Client-side logic

### Machine Learning
- **RandomForestClassifier** - 100 decision trees ensemble model
- **Model Accuracy**: 92% on training data
- **Training Data**: 500 synthetic student records

### File Storage
- **pickle** - Model serialization
- **CSV** - Data storage

## 📦 Project Structure

```
Student-Performance-Predictor/
│
├── app.py                      # Flask application & API routes
├── train_model.py              # Model training & data generation
├── model.pkl                   # Trained RandomForest model (binary)
├── training_data.csv           # Synthetic training dataset (500 rows)
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
└── templates/
    └── index.html              # Frontend UI
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Git (optional)

### Step 1: Clone or Download the Project
```bash
cd "d:\Full stack oddy\New folder\New folder"
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Train the Model (Optional - Model already trained)
```bash
python train_model.py
```
This generates:
- `model.pkl` - Trained model
- `training_data.csv` - Synthetic training data

### Step 5: Run the Flask Application
```bash
python app.py
```

The application will start on: **http://127.0.0.1:5000**

## 📋 Requirements

### Python Dependencies (requirements.txt)
```
Flask==2.3.0
scikit-learn==1.3.0
pandas==2.0.0
numpy==1.24.0
```

### Install Command
```bash
pip install -r requirements.txt
```

## 🎯 Features

### 1. **Predict Performance**
- Input student metrics (attendance, study hours, internal marks, assignments, activities)
- Get Pass/Fail prediction with confidence score
- Instant results display

### 2. **Confidence Meter**
- Shows model's confidence (0-100%)
- Based on RandomForest tree voting consensus
- Higher % = More reliable prediction

### 3. **Personalized Feedback**
Automatic suggestions based on:
- **Attendance**: Warn if <75%, encourage if 75-90%, praise if 90%+
- **Study Hours**: Recommend 2-4+ hours daily
- **Internal Marks**: Targeted guidance by performance level
- **Assignments**: Emphasize timely submission importance
- **Activities**: Encourage academic participation
- **Overall Status**: Congratulatory or urgent action messages

### 4. **Sample Data**
- "Use Sample" button pre-fills form with demo data
- Quick testing without manual input

### 5. **Responsive Design**
- Works on desktop, tablet, and mobile
- Beautiful gradient background
- Smooth animations and transitions

## 🧠 Machine Learning Model

### Model Type
**RandomForestClassifier** with 100 decision trees

### Features (Input)
1. **Attendance** (%)  - Range: 50-100
2. **Study Hours** (per day) - Range: 1-10
3. **Internal Marks** (0-100) - Range: 20-100
4. **Assignments Submitted** - Range: 0-10
5. **Activities Participation** - Binary (0=No, 1=Yes)

### Target (Output)
- **Pass** (1) or **Fail** (0)

### Decision Logic
```
score = (attendance × 0.3) + (internal_marks × 0.5) + (study_hours × 2)
if assignments_submitted < 5: score -= 10
Pass if score > 65, else Fail
```

### Performance
- **Training Accuracy**: 92%
- **Test Set Size**: 20% (100 samples)
- **Training Data Size**: 500 synthetic records

### How Confidence is Calculated
```
confidence = (trees_voting_pass / total_trees) × 100
Example: 78 trees vote "Pass" out of 100 = 78% confidence
```

## 📊 API Endpoints

### GET `/`
Returns the HTML interface

### POST `/predict`
**Request Body:**
```json
{
  "attendance": 85,
  "study_hours": 3,
  "internal_marks": 72,
  "assignments_submitted": 5,
  "activities_participation": 0
}
```

**Response:**
```json
{
  "prediction": "Pass",
  "confidence": 78.0,
  "remarks": [
    "Good attendance! Keep it above 90%...",
    "Consider increasing study hours to 4+ per day...",
    "..."
  ]
}
```

## 💡 Usage Examples

### Example 1: Strong Student
**Input:**
- Attendance: 95%
- Study Hours: 5/day
- Internal Marks: 85
- Assignments: 8
- Activities: Yes

**Output:** Pass (95% confidence)

### Example 2: Struggling Student
**Input:**
- Attendance: 60%
- Study Hours: 1/day
- Internal Marks: 35
- Assignments: 2
- Activities: No

**Output:** Fail (89% confidence)

## 📁 Training Data (training_data.csv)

Contains 500 synthetic student records with columns:
- `attendance` - Student attendance percentage
- `study_hours` - Daily study hours
- `internal_marks` - Internal exam marks
- `assignments_submitted` - Number of assignments completed
- `activities_participation` - Binary participation flag

### Sample Data
```
attendance,study_hours,internal_marks,assignments_submitted,activities_participation
88,5,24,4,0
78,7,48,5,1
64,4,66,3,1
92,1,87,3,0
...
```

## 🔧 Configuration

### Model Path
Default: `model.pkl` (in project root)

### Training Data Path
Default: `training_data.csv` (in project root)

### Flask Configuration
- **Host**: 127.0.0.1
- **Port**: 5000
- **Debug Mode**: On (development)

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError"
**Solution:** Install requirements
```bash
pip install -r requirements.txt
```

### Issue: "Port 5000 already in use"
**Solution:** Change port in app.py
```python
app.run(debug=True, port=5001)  # Use different port
```

### Issue: "model.pkl not found"
**Solution:** Retrain the model
```bash
python train_model.py
```

### Issue: Predictions seem incorrect
**Solution:** The model works with synthetic data patterns. For real-world predictions, retrain with actual student data.

## 🎨 Customization

### Change Model Color Scheme
Edit in `index.html`:
```css
--primary-color: #5b5dff;  /* Change to your color */
```

### Adjust Prediction Thresholds
Edit in `train_model.py`:
```python
return 1 if score > 65 else 0  # Change threshold from 65
```

### Add More Features
1. Add columns to synthetic data in `train_model.py`
2. Retrain model
3. Add form inputs in `index.html`
4. Update API in `app.py`

## 📈 Model Improvement Ideas

1. **More Training Data**: Collect real student data
2. **Feature Engineering**: Add GPA, previous scores, etc.
3. **Hyperparameter Tuning**: Optimize tree depth, features per split
4. **Cross-Validation**: Use k-fold CV for better evaluation
5. **Feature Importance**: Analyze which factors matter most
6. **Class Balancing**: Handle imbalanced pass/fail distribution

## 📝 License

This project is open source and available for educational use.

## 👤 Author

Created for hackathon demo - Student Performance Prediction

## 🤝 Contributing

Contributions welcome! Feel free to:
- Report bugs
- Suggest improvements
- Add new features
- Improve documentation

## 📞 Support

For issues or questions:
1. Check the Troubleshooting section
2. Review the API Endpoints
3. Check model accuracy in console output

## 🎓 Educational Value

This project demonstrates:
- **Machine Learning**: Classification with RandomForest
- **Web Development**: Flask REST API
- **Frontend Development**: HTML/CSS/JavaScript
- **Data Science**: Data generation, model training, evaluation
- **Full Stack**: End-to-end ML application development

---

**Made with ❤️ for students to succeed!**
