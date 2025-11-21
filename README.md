
# Student Performance Predictor

🎓 Predict student performance using a lightweight ML pipeline and a simple web interface.

This repository contains a small end-to-end demo: data, training scripts, a saved model, a Flask backend, and a static frontend for running predictions locally.

—

## ✨ Highlights

- Clean, reproducible training pipeline (scripts in `model/`).
- Quick demo API served by `backend/app.py` (Flask).
- Minimal static frontend in `frontend/index.html` for quick local testing.
- Datasets are under `dataset/` and small model artifacts are included for demo purposes.

## 🗂 Project Structure

```
./
├─ backend/               # Flask app and server code (backend/app.py)
├─ frontend/              # Static frontend (index.html)
├─ model/                 # Training scripts and helpers
├─ dataset/               # CSVs used to train the model
├─ demo/                  # Demo assets (video is kept locally, not tracked remotely)
├─ requirements.txt       # Python dependencies
├─ README.md              # This file
└─ LICENSE
```

## 🚀 Quick Start (local)

Requirements
- Python 3.10 or newer
- Git

Create & activate a virtual environment (PowerShell):

```powershell
python -m venv .venv
& ".\.venv\Scripts\Activate.ps1"
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run the backend (from repository root):

```powershell
python backend/app.py
```

Open `frontend/index.html` in your browser or send requests to the API (default: `http://127.0.0.1:5000`).

## 🧠 Train or Re-train the Model

Training scripts are in `model/`.

To train with the improved pipeline:

```powershell
python model/train_model_improved.py
```

This writes model artifacts (e.g. `model.pkl`, `scaler.pkl`) which the backend expects for inference.

## ⚙️ Backend API (example)

The Flask app exposes endpoints to run predictions — see `backend/app.py` for exact routes. Example (curl):

```powershell
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{ \"feature1\": 10, \"feature2\": 1 }"
```

Adjust payload to match the features your chosen training script expects.

```

## 🛠️ Development Notes

- Dependencies are listed in `requirements.txt`.
- If `backend/app.py` imports external services (e.g. any cloud/AI SDKs), ensure credentials and env vars are set before running.

## 🤝 Contributing

Contributions are welcome — open an issue or submit a PR. Please avoid committing large binaries; use Git LFS or external hosting.








