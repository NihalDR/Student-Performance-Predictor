# Student Performance Predictor

Brief student performance prediction project that demonstrates data preprocessing, model training, and a small Flask backend + static frontend for inference.

**Contents**
- `backend/` – Flask app and server code (`backend/app.py`).
- `frontend/` – Static frontend (`frontend/index.html`).
- `model/` – Model training scripts (`train_model.py`, `train_model_improved.py`) and related artifacts.
- `dataset/` – Original datasets used for training (`training_data.csv`, `student_history.csv`).
- `demo/` – Demo assets (video removed from git history; keep locally).

## Quick Start (local)

Prerequisites:
- Python 3.10+ (recommended)
- A virtual environment (venv)

1. Create and activate a virtual environment (PowerShell):

```powershell
python -m venv .venv
& ".\.venv\Scripts\Activate.ps1"
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the backend Flask app (from repository root):

```powershell
python backend/app.py
```

By default the server will start on `http://127.0.0.1:5000` (see `backend/app.py` for configuration).

4. Open the frontend in a browser (`frontend/index.html`) or navigate to the backend endpoints as implemented.

## Training the model
- Use `model/train_model.py` or `model/train_model_improved.py` to train a model from `dataset/training_data.csv`.
- Example (from repo root):

```powershell
python model/train_model_improved.py
```

Model artifacts (e.g. `model.pkl`, `scaler.pkl`) are included in the repository for demo purposes. If you re-train, update these artifacts accordingly.



## Deployment
- This project is a simple Flask app + static frontend. For production deployment consider using a WSGI server (Gunicorn/uWSGI) behind a reverse proxy, or deploy on a platform such as Heroku, Render, or Azure App Service.

## Contributing
- Open an issue or submit a PR with improvements. Please avoid committing large binary files — use Git LFS or external hosting.

## License
See the `LICENSE` file in the repository root.

## Contact
Project owner: GitHub `NihalDR` (repository `Student-Performance-Predictor`).

