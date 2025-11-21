import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import os


def train_and_save_model(output_path: str = 'model.pkl', data_path: str = 'training_data.csv'):
    """Train a RandomForest model on synthetic data and save it to `output_path`.
    Also saves the synthetic data to `data_path` as CSV.

    Returns the trained model object.
    """
    data_size = 500
    np.random.seed(42)

    data = {
        'attendance': np.random.randint(50, 100, data_size),
        'study_hours': np.random.randint(1, 10, data_size),
        'internal_marks': np.random.randint(20, 100, data_size),
        'assignments_submitted': np.random.randint(0, 10, data_size),
        'activities_participation': np.random.randint(0, 2, data_size)  # 0 = No, 1 = Yes
    }

    df = pd.DataFrame(data)
    
    # Save synthetic data to CSV
    data_dir = os.path.dirname(os.path.abspath(data_path))
    if data_dir and not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
    
    df.to_csv(data_path, index=False)
    print(f"Synthetic data saved as '{data_path}'")

    def determine_pass_fail(row):
        score = (row['attendance'] * 0.3) + (row['internal_marks'] * 0.5) + (row['study_hours'] * 2)
        if row['assignments_submitted'] < 5:
            score -= 10
        return 1 if score > 65 else 0

    df['performance'] = df.apply(determine_pass_fail, axis=1)

    X = df.drop('performance', axis=1)
    y = df['performance']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {acc * 100:.2f}%")

    # Ensure output directory exists
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"Model saved as '{output_path}'")
    return model


if __name__ == '__main__':
    train_and_save_model('model.pkl', 'training_data.csv')
