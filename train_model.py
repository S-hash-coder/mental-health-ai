"""
train_model.py
Run this ONCE to train and save the mental health risk model.
Usage: python train_model.py
"""
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ── Synthetic training data ──────────────────────────────────────────────────
# Features: [sleep_hours, stress_level, study_pressure, social_anxiety, mood_score]
# Labels:   0 = Low Risk, 1 = Medium Risk, 2 = High Risk

np.random.seed(42)
N = 600

def generate_samples(n, sleep_range, stress_range, pressure_range, anxiety_range, mood_range, label):
    return pd.DataFrame({
        'sleep_hours':    np.random.uniform(*sleep_range,   n),
        'stress_level':   np.random.uniform(*stress_range,  n),
        'study_pressure': np.random.uniform(*pressure_range,n),
        'social_anxiety': np.random.uniform(*anxiety_range, n),
        'mood_score':     np.random.uniform(*mood_range,    n),
        'risk_level':     label
    })

low    = generate_samples(N//3, (6,9),  (1,4),  (1,4),  (1,4),  (6,10), 0)
medium = generate_samples(N//3, (5,7),  (4,7),  (4,7),  (4,7),  (3,7),  1)
high   = generate_samples(N//3, (2,6),  (7,10), (7,10), (7,10), (1,4),  2)

df = pd.concat([low, medium, high]).sample(frac=1, random_state=42).reset_index(drop=True)

X = df[['sleep_hours', 'stress_level', 'study_pressure', 'social_anxiety', 'mood_score']]
y = df['risk_level']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── Train ────────────────────────────────────────────────────────────────────
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

print("=== Model Evaluation ===")
print(classification_report(y_test, clf.predict(X_test),
                             target_names=["Low Risk", "Medium Risk", "High Risk"]))

# ── Save ─────────────────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
with open("models/risk_model.pkl", "wb") as f:
    pickle.dump(clf, f)

print("✅ Model saved to models/risk_model.pkl")
