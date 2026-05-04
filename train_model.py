import numpy as np
import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Models with pipelines (IMPORTANT: scaling for LR & SVM)
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=10000))
    ]),
    
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC())
    ]),
    
    "Random Forest": Pipeline([
        ("model", RandomForestClassifier())
    ])
}

best_model = None
best_accuracy = 0

print("\n--- Model Comparison ---")

for name, pipeline in models.items():
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    
    print(f"{name}: {acc:.4f}")
    
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = pipeline

# Save best model
joblib.dump(best_model, "model.pkl")

print("\nBest Model Selected with Accuracy:", best_accuracy)
