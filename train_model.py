import numpy as np
import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns


# =========================
# 1. Load Dataset
# =========================
data = load_breast_cancer()
X = data.data
y = data.target


# =========================
# 2. Train-Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# =========================
# 3. Define Models
# =========================
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=10000))
    ]),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(probability=True))
    ]),

    "Random Forest": Pipeline([
        ("model", RandomForestClassifier())
    ])
}


# =========================
# 4. Train + Compare Models
# =========================
best_model = None
best_accuracy = 0
best_preds = None
best_name = ""

print("\n--- Model Comparison ---")

for name, pipeline in models.items():
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"{name}: {acc:.4f}")

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = pipeline
        best_preds = preds
        best_name = name


# =========================
# 5. Save Best Model
# =========================
joblib.dump(best_model, "model.pkl")


# =========================
# 6. Results
# =========================
print("\nBest Model:", best_name)
print("Best Accuracy:", best_accuracy)


# =========================
# 7. Confusion Matrix
# =========================
cm = confusion_matrix(y_test, best_preds)

print("\nConfusion Matrix:")
print(cm)


# =========================
# 8. Classification Report
# =========================
print("\nClassification Report:")
print(classification_report(y_test, best_preds))


# =========================
# 9. Visualization
# =========================
plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.savefig("confusion_matrix.png")
plt.show()
