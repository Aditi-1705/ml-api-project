import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Load dataset
data = load_breast_cancer()

# Use ONLY 5 features
X = data.data[:, [0, 1, 2, 3, 4]]

y = data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Models
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

best_model = None
best_accuracy = 0
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

        best_name = name

# Save best model
joblib.dump(best_model, "model.pkl")

# Final evaluation
final_preds = best_model.predict(X_test)

print("\nBest Model:", best_name)

print("Best Accuracy:", best_accuracy)

print("\nConfusion Matrix:")

print(confusion_matrix(y_test, final_preds))

print("\nClassification Report:")

print(classification_report(y_test, final_preds))
