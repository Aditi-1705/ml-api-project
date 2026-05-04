from sklearn.linear_model import LogisticRegression
import numpy as np
import pickle

# Dummy dataset
X = np.array([[1,2], [2,3], [3,4], [4,5]])
y = np.array([0, 0, 1, 1])

model = LogisticRegression()
model.fit(X, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved!")
