# train_model.py
import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 1. Get a toy dataset ---------------------------------------------------------
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Train a simple model ------------------------------------------------------
clf = LogisticRegression(max_iter=200, multi_class="ovr")
clf.fit(X_train, y_train)
print(f"Accuracy on hold-out: {clf.score(X_test, y_test):.3f}")

# 3. Persist to disk -----------------------------------------------------------
MODEL_FILE = "iris_logreg.joblib"
joblib.dump(clf, MODEL_FILE)
print(f"Model saved to {MODEL_FILE}")
