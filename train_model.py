# train_model.py
import mlflow
from mlflow import MlflowClient
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier

from model_registry import MODEL_ALIAS, MODEL_NAME

EXPERIMENT_NAME = "iris-classifier"
MAX_ITER = 200
TEST_SIZE = 0.2
RANDOM_STATE = 42

# 1. Get a toy dataset ---------------------------------------------------------
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run():
    # 2. Train a simple model --------------------------------------------------
    clf = OneVsRestClassifier(LogisticRegression(max_iter=MAX_ITER))
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    print(f"Accuracy on hold-out: {accuracy:.3f}")

    # 3. Track the run and register the model ----------------------------------
    mlflow.log_params(
        {
            "max_iter": MAX_ITER,
            "strategy": "one-vs-rest",
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
        }
    )
    mlflow.log_metric("accuracy", accuracy)
    model_info = mlflow.sklearn.log_model(
        clf,
        name="model",
        input_example=X_train[:2],
        registered_model_name=MODEL_NAME,
    )

# 4. Point the alias used by the APIs at the new version -----------------------
version = model_info.registered_model_version
MlflowClient().set_registered_model_alias(MODEL_NAME, MODEL_ALIAS, version)
print(f"Model registered as {MODEL_NAME} v{version} (@{MODEL_ALIAS})")
