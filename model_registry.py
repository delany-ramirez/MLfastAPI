"""Shared access to the Iris model stored in the MLflow Model Registry.

`train_model.py` registers each new version and moves the alias to it;
the APIs call `load_model()` once at start-up to serve that version.
"""

import mlflow
from mlflow.exceptions import MlflowException

MODEL_NAME = "iris-logreg"
MODEL_ALIAS = "champion"
MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"


def load_model():
    """Return the scikit-learn model currently behind the registry alias."""
    try:
        return mlflow.sklearn.load_model(MODEL_URI)
    except MlflowException as exc:
        raise RuntimeError(
            f"Could not load {MODEL_URI} from {mlflow.get_tracking_uri()}. "
            "Did you run `uv run train_model.py` first?"
        ) from exc
