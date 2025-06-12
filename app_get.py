"""
FastAPI version that exposes two **GET** routes:

1. /predict           ->  ?data=5.1,3.5,1.4,0.2
2. /predict_explicit  ->  ?sepal_len=5.1&sepal_wid=3.5&petal_len=1.4&petal_wid=0.2
"""

from fastapi import FastAPI, HTTPException, Query
import numpy as np
import joblib

MODEL_FILE = "iris_logreg.joblib"
TARGET_NAMES = ["setosa", "versicolor", "virginica"]

# --------------------------------------------------------------------------- #
# 1. Load model at startup                                                    #
# --------------------------------------------------------------------------- #
model = joblib.load(MODEL_FILE)

# --------------------------------------------------------------------------- #
# 2. FastAPI app                                                              #
# --------------------------------------------------------------------------- #
app = FastAPI(title="Iris classifier – GET version")

# -----  Route 1: single CSV parameter -------------------------------------- #
@app.get("/predict")
def predict_csv(
    data: str = Query(
        ...,
        description="Four comma-separated numbers: "
                    "sepal_len,sepal_wid,petal_len,petal_wid",
        examples={"setosa": {"value": "5.1,3.5,1.4,0.2"}}
    )
):
    try:
        values = [float(x) for x in data.split(",")]
    except ValueError:
        raise HTTPException(status_code=400, detail="All values must be numeric.")
    if len(values) != 4:
        raise HTTPException(status_code=400, detail="Need exactly 4 numbers.")

    pred = int(model.predict(np.array(values).reshape(1, -1))[0])
    return {"prediction": pred, "class_name": TARGET_NAMES[pred]}

# -----  Route 2: four named query params ----------------------------------- #
@app.get("/predict_explicit")
def predict_explicit(
    sepal_len: float,
    sepal_wid: float,
    petal_len: float,
    petal_wid: float,
):
    vec = np.array([sepal_len, sepal_wid, petal_len, petal_wid]).reshape(1, -1)
    pred = int(model.predict(vec)[0])
    return {"prediction": pred, "class_name": TARGET_NAMES[pred]}

## RUN IT:
## uvicorn app_get:app --reload
