# app.py
from typing import List

from fastapi import FastAPI, HTTPException
import numpy as np
from pydantic import BaseModel, Field

from model_registry import load_model


# -----------------------------------------------------------------------------#
# 1. Load model at start-up (only once)                                        #
# -----------------------------------------------------------------------------#
model = load_model()

# -----------------------------------------------------------------------------#
# 2. Define request/response schemas                                           #
# -----------------------------------------------------------------------------#
class IrisFeatures(BaseModel):
    data: List[float] = Field(..., min_length=4, max_length=4)

class PredictionOut(BaseModel):
    prediction: int      # 0, 1 or 2
    class_name: str      # human-readable

TARGET_NAMES = ["setosa", "versicolor", "virginica"]

# -----------------------------------------------------------------------------#
# 3. Create the FastAPI app                                                    #
# -----------------------------------------------------------------------------#
app = FastAPI(
    title="Iris classifier",
    description="Demo ML inference with FastAPI",
    version="0.1.0",
)

@app.post("/predict", response_model=PredictionOut)
def predict(features: IrisFeatures):
    """Return the predicted iris class for a single observation."""
    x = np.array(features.data).reshape(1, -1)
    try:
        pred_idx = int(model.predict(x)[0])
    except Exception as e:  # generic catch for demo purposes
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "prediction": pred_idx,
        "class_name": TARGET_NAMES[pred_idx],
    }


## RUN IT:
## uvicorn app_post:app --reload

# curl -X POST http://127.0.0.1:8000/predict -H "accept: application/json" -H "Content-Type: application/json" -d "{\"data\":[5.1,3.5,1.4,0.2]}"
