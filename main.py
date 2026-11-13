from pathlib import Path
import joblib
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
import numpy as np

# Load the trained model
model_path = Path("models/iris_random_forest_model.joblib")
if not model_path.exists():
    raise RuntimeError(f"Model file not found at {model_path}. Please run train.py first.")
model = joblib.load(model_path)

app = FastAPI(
    title="Iris Classification API",
    description="Iris prediction API",
    version="1.0.0"
)

class IrisInput(BaseModel):
    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)

@app.get("/")
def ready_to_serve():
    return {"status": "Ready to serve"}

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_name": "iris-random-forest"
    }

# Endpoint for making predictions with the Iris Random Forest model
@app.post("/predict")
def predict(data: IrisInput):
    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]
    prediction = int(model.predict(features)[0])
    print("Prediction is", prediction)
    return {
        "prediction": prediction,
        "class_name": ["setosa", "versicolor", "virginica"][prediction]
    }
