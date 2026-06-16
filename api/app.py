from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load model και threshold
model = joblib.load('models/saved/final_model.pkl')
threshold = joblib.load('models/saved/best_threshold.pkl')

# Input schema
class VisitorData(BaseModel):
    num_views: int
    added_to_cart: int
    session_duration_min: float
    total_items_viewed: float
    hour_of_day: int
    day_of_week: int

@app.get("/")
def root():
    return {"message": "E-commerce Conversion Prediction API"}

@app.post("/predict")
def predict(data: VisitorData):
    features = np.array([[
        data.num_views,
        data.added_to_cart,
        data.session_duration_min,
        data.total_items_viewed,
        data.hour_of_day,
        data.day_of_week
    ]])
    
    prob = model.predict_proba(features)[0][1]
    prediction = int(prob >= threshold)
    
    return {
        "will_purchase": bool(prediction),
        "probability": round(float(prob), 4),
        "threshold": round(float(threshold), 4)
    }