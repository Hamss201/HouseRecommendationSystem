from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
import pandas as pd
import numpy as np
import joblib
from keras.models import load_model
from datetime import datetime
from pathlib import Path

app = FastAPI()

# Load preprocessing pipeline and model once on startup
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "tensorflow"
MODEL_PATH = MODEL_DIR / "price_model.keras"
SCALER_PATH = MODEL_DIR / "price_pipeline.pkl"

preprocessor = joblib.load(SCALER_PATH)
model = load_model(MODEL_PATH)

class HouseFeatures(BaseModel):
    DIMENSIONS: float = Field(..., example=1200)
    BEDROOMS: int = Field(..., example=3)
    BATHROOMS: int = Field(..., example=2)
    TYPE: str = Field(..., example="Apartment")  # Adjust allowed values if you want
    AVAILABILITY: str = Field(..., example="Sale")  # Adjust as needed
    DATE: str = Field(..., example="2025-06-17")  # ISO date string

@app.post("/predict")
async def predict_price(features: HouseFeatures):
    try:
        # Parse date and extract year, month
        date_obj = datetime.fromisoformat(features.DATE)
        year = date_obj.year
        month = date_obj.month
        
        # Create DataFrame for preprocessing (order matters)
        input_df = pd.DataFrame([{
            "DIMENSIONS": features.DIMENSIONS,
            "BEDROOMS": features.BEDROOMS,
            "BATHROOMS": features.BATHROOMS,
            "TYPE": features.TYPE,
            "AVAILABILITY": features.AVAILABILITY,
            "YEAR": year,
            "MONTH": month
        }])
        
        # Transform features using pipeline
        X = preprocessor.transform(input_df)
        
        # Predict log price
        log_price_pred = model.predict(X)
        
        # Convert back from log scale
        price_pred = np.expm1(log_price_pred)[0][0]
        
        return {"predicted_price": round(float(price_pred), 2)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
