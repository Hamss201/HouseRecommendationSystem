from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from pathlib import Path
import os

app = FastAPI()

# Base directory (modify if needed)
BASE_DIR = Path(__file__).resolve().parent
PREPROCESSOR_PATH = os.getenv("PREPROCESSOR_PATH", "Tensorflow/preprocessor.pkl")
MODEL_PATH = os.getenv("MODEL_PATH", "Tensorflow/tensorflow_model")
DATA_PATH = os.getenv("DATA_PATH", "Dataset/House-Recommendation-Properties.csv")


# Load components
preprocessor = joblib.load(PREPROCESSOR_PATH)
model = tf.saved_model.load(str(MODEL_PATH))
df = pd.read_csv(DATA_PATH)

# Define training columns
numeric_features = ['PRICE', 'BEDROOMS', 'BATHROOMS', 'DIMENSIONS']
categorical_features = ['REALTOR', 'PROPERTYNAME', 'AVAILABILITY', 'TYPE', 'DESCRIPTION', 'ADDRESS', 'IMAGE']
df = df[numeric_features + categorical_features].dropna()

# Normalize dataset embeddings for comparison
processed_df = preprocessor.transform(df)
norm_embeddings = processed_df / np.linalg.norm(processed_df, axis=1, keepdims=True)

# Define request schema
class Listing(BaseModel):
    PRICE: float
    BEDROOMS: int
    BATHROOMS: int
    DIMENSIONS: float
    REALTOR: str
    PROPERTYNAME: str
    AVAILABILITY: str
    TYPE: str
    DESCRIPTION: str
    ADDRESS: str
    IMAGE: str

@app.post("/recommend/")
def recommend_property(listing: Listing, top_k: int = 5):
    try:
        # Prepare query input
        input_df = pd.DataFrame([listing.dict()])
        input_processed = preprocessor.transform(input_df)
        norm = np.linalg.norm(input_processed, axis=1, keepdims=True)
        
        if np.any(norm == 0):
            raise HTTPException(status_code=400, detail="Invalid input; normalization failed due to zero vector.")

        input_normalized = input_processed / norm

        # Inference using custom model method
        query_tensor = tf.constant(input_normalized.astype(np.float32))
        similarity_scores = model.recommend(query_tensor).numpy().flatten()

        # Top K similar indices
        top_indices = np.argsort(similarity_scores)[::-1][:top_k]
        top_matches = df.iloc[top_indices].copy()
        top_matches["similarity_score"] = similarity_scores[top_indices]
        
         # Sort by similarity score in descending order
        top_matches = top_matches.sort_values(by="similarity_score", ascending=False)

        # Return JSON
        return {"recommendations": top_matches.to_dict(orient="records")}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))