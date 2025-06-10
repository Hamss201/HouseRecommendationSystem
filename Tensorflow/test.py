import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from pathlib import Path

# Define base directory
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "Dataset" / "NY-House-Dataset.csv"
PREPROCESSOR_PATH = BASE_DIR / "Tensorflow" / "preprocessor.pkl"
MODEL_PATH = BASE_DIR / "Tensorflow" / "tensorflow_model"

# Load dataset
df = pd.read_csv(DATA_PATH)

numeric_features = ['PRICE', 'BEDS', 'BATH', 'PROPERTYSQFT']
categorical_features = ['TYPE', 'ADDRESS']
df = df[numeric_features + categorical_features].dropna()

# Load preprocessor
preprocessor = joblib.load(PREPROCESSOR_PATH)
X_all = preprocessor.transform(df)

# Load TensorFlow model
loaded_model = tf.saved_model.load(str(MODEL_PATH))
recommender_fn = loaded_model.recommend

# Select a query sample
query_idx = 10
query = X_all[query_idx]
query = query.toarray() if hasattr(query, "toarray") else query
query = query.reshape(1, -1)

# Perform inference
similarities = recommender_fn(tf.constant(query, dtype=tf.float32)).numpy()[0]

# Get top 5 matches excluding self
top_indices = similarities.argsort()[-6:][::-1]
top_indices = top_indices[top_indices != query_idx][:5]

# Display results
print(f"\n Query Listing:\n{df.iloc[query_idx]}")
print("\n Top 5 Similar Properties:\n")
for idx in top_indices:
    print(df.iloc[idx])
    print("—" * 40)

print("\n Similarity scores of top matches:", similarities[top_indices])