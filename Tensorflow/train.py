import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Define base directory
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "Dataset" / "NY-House-Dataset.csv"
PREPROCESSOR_PATH = BASE_DIR / "Tensorflow" / "preprocessor.pkl"
MODEL_SAVE_PATH = BASE_DIR / "Tensorflow" / "tensorflow_model"

# Load the dataset
df = pd.read_csv(DATA_PATH)

# Define features
numeric_features = ['PRICE', 'BEDS', 'BATH', 'PROPERTYSQFT']
categorical_features = ['TYPE', 'ADDRESS']
df = df[numeric_features + categorical_features].dropna()

# Create and fit the preprocessor
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
])

X_processed = preprocessor.fit_transform(df)

# Save the preprocessor
joblib.dump(preprocessor, PREPROCESSOR_PATH)

# Normalize embeddings for cosine similarity
norm_embeddings = X_processed / np.linalg.norm(X_processed, axis=1, keepdims=True)

# Define TensorFlow model
class RecommenderModel(tf.Module):
    def __init__(self, normalized_embeddings):
        super().__init__()
        self.embeddings = tf.constant(normalized_embeddings.astype(np.float32))

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.float32)])
    def recommend(self, query):
        query_norm = tf.nn.l2_normalize(query, axis=1)
        return tf.matmul(query_norm, tf.transpose(self.embeddings))  # Cosine similarity

# Save the model
model = RecommenderModel(norm_embeddings)
tf.saved_model.save(model, str(MODEL_SAVE_PATH))