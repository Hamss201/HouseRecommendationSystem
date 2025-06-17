import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2
import joblib

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "dataset" / "House-Price-Prediction-Properties-Expanded.csv"
MODEL_DIR = BASE_DIR / "tensorflow"
MODEL_PATH = MODEL_DIR / "price_model.keras"
SCALER_PATH = MODEL_DIR / "price_pipeline.pkl"

# Load and clean data
df = pd.read_csv(DATA_PATH)

# Convert DATE to datetime and extract year and month
df['DATE'] = pd.to_datetime(df['DATE'])
df['YEAR'] = df['DATE'].dt.year
df['MONTH'] = df['DATE'].dt.month

# Keep relevant columns and drop NA
df = df[["DIMENSIONS", "BEDROOMS", "BATHROOMS", "TYPE", "AVAILABILITY", "YEAR", "MONTH", "PRICE"]].dropna()

# Features and target
X = df.drop(columns=["PRICE"])
y = np.log1p(df["PRICE"])  # Log transform target

# Column types
numeric_features = ["DIMENSIONS", "BEDROOMS", "BATHROOMS", "YEAR", "MONTH"]
categorical_features = ["TYPE", "AVAILABILITY"]

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ("num", MinMaxScaler(), numeric_features),
    ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), categorical_features)
])

# Fit and transform features
X_transformed = preprocessor.fit_transform(X)

# Save preprocessor for later use
joblib.dump(preprocessor, SCALER_PATH)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

# Build model with regularization and dropout
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=l2(0.001)),
    Dropout(0.3),
    Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.3),
    Dense(1)  # Predict log-price
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train model
model.fit(X_train, y_train, epochs=100, validation_split=0.1, verbose=1)

# Save model
model.save(MODEL_PATH)
