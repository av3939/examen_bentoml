# src/train_model.py

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import bentoml
import os

# Chargement des données
data_dir = "data/processed"
X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).values.ravel()
y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).values.ravel()

# Création du pipeline avec scaler + modèle
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])

# Entraînement du pipeline
pipeline.fit(X_train, y_train)

# Prédictions
y_pred = pipeline.predict(X_test)

# Évaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")

# Sauvegarde avec BentoML
bento_model = bentoml.sklearn.save_model(
    "admission_model",
    pipeline,
    metadata={"rmse": rmse, "r2": r2}
)

print(f"Modèle enregistré : {bento_model}")
