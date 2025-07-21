# src/prepare_data.py

import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Chemin vers les données brutes
raw_data_path = "data/raw/admission.csv"
processed_data_dir = "data/processed"

# Chargement des données
df = pd.read_csv(raw_data_path)
df.columns = df.columns.str.strip()

# Suppression de colonnes inutiles
if "Serial No." in df.columns:
    df = df.drop(columns=["Serial No."])

# Séparation des variables explicatives et de la cible
X = df.drop(columns=["Chance of Admit"])
y = df["Chance of Admit"]

# Division en jeu d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Création du dossier si non existant
os.makedirs(processed_data_dir, exist_ok=True)

# Sauvegarde
X_train.to_csv(os.path.join(processed_data_dir, "X_train.csv"), index=False)
X_test.to_csv(os.path.join(processed_data_dir, "X_test.csv"), index=False)
y_train.to_csv(os.path.join(processed_data_dir, "y_train.csv"), index=False)
y_test.to_csv(os.path.join(processed_data_dir, "y_test.csv"), index=False)

print("Données préparées et sauvegardées sans normalisation.")
