# src/debug_columns.py

import pandas as pd

df = pd.read_csv("data/raw/admission.csv")
df.columns = df.columns.str.strip()
