import pandas as pd
import numpy as np
from pathlib import Path

# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"

# Load sensor data
X = pd.read_csv(
    DATA_DIR / "secom.data",
    sep=r"\s+",
    header=None,
    na_values="NaN"
)

# Load labels and timestamps
y = pd.read_csv(
    DATA_DIR / "secom_labels.data",
    sep=r"\s+",
    header=None
)

# Separate target and timestamp
y.columns = ["target", "timestamp"]

print("\n========== DATASET INFORMATION ==========")

print("\nSensor Data Shape:")
print(X.shape)

print("\nLabel Data Shape:")
print(y.shape)

print("\nFirst Five Rows:")
print(X.head())

print("\nDataset Information:")
print(X.info())

print("\nMissing Values:")
print(X.isnull().sum().sum())

print("\nMissing Values Percentage:")
print(
    round(
        X.isnull().sum().sum() / X.size * 100,
        2
    ),
    "%"
)

print("\nTarget Distribution:")
print(y["target"].value_counts())

print("\nTarget Percentage:")
print(
    y["target"].value_counts(normalize=True) * 100
)

print("\nDuplicate Rows:")
print(X.duplicated().sum())

print("\nConstant Features:")
print((X.nunique(dropna=True) <= 1).sum())