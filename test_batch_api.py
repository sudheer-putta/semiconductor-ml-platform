import pandas as pd
import requests

# Load 10 actual samples (590 features each)
X = pd.read_csv(
    "data/raw/secom.data",
    sep=r"\s+",
    header=None,
    na_values="NaN"
)

# Convert missing values to JSON-compatible None
X_sample = X.head(10).astype(object)
X_sample = X_sample.where(pd.notna(X_sample), None)

# Prepare batch request
payload = {
    "features": X_sample.values.tolist()
}

# Send request to FastAPI
response = requests.post(
    "http://127.0.0.1:8000/predict-batch",
    json=payload
)

print("Status:", response.status_code)
print("Response:")
print(response.json())