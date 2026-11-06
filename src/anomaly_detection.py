
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORT_DIR = BASE_DIR / "reports"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Load dataset
X = pd.read_csv(
    RAW_DIR / "secom.data",
    sep=r"\s+",
    header=None,
    na_values="NaN"
)

labels = pd.read_csv(
    RAW_DIR / "secom_labels.data",
    sep=r"\s+",
    header=None
)

y = labels.iloc[:, 0].map({-1: 0, 1: 1})

print("Dataset shape:", X.shape)

# Preprocessing
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

selector = VarianceThreshold(threshold=0.0)
X_selected = selector.fit_transform(X_imputed)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

print("Processed shape:", X_scaled.shape)

# Train Isolation Forest
model = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=42,
    n_jobs=-1
)

model.fit(X_scaled)

# Predict anomalies
# Isolation Forest: -1 = anomaly, 1 = normal
anomaly_predictions = model.predict(X_scaled)

anomaly_labels = (anomaly_predictions == -1).astype(int)

# Calculate anomaly scores
anomaly_scores = -model.decision_function(X_scaled)

# Save results
results = pd.DataFrame({
    "actual_quality": y,
    "anomaly_prediction": anomaly_labels,
    "anomaly_score": anomaly_scores
})

results.to_csv(
    PROCESSED_DIR / "anomaly_results.csv",
    index=False
)

# Compare with actual failure labels
print("\nAnomaly Distribution:")
print(results["anomaly_prediction"].value_counts())

print("\nActual Failure vs Anomaly Detection:")
print(pd.crosstab(
    results["actual_quality"],
    results["anomaly_prediction"],
    rownames=["Actual Quality"],
    colnames=["Anomaly Prediction"]
))

print("\nClassification Report:")
print(classification_report(
    y,
    anomaly_labels,
    target_names=["Pass", "Fail"],
    zero_division=0
))

# Visualize anomaly scores
plt.figure(figsize=(10, 6))

plt.hist(anomaly_scores, bins=50)

plt.title("Semiconductor Anomaly Score Distribution")
plt.xlabel("Anomaly Score")
plt.ylabel("Number of Samples")

plt.tight_layout()
plt.savefig(REPORT_DIR / "anomaly_score_distribution.png")
plt.show()

print("\nAnomaly detection completed!")
print("Results saved to data/processed/anomaly_results.csv")

