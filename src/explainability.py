
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Load original dataset
X = pd.read_csv(
    RAW_DIR / "secom.data",
    sep=r"\s+",
    header=None,
    na_values="NaN"
)

print("Dataset shape:", X.shape)

# Connect to MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")

MODEL_NAME = "Semiconductor_Quality_Model"
MODEL_VERSION = "1"

print("\nLoading registered MLflow model...")

model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
pipeline = mlflow.sklearn.load_model(model_uri)

print("Model loaded successfully!")

# Extract preprocessing steps and classifier
imputer = pipeline.named_steps["imputer"]
selector = pipeline.named_steps["variance"]
scaler = pipeline.named_steps["scaler"]
classifier = pipeline.named_steps["classifier"]

# Apply the same preprocessing used during training
X_imputed = imputer.transform(X)
X_selected = selector.transform(X_imputed)
X_scaled = scaler.transform(X_selected)

# Get original sensor feature names retained by VarianceThreshold
selected_features = X.columns[selector.get_support()]

X_processed = pd.DataFrame(
    X_scaled,
    columns=selected_features
)

print("Processed data shape:", X_processed.shape)

# Select background and explanation samples
background = X_processed.sample(
    n=min(100, len(X_processed)),
    random_state=42
)

explain_data = X_processed.sample(
    n=min(300, len(X_processed)),
    random_state=42
)

# SHAP explanation for Logistic Regression
print("\nCalculating SHAP values...")

explainer = shap.LinearExplainer(
    classifier,
    background
)

shap_values = explainer(explain_data)

print("SHAP calculation completed!")

# Summary plot
plt.figure()

shap.summary_plot(
    shap_values,
    explain_data,
    show=False,
    max_display=20
)

plt.title("SHAP Feature Importance - Semiconductor Quality")

plt.tight_layout()

plt.savefig(
    REPORT_DIR / "shap_summary.png",
    bbox_inches="tight",
    dpi=300
)

plt.show()

# Calculate mean absolute SHAP importance
importance = pd.DataFrame({
    "feature": selected_features,
    "mean_abs_shap": abs(shap_values.values).mean(axis=0)
})

importance = importance.sort_values(
    by="mean_abs_shap",
    ascending=False
)

# Save feature importance
importance.to_csv(
    REPORT_DIR / "shap_feature_importance.csv",
    index=False
)

print("\nTop 20 Important Sensor Features:")
print(importance.head(20).to_string(index=False))

print("\nSHAP explainability completed!")
print("Graph saved: reports/shap_summary.png")
print("Feature importance saved: reports/shap_feature_importance.csv")
