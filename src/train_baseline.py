
from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    balanced_accuracy_score
)


# 1. Define project paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"


# 2. Load sensor data and labels
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

labels.columns = ["target", "timestamp"]


# 3. Validate the dataset
if len(X) != len(labels):
    raise ValueError("Sensor data and labels have different row counts.")

print("Sensor shape:", X.shape)
print("Label shape:", labels.shape)
print("\nTarget distribution:")
print(labels["target"].value_counts())


# 4. Separate features and target
# Timestamp is excluded from model features.
y = labels["target"]
X = X.copy()


# 5. Split BEFORE fitting preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# 6. Build preprocessing and model pipeline
pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),

    ("remove_constant", VarianceThreshold(threshold=0.0)),

    ("scaler", StandardScaler()),

    ("classifier", LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        solver="liblinear",
        random_state=42
    ))
])


# 7. Train the model
print("\nTraining baseline model...")
pipeline.fit(X_train, y_train)


# 8. Make predictions
y_pred = pipeline.predict(X_test)
y_scores = pipeline.predict_proba(X_test)[:, 1]


# 9. Evaluate the model
print("\n========== MODEL EVALUATION ==========")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    labels=[-1, 1],
    target_names=["Pass", "Fail"],
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=[-1, 1]))

print(
    "\nPR-AUC (Average Precision):",
    round(average_precision_score(y_test == 1, y_scores), 4)
)

print(
    "\nBalanced Accuracy:",
    round(balanced_accuracy_score(y_test, y_pred), 4)
)


# 10. Print the number of retained features
retained_features = pipeline.named_steps[
    "remove_constant"
].get_support().sum()

print("\nFeatures before preprocessing:", X.shape[1])
print("Features after removing constants:", retained_features)
print("\nBaseline training completed!")
