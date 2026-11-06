
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    precision_recall_curve,
    classification_report,
    confusion_matrix,
    average_precision_score,
    balanced_accuracy_score
)


# 1. Load data
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"

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

y = labels.iloc[:, 0]


# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Convert labels into binary form:
# Pass = 0, Fail = 1
y_train_binary = (y_train == 1).astype(int)
y_test_binary = (y_test == 1).astype(int)


# 3. Build Random Forest pipeline
rf_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("remove_constant", VarianceThreshold()),

    ("classifier", RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ))
])


# 4. Generate out-of-fold probabilities on training data
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("Generating out-of-fold predictions...")

oof_probabilities = cross_val_predict(
    rf_pipeline,
    X_train,
    y_train_binary,
    cv=cv,
    method="predict_proba",
    n_jobs=-1
)[:, 1]


# 5. Calculate threshold metrics
precision, recall, thresholds = precision_recall_curve(
    y_train_binary,
    oof_probabilities
)

# Last precision and recall values have no corresponding threshold.
precision = precision[:-1]
recall = recall[:-1]

f2_scores = (
    5 * precision * recall
    / (4 * precision + recall + 1e-10)
)

results = pd.DataFrame({
    "Threshold": thresholds,
    "Precision": precision,
    "Recall": recall,
    "F2 Score": f2_scores
})

# Display the best threshold candidates
top_results = results.sort_values(
    "F2 Score",
    ascending=False
).head(10)

print("\n========== TOP THRESHOLDS ==========")
print(top_results.to_string(index=False))


# 6. Select threshold using F2 score
best_index = np.argmax(f2_scores)
best_threshold = thresholds[best_index]

print("\nSelected threshold:", round(best_threshold, 4))


# 7. Train final candidate on all training data
rf_pipeline.fit(X_train, y_train_binary)

# Predict probabilities on untouched test data
test_probabilities = rf_pipeline.predict_proba(X_test)[:, 1]

# Apply selected threshold
test_predictions = (
    test_probabilities >= best_threshold
).astype(int)


# 8. Final evaluation
print("\n========== TEST SET EVALUATION ==========")

print("\nClassification Report:")
print(classification_report(
    y_test_binary,
    test_predictions,
    target_names=["Pass", "Fail"],
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test_binary, test_predictions))

print(
    "\nTest PR-AUC:",
    round(
        average_precision_score(
            y_test_binary,
            test_probabilities
        ),
        4
    )
)

print(
    "\nTest Balanced Accuracy:",
    round(
        balanced_accuracy_score(
            y_test_binary,
            test_predictions
        ),
        4
    )
)
