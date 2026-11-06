
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report
)


# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"


# ==========================================
# 2. LOAD DATASET
# ==========================================

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

# Convert labels:
# -1 = Pass (0)
#  1 = Fail (1)

y = labels.iloc[:, 0].map({
    -1: 0,
    1: 1
})

print("Dataset shape:", X.shape)
print("Target distribution:")
print(y.value_counts())


# ==========================================
# 3. TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ==========================================
# 4. BUILD ML PIPELINE
# ==========================================

pipeline = Pipeline([

    ("imputer", SimpleImputer(
        strategy="median"
    )),

    ("variance", VarianceThreshold(
        threshold=0.0
    )),

    ("scaler", StandardScaler()),

    ("classifier", LogisticRegression(
        class_weight="balanced",
        max_iter=5000,
        random_state=42
    ))

])


# ==========================================
# 5. CONFIGURE MLFLOW
# ==========================================

mlflow.set_tracking_uri(
    "http://127.0.0.1:5000"
)

mlflow.set_experiment(
    "Semiconductor_Quality_Prediction"
)


# ==========================================
# 6. START MLFLOW RUN
# ==========================================

with mlflow.start_run(
    run_name="Logistic_Regression_v1"
):

    # Train model
    print("\nTraining model...")

    pipeline.fit(
        X_train,
        y_train
    )

    # Predictions
    predictions = pipeline.predict(X_test)

    probabilities = pipeline.predict_proba(X_test)[:, 1]


    # ======================================
    # 7. MODEL EVALUATION
    # ======================================

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities
    )

    balanced_acc = balanced_accuracy_score(
        y_test,
        predictions
    )

    cm = confusion_matrix(
        y_test,
        predictions
    )


    # ======================================
    # 8. LOG PARAMETERS
    # ======================================

    mlflow.log_params({

        "model": "Logistic Regression",
        "class_weight": "balanced",
        "max_iter": 5000,
        "test_size": 0.20,
        "random_state": 42,
        "imputation": "median",
        "feature_selection": "VarianceThreshold",
        "scaling": "StandardScaler"

    })


    # ======================================
    # 9. LOG METRICS
    # ======================================

    mlflow.log_metrics({

        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "pr_auc": pr_auc,
        "balanced_accuracy": balanced_acc

    })


    # ======================================
    # 10. LOG CONFUSION MATRIX
    # ======================================

    mlflow.log_dict(
        {
            "confusion_matrix": cm.tolist(),
            "labels": ["Pass", "Fail"]
        },
        "evaluation/confusion_matrix.json"
    )


    # ======================================
    # 11. SAVE MODEL IN MLFLOW
    # ======================================

    print("\nLogging model to MLflow...")

    model_info = mlflow.sklearn.log_model(

        sk_model=pipeline,

        name="semiconductor_quality_model",

        skops_trusted_types=["numpy.dtype"]

    )


    # ======================================
    # 12. REGISTER MODEL
    # ======================================

    print("\nRegistering model...")

    registered_model = mlflow.register_model(

        model_uri=model_info.model_uri,

        name="Semiconductor_Quality_Model"

    )


    # ======================================
    # 13. DISPLAY RESULTS
    # ======================================

    print("\n========== MODEL RESULTS ==========")

    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("PR-AUC:", pr_auc)
    print("Balanced Accuracy:", balanced_acc)

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Pass", "Fail"],
            zero_division=0
        )
    )

    print("\n========== MLFLOW REGISTRY ==========")

    print("Model registered successfully!")
    print("Model name:", registered_model.name)
    print("Model version:", registered_model.version)

    print("\nRun ID:", mlflow.active_run().info.run_id)

    print("\nMLflow Experiment Completed!")
