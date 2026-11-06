

from pathlib import Path
import os

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import requests


# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

REPORTS_DIR = BASE_DIR / "reports"
RAW_DIR = BASE_DIR / "data" / "raw"

MODEL_NAME = "Semiconductor_Quality_Model"
MODEL_VERSION = "1"

API_URL = os.getenv("API_URL", "http://api:8000/predict-batch")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://host.docker.internal:5000")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

st.set_page_config(
    page_title="Semiconductor Quality Platform",
    page_icon="🏭",
    layout="wide"
)


# --------------------------------------------------
# 2. HELPER FUNCTIONS
# --------------------------------------------------

def find_report(prefix):
    """Find a report file using its filename prefix."""

    matches = list(REPORTS_DIR.glob(f"{prefix}*"))

    return matches[0] if matches else None


def load_csv(prefix):
    """Load a CSV report if it exists."""

    path = find_report(prefix)

    if path is None:
        return None

    try:
        return pd.read_csv(path)

    except Exception:
        return None


def show_image(prefix, caption):
    """Display an existing report image."""

    path = find_report(prefix)

    if path and path.suffix.lower() in [".png", ".jpg", ".jpeg"]:

        st.image(
            str(path),
            caption=caption,
            use_container_width=True
        )

    else:

        st.info(
            f"{caption} is not available in the reports folder."
        )


@st.cache_data
def load_dataset():
    """Load the original SECOM sensor dataset."""

    data_path = RAW_DIR / "secom.data"
    label_path = RAW_DIR / "secom_labels.data"

    if not data_path.exists() or not label_path.exists():

        return None, None

    X = pd.read_csv(
        data_path,
        sep=r"\s+",
        header=None,
        na_values="NaN"
    )

    labels = pd.read_csv(
        label_path,
        sep=r"\s+",
        header=None
    )

    y = labels.iloc[:, 0].map({
        -1: 0,
        1: 1
    })

    return X, y


def predict_with_api(input_data):
    """Send all uploaded sensor samples to FastAPI in one batch request."""

    total_samples = len(input_data)

    if total_samples == 0:
        raise ValueError("The uploaded CSV contains no samples.")

    if input_data.shape[1] != 590:
        raise ValueError(
            f"Expected 590 sensor features, received {input_data.shape[1]}."
        )

    status_text = st.empty()
    progress_bar = st.progress(0)

    status_text.write(
        f"Preparing {total_samples} sensor samples for batch prediction..."
    )

    # Convert missing values to JSON null and numeric values to float.
    samples = [
        [
            None if pd.isna(value) else float(value)
            for value in row
        ]
        for row in input_data.itertuples(index=False, name=None)
    ]

    status_text.write(
        f"Sending {total_samples} samples to the FastAPI batch endpoint..."
    )
    progress_bar.progress(0.5)

    try:
        response = requests.post(
            API_URL,
            json={"features": samples},
            timeout=180
        )
        response.raise_for_status()
        result = response.json()

        predictions = [int(value) for value in result["predictions"]]
        fail_probabilities = [
            float(value) for value in result["fail_probabilities"]
        ]

        if len(predictions) != total_samples:
            raise ValueError(
                "The API prediction count does not match the uploaded sample count."
            )

        if len(fail_probabilities) != total_samples:
            raise ValueError(
                "The API probability count does not match the uploaded sample count."
            )

        progress_bar.progress(1.0)
        status_text.success(
            f"Successfully predicted all {total_samples} samples in one batch request."
        )

        return predictions, fail_probabilities

    except requests.exceptions.RequestException as error:
        status_text.empty()
        progress_bar.empty()
        raise RuntimeError(f"FastAPI batch request failed: {error}") from error

    except (KeyError, TypeError, ValueError) as error:
        status_text.empty()
        progress_bar.empty()
        raise RuntimeError(f"Invalid batch prediction response: {error}") from error

# --------------------------------------------------
# 3. APPLICATION HEADER
# --------------------------------------------------

st.title("🏭 Semiconductor Manufacturing Intelligence")

st.caption(
    "Machine Learning | Quality Prediction | "
    "Operational Clustering | Anomaly Monitoring | "
    "Data Drift Monitoring"
)

st.divider()


# --------------------------------------------------
# 4. LOAD DATA AND MODEL REPORTS
# --------------------------------------------------

X, y = load_dataset()

cluster_data = load_csv("cluster_assignments")

cluster_summary = load_csv("cluster_summary")

anomaly_data = load_csv("anomaly_detection")


# --------------------------------------------------
# 5. SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("Platform Navigation")

page = st.sidebar.radio(
    "Select Module",
    [
        "Overview",
        "Quality Prediction",
        "Operational Clusters",
        "Anomaly Monitoring",
        "Model Explainability",
        "Data Drift Monitoring"
    ]
)

st.sidebar.divider()

st.sidebar.write("**Model:** Logistic Regression")

st.sidebar.write("**Registry:** MLflow")

st.sidebar.write("**Model Version:** 1")

st.sidebar.write("**Dataset:** UCI SECOM")

st.sidebar.write("**Backend:** FastAPI")


# --------------------------------------------------
# 6. OVERVIEW
# --------------------------------------------------

if page == "Overview":

    st.header("Platform Overview")

    if X is not None:

        total_samples = len(X)

        total_features = X.shape[1]

        pass_samples = int((y == 0).sum())

        fail_samples = int((y == 1).sum())

        # ------------------------------------------
        # DATASET METRICS
        # ------------------------------------------

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Total Samples",
            total_samples
        )

        col2.metric(
            "Sensor Features",
            total_features
        )

        col3.metric(
            "Pass Samples",
            pass_samples
        )

        col4.metric(
            "Fail Samples",
            fail_samples
        )

        # ------------------------------------------
        # QUALITY DISTRIBUTION
        # ------------------------------------------

        st.subheader("Quality Distribution")

        distribution = pd.DataFrame({
            "Quality": ["Pass", "Fail"],
            "Samples": [
                pass_samples,
                fail_samples
            ]
        })

        fig, ax = plt.subplots(figsize=(8, 4))

        bars = ax.bar(
            distribution["Quality"],
            distribution["Samples"]
        )

        ax.set_title("Pass vs Fail Distribution")

        ax.set_xlabel("Quality")

        ax.set_ylabel("Number of Samples")

        ax.set_ylim(0, 1600)

        ax.grid(
            axis="y",
            alpha=0.25
        )

        ax.set_axisbelow(True)

        for bar in bars:

            height = bar.get_height()

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 20,
                f"{int(height)}",
                ha="center"
            )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

        # ------------------------------------------
        # DATASET PREVIEW
        # ------------------------------------------

        st.subheader("Dataset Preview")

        st.dataframe(
            X.head(10),
            use_container_width=True
        )

    else:

        st.warning(
            "Original SECOM dataset files were not found."
        )

    st.divider()

    # ----------------------------------------------
    # MODEL PERFORMANCE FROM MLFLOW
    # ----------------------------------------------

    st.subheader("Model Performance")

    try:

        client = mlflow.tracking.MlflowClient()

        experiments = client.search_experiments(
            filter_string=(
                "name = 'Semiconductor_Quality_Prediction'"
            )
        )

        if experiments:

            runs = client.search_runs(
                experiment_ids=[
                    experiments[0].experiment_id
                ],
                order_by=[
                    "attributes.start_time DESC"
                ],
                max_results=1
            )

            if runs:

                metrics = runs[0].data.metrics

                metric_names = [
                    ("Precision", "precision"),
                    ("Recall", "recall"),
                    ("F1 Score", "f1_score"),
                    ("PR-AUC", "pr_auc"),
                    ("Balanced Accuracy", "balanced_accuracy")
                ]

                columns = st.columns(5)

                for column, (label, key) in zip(
                    columns,
                    metric_names
                ):

                    value = metrics.get(key)

                    if value is not None:

                        column.metric(
                            label,
                            f"{value:.3f}"
                        )

                    else:

                        column.metric(
                            label,
                            "N/A"
                        )

            else:

                st.info(
                    "No MLflow training runs found."
                )

        else:

            st.info(
                "MLflow experiment not found."
            )

    except Exception as error:

        st.warning(
            f"Could not connect to MLflow: {error}"
        )


# --------------------------------------------------
# 7. QUALITY PREDICTION - FASTAPI INTEGRATION
# --------------------------------------------------

elif page == "Quality Prediction":

    st.header("Semiconductor Quality Prediction")

    st.write(
        "Upload a CSV containing the original 590 sensor features. "
        "The FastAPI backend uses the registered MLflow model "
        "to predict semiconductor quality."
    )

    st.info(
        "Upload sensor data with exactly 590 feature columns, "
        "in the same order used during model training. "
        "Do not include target or timestamp columns."
    )

    uploaded_file = st.file_uploader(
        "Upload sensor data CSV",
        type=["csv"],
        key="quality_prediction_upload"
    )

    if uploaded_file is not None:

        try:

            input_data = pd.read_csv(uploaded_file)

            st.subheader("Uploaded Data")

            st.write(
                f"Samples: {len(input_data)} | "
                f"Sensor features: {input_data.shape[1]}"
            )

            with st.expander("Preview uploaded sensor data"):

                st.dataframe(
                    input_data.head(10),
                    use_container_width=True
                )

            if input_data.empty:

                st.error("The uploaded CSV contains no samples.")

            elif input_data.shape[1] != 590:

                st.error(
                    f"Expected 590 sensor columns, "
                    f"but received {input_data.shape[1]}."
                )

            elif st.button(
                "Predict Quality",
                type="primary",
                key="run_quality_prediction"
            ):

                with st.spinner("Sending sensor data to FastAPI..."):

                    predictions, fail_probabilities = (
                        predict_with_api(input_data)
                    )

                # Build a compact results table instead of showing
                # all 590 sensor columns in the prediction results.
                results = pd.DataFrame({
                    "Sample ID": np.arange(1, len(predictions) + 1),
                    "Prediction": np.where(
                        np.array(predictions) == 1,
                        "Fail",
                        "Pass"
                    ),
                    "Fail Probability (%)": (
                        np.array(fail_probabilities, dtype=float) * 100
                    ).round(4)
                })

                st.success(
                    "Quality prediction completed successfully!"
                )

                st.subheader("Prediction Summary")

                fail_count = int(np.sum(np.array(predictions) == 1))
                pass_count = int(np.sum(np.array(predictions) == 0))
                total_count = len(predictions)

                col1, col2, col3 = st.columns(3)

                col1.metric("Total Samples", total_count)
                col2.metric("Predicted Pass", pass_count)
                col3.metric("Predicted Fail", fail_count)

                st.subheader("Prediction Results")

                st.dataframe(
                    results,
                    column_config={
                        "Sample ID": st.column_config.NumberColumn(
                            "Sample ID",
                            format="%d"
                        ),
                        "Prediction": st.column_config.TextColumn(
                            "Prediction"
                        ),
                        "Fail Probability (%)": st.column_config.ProgressColumn(
                            "Fail Probability",
                            help="Model-estimated probability of the Fail class.",
                            format="%.2f%%",
                            min_value=0,
                            max_value=100
                        )
                    },
                    hide_index=True,
                    use_container_width=True
                )

                # Visual comparison of estimated fail probability
                st.subheader("Fail Probability by Sample")

                probability_chart = results.set_index("Sample ID")[
                    ["Fail Probability (%)"]
                ]

                st.bar_chart(
                    probability_chart,
                    use_container_width=True
                )

                # Model information
                st.subheader("Model Information")

                info_col1, info_col2, info_col3 = st.columns(3)

                info_col1.write(f"**Model:** {MODEL_NAME}")
                info_col2.write(f"**Model Version:** {MODEL_VERSION}")
                info_col3.write("**Backend:** FastAPI")

                # Download compact prediction report
                csv = results.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "Download Prediction Report",
                    data=csv,
                    file_name="quality_prediction_results.csv",
                    mime="text/csv"
                )

                st.caption(
                    "Fail probabilities are model estimates, not "
                    "confirmed defect rates. Predictions depend on the "
                    "sensor columns being in the correct training order."
                )

        except pd.errors.EmptyDataError:

            st.error("The uploaded CSV file is empty.")

        except requests.exceptions.ConnectionError:

            st.error(
                "Cannot connect to FastAPI. "
                "Make sure the API server is running."
            )

        except requests.exceptions.RequestException as error:

            st.error(f"FastAPI request failed: {error}")

        except Exception as error:

            st.error(f"Prediction failed: {error}")


# --------------------------------------------------
# 8. OPERATIONAL CLUSTERS
# --------------------------------------------------

elif page == "Operational Clusters":

    st.header("Operational-State Clustering")

    st.write(
        "Explore the operational groups discovered "
        "using unsupervised machine learning."
    )

    show_image(
        "operational_clusters",
        "PCA Operational-State Clusters"
    )

    if cluster_summary is not None:

        st.subheader("Cluster Summary")

        st.dataframe(
            cluster_summary,
            use_container_width=True
        )

    else:

        st.info(
            "Cluster summary CSV was not found."
        )

    if cluster_data is not None:

        st.subheader("Cluster Assignments")

        st.dataframe(
            cluster_data.head(50),
            use_container_width=True
        )

        st.download_button(
            "Download Cluster Assignments",
            data=cluster_data.to_csv(index=False),
            file_name="cluster_assignments.csv",
            mime="text/csv"
        )

    else:

        st.info(
            "Cluster assignment CSV was not found."
        )


# --------------------------------------------------
# 9. ANOMALY MONITORING
# --------------------------------------------------

elif page == "Anomaly Monitoring":

    st.header("Anomaly Detection and Monitoring")

    st.write(
        "Review unusual sensor patterns detected "
        "by the Isolation Forest model."
    )

    show_image(
        "anomaly_score_distribution",
        "Anomaly Score Distribution"
    )

    show_image(
        "anomaly_visualization",
        "Anomaly Visualization"
    )

    if anomaly_data is not None:

        st.subheader("Anomaly Detection Results")

        st.dataframe(
            anomaly_data.head(100),
            use_container_width=True
        )

        st.download_button(
            "Download Anomaly Results",
            data=anomaly_data.to_csv(index=False),
            file_name="anomaly_results.csv",
            mime="text/csv"
        )

    else:

        st.info(
            "Anomaly detection CSV was not found. "
            "Check the reports folder."
        )


# --------------------------------------------------
# 10. MODEL EXPLAINABILITY
# --------------------------------------------------

elif page == "Model Explainability":

    st.header("Model Explainability")

    st.write(
        "SHAP explains how individual sensor features "
        "influence the model's predictions."
    )

    show_image(
        "shap_summary",
        "SHAP Feature Importance Summary"
    )

    st.info(
        "Pink indicates higher feature values; "
        "blue indicates lower feature values. "
        "Positive and negative SHAP values indicate "
        "the direction of influence on the explained model output."
    )


# --------------------------------------------------
# 11. DATA DRIFT MONITORING
# --------------------------------------------------

elif page == "Data Drift Monitoring":

    st.header("Data Drift Monitoring")

    st.write(
        "Monitor changes in sensor feature distributions "
        "using Evidently AI."
    )

    drift_html = (
        REPORTS_DIR / "data_drift_report.html"
    )

    drift_json = (
        REPORTS_DIR / "data_drift_report.json"
    )

    if drift_html.exists():

        st.success(
            "Evidently drift report loaded successfully!"
        )

        # ------------------------------------------
        # DRIFT SUMMARY
        # ------------------------------------------

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Sensor Features",
            "474"
        )

        col2.metric(
            "Drifted Features",
            "389"
        )

        col3.metric(
            "Drift Percentage",
            "82.07%"
        )

        st.divider()

        # ------------------------------------------
        # EVIDENTLY REPORT
        # ------------------------------------------

        st.subheader(
            "Evidently Data Drift Report"
        )

        html_content = drift_html.read_text(
            encoding="utf-8"
        )

        components.html(
            html_content,
            height=1200,
            scrolling=True
        )

        st.divider()

        # ------------------------------------------
        # DOWNLOAD REPORTS
        # ------------------------------------------

        col1, col2 = st.columns(2)

        with col1:

            st.download_button(
                "Download HTML Report",
                data=drift_html.read_bytes(),
                file_name="data_drift_report.html",
                mime="text/html"
            )

        if drift_json.exists():

            with col2:

                st.download_button(
                    "Download JSON Report",
                    data=drift_json.read_bytes(),
                    file_name="data_drift_report.json",
                    mime="application/json"
                )

        st.info(
            "This report compares the first 80% "
            "and remaining 20% of the historical SECOM "
            "dataset. It is not live production monitoring."
        )

    else:

        st.warning(
            "Data drift report not found. "
            "Run python src/data_drift.py first."
        )


# --------------------------------------------------
# 12. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Semiconductor Manufacturing Intelligence Platform | "
    "Built with Python, Scikit-learn, MLflow, "
    "FastAPI, Evidently AI and Streamlit"
)
