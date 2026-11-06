# Semiconductor Manufacturing Intelligence Platform

An end-to-end machine learning and MLOps project for semiconductor manufacturing quality prediction, anomaly detection, operational clustering, model explainability, and data drift monitoring.

## Project Overview

This platform uses the UCI SECOM semiconductor manufacturing dataset to explore sensor measurements and identify patterns associated with manufacturing failures. It combines machine learning, experiment tracking, model serving, and an interactive monitoring dashboard.

## Key Features

- **Quality Prediction:** Predicts Pass/Fail outcomes using a trained Logistic Regression model.
- **Operational Clustering:** Groups sensor samples using PCA and clustering techniques.
- **Anomaly Detection:** Identifies unusual sensor patterns using unsupervised learning.
- **Model Explainability:** Uses SHAP to analyze influential sensor features.
- **Data Drift Monitoring:** Compares historical dataset batches using Evidently.
- **Experiment Tracking:** Tracks model experiments and registers models using MLflow.
- **REST API:** Serves predictions through FastAPI, including batch prediction.
- **Interactive Dashboard:** Displays model results and monitoring insights using Streamlit.
- **Docker Deployment:** Runs the API and dashboard using Docker Compose.

## Technology Stack

Python | Pandas | NumPy | Scikit-learn | MLflow | Evidently | SHAP | FastAPI | Streamlit | Docker | Git

## Architecture

```text
UCI SECOM Dataset
        |
        v
Data Preprocessing & EDA
        |
        v
Machine Learning Model
        |
        v
MLflow Experiment Tracking
        |
        v
MLflow Model Registry
        |
        v
FastAPI Prediction Service
        |
        v
Streamlit Monitoring Dashboard
        |
        v
Docker Compose Deployment
```

Additional modules support clustering, anomaly detection, explainability, and drift analysis.

## Project Structure

```text
semiconductor-ml-platform/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── api.py
│   ├── download_data.py
│   ├── eda.py
│   ├── train_with_mlflow.py
│   ├── clustering.py
│   ├── anomaly_detection.py
│   ├── data_drift.py
│   ├── explainability.py
│   └── threshold_tuning.py
│
├── reports/
├── models/
├── notebooks/
├── app.py
├── Dockerfile.api
├── Dockerfile.streamlit
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/sudheer-putta/semiconductor-ml-platform.git
cd semiconductor-ml-platform
```

### 2. Create a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the application using Docker

```bash
docker compose up --build -d
```

### 5. Access the services

- Streamlit Dashboard: http://localhost:8501
- FastAPI Documentation: http://localhost:8000/docs
- API Health Check: http://localhost:8000/health

The API requires access to the MLflow tracking server and registered model artifacts. Configure the MLflow tracking URI and artifact access for your environment before starting the containers.

## Dataset

This project uses the **SECOM dataset** from the UCI Machine Learning Repository.

- 1,567 manufacturing samples
- 590 sensor features
- 1,463 Pass samples
- 104 Fail samples

Dataset: https://archive.ics.uci.edu/dataset/179/secom

The dataset is highly imbalanced. Model predictions should be interpreted with this limitation in mind.

## Important Notes

- The dataset contains anonymized sensor measurements.
- The clustering and anomaly detection outputs are analytical results, not confirmed manufacturing process states or defect labels.
- Drift analysis compares historical batches from the available dataset; it is not live production monitoring.
- Model predictions are estimates and should not be treated as confirmed manufacturing decisions.
- Performance on this dataset does not establish production readiness for a real semiconductor fab.

## Author

**Sudheer Putta**

GitHub: https://github.com/sudheer-putta
