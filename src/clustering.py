
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORT_DIR = BASE_DIR / "reports"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

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


# --------------------------------------------------
# 3. HANDLE MISSING VALUES
# --------------------------------------------------

imputer = SimpleImputer(strategy="median")

X_imputed = imputer.fit_transform(X)

print("Shape after imputation:", X_imputed.shape)


# --------------------------------------------------
# 4. REMOVE CONSTANT FEATURES
# --------------------------------------------------

variance_filter = VarianceThreshold(threshold=0.0)

X_filtered = variance_filter.fit_transform(X_imputed)

print("Shape after removing constant features:", X_filtered.shape)


# --------------------------------------------------
# 5. FEATURE SCALING
# --------------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_filtered)

print("Shape after scaling:", X_scaled.shape)


# --------------------------------------------------
# 6. PCA DIMENSIONALITY REDUCTION
# --------------------------------------------------

pca = PCA(
    n_components=20,
    random_state=42
)

X_pca = pca.fit_transform(X_scaled)

explained_variance = pca.explained_variance_ratio_.sum()

print("\nPCA output shape:", X_pca.shape)

print(
    "Total explained variance:",
    round(explained_variance * 100, 2),
    "%"
)


# --------------------------------------------------
# 7. FIND BEST NUMBER OF CLUSTERS
# --------------------------------------------------

best_score = -1
best_k = None
best_model = None

print("\nEvaluating different cluster counts...")

for k in range(2, 9):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    cluster_labels = kmeans.fit_predict(X_pca)

    score = silhouette_score(
        X_pca,
        cluster_labels
    )

    print(f"K = {k}, Silhouette Score = {score:.4f}")

    if score > best_score:

        best_score = score
        best_k = k
        best_model = kmeans


# --------------------------------------------------
# 8. FINAL CLUSTER ASSIGNMENT
# --------------------------------------------------

cluster_labels = best_model.labels_

print("\nBest number of clusters:", best_k)
print("Best silhouette score:", round(best_score, 4))


# --------------------------------------------------
# 9. CREATE CLUSTER SUMMARY
# --------------------------------------------------

results = pd.DataFrame({
    "cluster": cluster_labels,
    "actual_quality": y
})

cluster_summary = results.groupby("cluster").agg(
    total_samples=("actual_quality", "count"),
    pass_samples=("actual_quality", lambda x: (x == 0).sum()),
    fail_samples=("actual_quality", lambda x: (x == 1).sum())
)

cluster_summary["failure_percentage"] = (
    cluster_summary["fail_samples"]
    / cluster_summary["total_samples"]
) * 100

print("\nCluster Summary:")
print(cluster_summary)


# --------------------------------------------------
# 10. SAVE CLUSTER RESULTS
# --------------------------------------------------

results["pca_1"] = X_pca[:, 0]
results["pca_2"] = X_pca[:, 1]

results.to_csv(
    PROCESSED_DIR / "cluster_results.csv",
    index=False
)

cluster_summary.to_csv(
    REPORT_DIR / "cluster_summary.csv"
)


# --------------------------------------------------
# 11. VISUALIZE CLUSTERS
# --------------------------------------------------

plt.figure(figsize=(10, 7))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=cluster_labels,
    cmap="viridis",
    s=25,
    alpha=0.7
)

plt.title("Semiconductor Operational-State Clusters")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.colorbar(label="Cluster")

plt.tight_layout()

plt.savefig(
    REPORT_DIR / "operational_clusters.png",
    dpi=300
)

plt.show()


print("\nClustering completed successfully!")
print("Cluster results saved.")

