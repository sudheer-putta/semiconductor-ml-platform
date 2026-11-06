
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


# =====================================================
# 1. PROJECT PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw"
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# 2. LOAD SEMICONDUCTOR DATASET
# =====================================================

print("\nLoading semiconductor dataset...")

X = pd.read_csv(
    RAW_DIR / "secom.data",
    sep=r"\s+",
    header=None,
    na_values="NaN"
)

print("\nOriginal Dataset Shape:", X.shape)


# =====================================================
# 3. REMOVE CONSTANT FEATURES
# =====================================================

print("\nRemoving constant features...")

# Median imputation
imputer = SimpleImputer(strategy="median")

X_imputed = imputer.fit_transform(X)

# Remove features with zero variance
variance_filter = VarianceThreshold(threshold=0.0)

X_filtered = variance_filter.fit_transform(X_imputed)

print("Shape after removing constant features:", X_filtered.shape)


# =====================================================
# 4. FEATURE SCALING
# =====================================================

print("\nScaling features...")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_filtered)

print("Scaled Dataset Shape:", X_scaled.shape)


# =====================================================
# 5. FIND A SUITABLE NUMBER OF CLUSTERS
# =====================================================

print("\nEvaluating different cluster counts...")

silhouette_results = {}

for k in range(2, 7):

    kmeans_temp = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20
    )

    temp_labels = kmeans_temp.fit_predict(X_scaled)

    score = silhouette_score(
        X_scaled,
        temp_labels,
        sample_size=min(1000, len(X_scaled)),
        random_state=42
    )

    silhouette_results[k] = score

    print(f"K = {k}, Silhouette Score = {score:.4f}")


# Select the cluster count with the highest score
best_k = max(
    silhouette_results,
    key=silhouette_results.get
)

print("\nBest Number of Clusters:", best_k)


# =====================================================
# 6. FINAL K-MEANS MODEL
# =====================================================

print("\nTraining final K-Means model...")

kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=20
)

cluster_labels = kmeans.fit_predict(X_scaled)

print("Clustering completed!")


# =====================================================
# 7. CLUSTER DISTRIBUTION
# =====================================================

print("\n========== CLUSTER DISTRIBUTION ==========")

cluster_counts = pd.Series(
    cluster_labels
).value_counts().sort_index()

print("\nNumber of observations in each cluster:")
print(cluster_counts)

cluster_percentages = (
    cluster_counts / len(cluster_labels) * 100
).round(2)

print("\nCluster percentages:")
print(cluster_percentages)


# =====================================================
# 8. CLUSTER EVALUATION
# =====================================================

print("\n========== CLUSTER EVALUATION ==========")

final_silhouette = silhouette_score(
    X_scaled,
    cluster_labels,
    sample_size=min(1000, len(X_scaled)),
    random_state=42
)

print(
    "\nFinal Silhouette Score:",
    round(final_silhouette, 4)
)

print(
    "Inertia:",
    round(kmeans.inertia_, 2)
)


# =====================================================
# 9. PCA DIMENSIONALITY REDUCTION
# =====================================================

print("\nApplying PCA...")

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

explained_variance = pca.explained_variance_ratio_

print("\nExplained Variance Ratio:")
print(explained_variance)

print(
    "Total Variance Explained:",
    round(explained_variance.sum() * 100, 2),
    "%"
)


# =====================================================
# 10. PCA CLUSTER VISUALIZATION
# =====================================================

print("\nGenerating cluster visualization...")

plt.figure(figsize=(12, 8))

scatter = plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=cluster_labels,
    cmap="viridis",
    s=35,
    alpha=0.7
)

plt.title("Semiconductor Operational-State Clusters")

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.colorbar(
    scatter,
    label="Cluster"
)

plt.grid(alpha=0.2)

plt.tight_layout()

plot_path = REPORT_DIR / "operational_state_clusters.png"

plt.savefig(plot_path, dpi=300)

plt.show()

print("Cluster plot saved:", plot_path)


# =====================================================
# 11. SAVE CLUSTER ASSIGNMENTS
# =====================================================

cluster_results = pd.DataFrame({
    "sample_id": np.arange(len(cluster_labels)),
    "cluster": cluster_labels
})

cluster_path = REPORT_DIR / "cluster_assignments.csv"

cluster_results.to_csv(
    cluster_path,
    index=False
)

print("\nCluster assignments saved:", cluster_path)


# =====================================================
# 12. SAVE CLUSTER SUMMARY
# =====================================================

summary = pd.DataFrame({
    "cluster": cluster_counts.index,
    "sample_count": cluster_counts.values,
    "percentage": cluster_percentages.values
})

summary_path = REPORT_DIR / "cluster_summary.csv"

summary.to_csv(
    summary_path,
    index=False
)

print("Cluster summary saved:", summary_path)


# =====================================================
# 13. FINAL RESULTS
# =====================================================

print("\n========== CLUSTERING COMPLETED ==========")

print("Dataset Samples:", len(X))

print("Original Features:", X.shape[1])

print("Features After Filtering:", X_filtered.shape[1])

print("Selected Clusters:", best_k)

print("Silhouette Score:", round(final_silhouette, 4))

print("\nCluster Summary:")
print(summary)

print("\nGenerated Files:")
print("1. reports/operational_state_clusters.png")
print("2. reports/cluster_assignments.csv")
print("3. reports/cluster_summary.csv")
