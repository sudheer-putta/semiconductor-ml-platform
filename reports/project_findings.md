# Semiconductor Manufacturing Quality & Anomaly Monitoring

## Dataset
- Dataset: UCI SECOM
- Manufacturing observations: 1,567
- Sensor features: 590
- Passing samples: 1,463
- Failed samples: 104

## Supervised Learning
- Baseline: Logistic Regression
- Model comparison: Logistic Regression, Random Forest, Extra Trees, SVM
- Best candidate by cross-validation PR-AUC: Random Forest
- Cross-validation PR-AUC: 0.2178
- Tuned decision threshold: 0.0871
- Test PR-AUC: 0.1905
- Test failure recall: 0.62

## Clustering
- PCA components retaining 95% variance: 170
- K-Means tested with 2–8 clusters
- Best silhouette score: 0.7567 at K=2
- The smaller cluster contained only four observations; therefore, the result requires further investigation.

## Anomaly Detection
- Algorithm: Isolation Forest
- Contamination setting: 0.05
- Anomalies identified: 79
- Failure rate in anomaly group: 12.66%
- Failure rate in normal group: 6.32%

## Limitations
- SECOM does not provide recipe, chamber, wafer, or verified golden-population identifiers.
- Anomaly labels are algorithm-generated, not confirmed defects.
- Results are exploratory and require further validation before deployment.
