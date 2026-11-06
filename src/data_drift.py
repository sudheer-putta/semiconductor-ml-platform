
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw"
REPORTS_DIR = BASE_DIR / "reports"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. LOAD SENSOR DATA
# --------------------------------------------------

print("Loading SECOM sensor dataset...")

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

print("Original dataset shape:", X.shape)

# Give sensor columns consistent names
X.columns = [f"sensor_{i}" for i in range(X.shape[1])]

# --------------------------------------------------
# 3. CREATE REFERENCE AND CURRENT BATCHES
# --------------------------------------------------

# SECOM records are treated in their original sequence.
split_index = int(len(X) * 0.80)

reference_data = X.iloc[:split_index].copy()
current_data = X.iloc[split_index:].copy()

print("Reference batch shape:", reference_data.shape)
print("Current batch shape:", current_data.shape)

# Remove columns that contain no usable values
# or have no variation across the combined batches.
combined = pd.concat(
    [reference_data, current_data],
    ignore_index=True
)

usable_columns = combined.columns[
    combined.notna().any() &
    (combined.nunique(dropna=True) > 1)
]

reference_data = reference_data[usable_columns]
current_data = current_data[usable_columns]

print("Features used for drift monitoring:", len(usable_columns))

# --------------------------------------------------
# 4. GENERATE EVIDENTLY REPORT
# --------------------------------------------------

print("\nGenerating Evidently data drift report...")

report = Report(
    metrics=[
        DataDriftPreset()
    ]
)

snapshot = report.run(
    current_data=current_data,
    reference_data=reference_data
)

# --------------------------------------------------
# 5. SAVE REPORTS
# --------------------------------------------------

html_path = REPORTS_DIR / "data_drift_report.html"
json_path = REPORTS_DIR / "data_drift_report.json"

snapshot.save_html(str(html_path))
snapshot.save_json(str(json_path))

print("\nData drift monitoring completed!")
print("HTML report:", html_path)
print("JSON report:", json_path)

print("\nOpen the HTML report in your browser to inspect")
print("feature-level distribution changes.")
