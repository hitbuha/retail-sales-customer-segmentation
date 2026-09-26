import pandas as pd
from pathlib import Path

print("=" * 60)
print("STEP 11 - ML PROJECT INTEGRATION")
print("=" * 60)

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"

CLUSTER_FILE = (
    DATASET_DIR / "clustering" /
    "customer_clusters_with_features.csv"
)

PROFILE_FILE = (
    DATASET_DIR / "clustering" /
    "customer_cluster_profile.csv"
)

SALES_FILE = (
    DATASET_DIR / "sales_prediction" /
    "sales_predictions.csv"
)

MODEL_FILE = (
    DATASET_DIR / "sales_prediction" /
    "regression_model_comparison.csv"
)

OUTPUT_DIR = DATASET_DIR / "final_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# 1. LOAD CUSTOMER SEGMENTATION RESULTS
# ---------------------------------------------------------

print("\nLoading customer segmentation results...")

customers = pd.read_csv(CLUSTER_FILE)

print("Customer segmentation loaded.")
print("Customers:", len(customers))

# Create descriptive cluster names
cluster_names = {
    0: "High-Engagement / High-Value",
    1: "Lower-Engagement / Lower-Value"
}

customers["segment"] = customers["cluster"].map(cluster_names)

# Save final customer segmentation
customer_output = OUTPUT_DIR / "final_customer_segments.csv"
customers.to_csv(customer_output, index=False)

print("Customer segments saved.")


# ---------------------------------------------------------
# 2. LOAD CUSTOMER CLUSTER PROFILE
# ---------------------------------------------------------

print("\nLoading cluster profiles...")

profiles = pd.read_csv(PROFILE_FILE)

profile_output = OUTPUT_DIR / "final_cluster_profiles.csv"
profiles.to_csv(profile_output, index=False)

print("Cluster profiles saved.")


# ---------------------------------------------------------
# 3. LOAD SALES PREDICTION RESULTS
# ---------------------------------------------------------

print("\nLoading sales prediction results...")

sales = pd.read_csv(SALES_FILE)

sales["date"] = pd.to_datetime(sales["date"])

print("Sales prediction results loaded.")
print("Test records:", len(sales))


# ---------------------------------------------------------
# 4. LOAD MODEL COMPARISON
# ---------------------------------------------------------

print("\nLoading regression model comparison...")

models = pd.read_csv(MODEL_FILE)

model_output = OUTPUT_DIR / "final_model_comparison.csv"
models.to_csv(model_output, index=False)

print("Model comparison saved.")


# ---------------------------------------------------------
# 5. FINAL SALES PREDICTIONS
# ---------------------------------------------------------

sales_output = OUTPUT_DIR / "final_sales_predictions.csv"
sales.to_csv(sales_output, index=False)

print("Sales predictions saved.")


# ---------------------------------------------------------
# 6. CREATE PROJECT SUMMARY
# ---------------------------------------------------------

print("\nCreating project summary...")

summary = {
    "Total Customers": len(customers),
    "Customer Segment 0": int(
        (customers["cluster"] == 0).sum()
    ),
    "Customer Segment 1": int(
        (customers["cluster"] == 1).sum()
    ),
    "Sales Test Records": len(sales),
    "Number of Regression Models": len(models),
    "PCA Components Used": 3,
    "K-Means Clusters": 2
}

summary_df = pd.DataFrame(
    list(summary.items()),
    columns=["Metric", "Value"]
)

summary_output = OUTPUT_DIR / "project_summary.csv"
summary_df.to_csv(summary_output, index=False)


# ---------------------------------------------------------
# 7. DISPLAY SUMMARY
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("FINAL PROJECT SUMMARY")
print("=" * 60)

for key, value in summary.items():
    print(f"{key:<35}: {value}")


# ---------------------------------------------------------
# 8. DISPLAY MODEL RESULTS
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("REGRESSION MODEL RESULTS")
print("=" * 60)

print(models.to_string(index=False))


# ---------------------------------------------------------
# 9. FINAL OUTPUT
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 11 INTEGRATION COMPLETED")
print("=" * 60)

print("\nFinal files created:")

print("1. final_customer_segments.csv")
print("2. final_cluster_profiles.csv")
print("3. final_sales_predictions.csv")
print("4. final_model_comparison.csv")
print("5. project_summary.csv")

print("\nOutput directory:")
print(OUTPUT_DIR)