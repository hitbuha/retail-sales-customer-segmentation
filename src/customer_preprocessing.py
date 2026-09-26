import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. File paths
# ============================================================

input_file = r"dataset\ml_ready\customer_segmentation_data.csv"

output_dir = r"dataset\preprocessed"

output_file = os.path.join(
    output_dir,
    "customer_features_scaled.csv"
)

os.makedirs(output_dir, exist_ok=True)


# ============================================================
# 2. Load dataset
# ============================================================

print("Loading customer segmentation dataset...")

df = pd.read_csv(input_file)


# ============================================================
# 3. Select ML features
# ============================================================

features = [
    "recency",
    "frequency",
    "monetary",
    "average_order_value",
    "total_quantity",
    "unique_products"
]

X = df[features].copy()


print("\nFeatures selected:")
print(features)


# ============================================================
# 4. Check missing values
# ============================================================

print("\nMissing values before preprocessing:")

print(X.isnull().sum())


# ============================================================
# 5. Log transformation
# ============================================================

print("\nApplying log transformation...")

X_log = np.log1p(X)


# ============================================================
# 6. Standardization
# ============================================================

print("Applying StandardScaler...")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_log)


# ============================================================
# 7. Convert back to DataFrame
# ============================================================

X_scaled = pd.DataFrame(
    X_scaled,
    columns=features
)


# ============================================================
# 8. Add CustomerID
# ============================================================

X_scaled.insert(
    0,
    "CustomerID",
    df["CustomerID"].values
)


# ============================================================
# 9. Save scaled dataset
# ============================================================

X_scaled.to_csv(
    output_file,
    index=False
)


# ============================================================
# 10. Check results
# ============================================================

print("\n==============================================")
print("PREPROCESSING COMPLETED")
print("==============================================")

print("\nScaled dataset shape:")
print(X_scaled.shape)

print("\nFirst 5 rows:")
print(X_scaled.head())


# ============================================================
# 11. Verify mean and standard deviation
# ============================================================

print("\nMean after scaling:")

print(
    X_scaled[features].mean()
)


print("\nStandard deviation after scaling:")

print(
    X_scaled[features].std()
)


# ============================================================
# 12. Save scaler parameters
# ============================================================

scaler_parameters = pd.DataFrame({
    "feature": features,
    "mean": scaler.mean_,
    "scale": scaler.scale_
})

scaler_parameters.to_csv(
    os.path.join(
        output_dir,
        "scaler_parameters.csv"
    ),
    index=False
)


print("\nSaved successfully at:")

print(os.path.abspath(output_file))