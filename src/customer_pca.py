import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.decomposition import PCA


# ============================================================
# 1. File paths
# ============================================================

input_file = r"dataset\preprocessed\customer_features_scaled.csv"

output_dir = r"dataset\pca"

output_file = os.path.join(
    output_dir,
    "customer_pca_data.csv"
)

os.makedirs(output_dir, exist_ok=True)


# ============================================================
# 2. Load scaled dataset
# ============================================================

print("Loading scaled customer dataset...")

df = pd.read_csv(input_file)

print("Dataset loaded successfully.")


# ============================================================
# 3. Select PCA features
# ============================================================

features = [
    "recency",
    "frequency",
    "monetary",
    "average_order_value",
    "total_quantity",
    "unique_products"
]

X = df[features]

print("\nFeatures used for PCA:")
print(features)

print("\nInput shape:")
print(X.shape)


# ============================================================
# 4. Apply PCA
# ============================================================

print("\nApplying PCA...")

pca = PCA()

X_pca = pca.fit_transform(X)


# ============================================================
# 5. Explained variance
# ============================================================

explained_variance = pca.explained_variance_ratio_

cumulative_variance = explained_variance.cumsum()


print("\n==============================================")
print("PCA EXPLAINED VARIANCE")
print("==============================================")

for i, variance in enumerate(explained_variance):

    print(
        f"PC{i + 1}: "
        f"{variance * 100:.2f}% "
        f"| Cumulative: "
        f"{cumulative_variance[i] * 100:.2f}%"
    )


# ============================================================
# 6. PCA components
# ============================================================

print("\n==============================================")
print("PCA COMPONENT LOADINGS")
print("==============================================")

components = pd.DataFrame(
    pca.components_,
    columns=features,
    index=[
        f"PC{i + 1}"
        for i in range(len(features))
    ]
)

print(components)


# ============================================================
# 7. Save explained variance
# ============================================================

variance_df = pd.DataFrame({
    "Principal_Component": [
        f"PC{i + 1}"
        for i in range(len(explained_variance))
    ],
    "Explained_Variance": explained_variance,
    "Explained_Variance_Percentage":
        explained_variance * 100,
    "Cumulative_Variance":
        cumulative_variance,
    "Cumulative_Variance_Percentage":
        cumulative_variance * 100
})

variance_df.to_csv(
    os.path.join(
        output_dir,
        "explained_variance.csv"
    ),
    index=False
)


# ============================================================
# 8. Scree plot
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, len(explained_variance) + 1),
    explained_variance * 100,
    marker="o"
)

plt.xlabel("Principal Component")
plt.ylabel("Explained Variance (%)")

plt.title("PCA Scree Plot")

plt.xticks(
    range(1, len(explained_variance) + 1)
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_dir,
        "pca_scree_plot.png"
    )
)

plt.close()


# ============================================================
# 9. Cumulative variance plot
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, len(cumulative_variance) + 1),
    cumulative_variance * 100,
    marker="o"
)

plt.axhline(
    y=90,
    linestyle="--",
    label="90% Variance"
)

plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance (%)")

plt.title("PCA Cumulative Explained Variance")

plt.xticks(
    range(1, len(cumulative_variance) + 1)
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_dir,
        "pca_cumulative_variance.png"
    )
)

plt.close()


# ============================================================
# 10. Determine components for 90% variance
# ============================================================

n_components_90 = (
    (cumulative_variance >= 0.90).argmax() + 1
)

print("\n==============================================")
print("COMPONENT SELECTION")
print("==============================================")

print(
    "Components required for 90% variance:",
    n_components_90
)


# ============================================================
# 11. Create reduced PCA dataset
# ============================================================

pca_reduced = PCA(
    n_components=n_components_90
)

X_reduced = pca_reduced.fit_transform(X)


# ============================================================
# 12. Create PCA DataFrame
# ============================================================

pca_columns = [
    f"PC{i + 1}"
    for i in range(n_components_90)
]

pca_df = pd.DataFrame(
    X_reduced,
    columns=pca_columns
)


# Keep CustomerID for identification
pca_df.insert(
    0,
    "CustomerID",
    df["CustomerID"].values
)


# ============================================================
# 13. Save PCA dataset
# ============================================================

pca_df.to_csv(
    output_file,
    index=False
)


# ============================================================
# 14. Display final result
# ============================================================

print("\n==============================================")
print("PCA COMPLETED")
print("==============================================")

print("\nOriginal feature count:", len(features))

print(
    "Reduced feature count:",
    n_components_90
)

print("\nPCA dataset shape:")

print(pca_df.shape)

print("\nFirst 5 rows:")

print(pca_df.head())

print("\nSaved successfully at:")

print(os.path.abspath(output_file))