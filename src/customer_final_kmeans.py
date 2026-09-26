import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.cluster import KMeans


# ============================================================
# 1. File paths
# ============================================================

input_file = r"dataset\pca\customer_pca_data.csv"

output_dir = r"dataset\clustering"

output_file = os.path.join(
    output_dir,
    "customer_clusters.csv"
)

os.makedirs(output_dir, exist_ok=True)


# ============================================================
# 2. Load PCA dataset
# ============================================================

print("Loading PCA dataset...")

df = pd.read_csv(input_file)


# ============================================================
# 3. Select PCA features
# ============================================================

features = [
    "PC1",
    "PC2",
    "PC3"
]

X = df[features]


# ============================================================
# 4. Final K-Means model
# ============================================================

K = 2

print("\n==============================================")
print("FINAL K-MEANS MODEL")
print("==============================================")

print("Number of clusters:", K)

kmeans = KMeans(
    n_clusters=K,
    random_state=42,
    n_init=10
)

df["cluster"] = kmeans.fit_predict(X)


# ============================================================
# 5. Cluster sizes
# ============================================================

print("\n==============================================")
print("CLUSTER SIZES")
print("==============================================")

cluster_counts = df["cluster"].value_counts().sort_index()

print(cluster_counts)


# ============================================================
# 6. Cluster centers
# ============================================================

print("\n==============================================")
print("CLUSTER CENTERS")
print("==============================================")

cluster_centers = pd.DataFrame(
    kmeans.cluster_centers_,
    columns=features
)

cluster_centers.insert(
    0,
    "cluster",
    range(K)
)

print(cluster_centers)


# ============================================================
# 7. Save cluster centers
# ============================================================

cluster_centers.to_csv(
    os.path.join(
        output_dir,
        "cluster_centers.csv"
    ),
    index=False
)


# ============================================================
# 8. Save customer clusters
# ============================================================

df.to_csv(
    output_file,
    index=False
)


# ============================================================
# 9. 2D visualization: PC1 vs PC2
# ============================================================

plt.figure(figsize=(9, 6))

for cluster in sorted(df["cluster"].unique()):

    cluster_data = df[
        df["cluster"] == cluster
    ]

    plt.scatter(
        cluster_data["PC1"],
        cluster_data["PC2"],
        label=f"Cluster {cluster}",
        alpha=0.6
    )


# Plot cluster centers

plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    marker="X",
    s=200,
    label="Centroids"
)

plt.xlabel("PC1")
plt.ylabel("PC2")

plt.title("Customer Segmentation using K-Means")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_dir,
        "customer_clusters_pc1_pc2.png"
    )
)

plt.close()


# ============================================================
# 10. Final output
# ============================================================

print("\n==============================================")
print("FINAL K-MEANS COMPLETED")
print("==============================================")

print("\nDataset shape:")
print(df.shape)

print("\nFirst 10 customers with clusters:")

print(
    df[
        [
            "CustomerID",
            "PC1",
            "PC2",
            "PC3",
            "cluster"
        ]
    ].head(10)
)

print("\nFiles saved in:")

print(os.path.abspath(output_dir))