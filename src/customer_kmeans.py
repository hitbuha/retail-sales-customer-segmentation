import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ============================================================
# 1. File paths
# ============================================================

input_file = r"dataset\pca\customer_pca_data.csv"

output_dir = r"dataset\clustering"

os.makedirs(output_dir, exist_ok=True)


# ============================================================
# 2. Load PCA dataset
# ============================================================

print("Loading PCA dataset...")

df = pd.read_csv(input_file)

print("Dataset loaded successfully.")


# ============================================================
# 3. Select PCA features
# ============================================================

features = [
    "PC1",
    "PC2",
    "PC3"
]

X = df[features]

print("\nFeatures used for K-Means:")
print(features)

print("\nInput shape:")
print(X.shape)


# ============================================================
# 4. Test different K values
# ============================================================

k_values = range(2, 11)

inertia_values = []
silhouette_values = []


print("\n==============================================")
print("TESTING DIFFERENT K VALUES")
print("==============================================")


for k in k_values:

    print(f"\nTesting K = {k}")

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(X)

    inertia = kmeans.inertia_

    silhouette = silhouette_score(
        X,
        labels
    )

    inertia_values.append(inertia)

    silhouette_values.append(silhouette)

    print(f"Inertia: {inertia:.2f}")
    print(f"Silhouette Score: {silhouette:.4f}")


# ============================================================
# 5. Create evaluation table
# ============================================================

evaluation_df = pd.DataFrame({
    "K": list(k_values),
    "Inertia": inertia_values,
    "Silhouette_Score": silhouette_values
})


print("\n==============================================")
print("K-MEANS EVALUATION RESULTS")
print("==============================================")

print(evaluation_df)


# ============================================================
# 6. Save evaluation results
# ============================================================

evaluation_df.to_csv(
    os.path.join(
        output_dir,
        "kmeans_evaluation.csv"
    ),
    index=False
)


# ============================================================
# 7. Elbow Method plot
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    list(k_values),
    inertia_values,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia / WCSS")

plt.title("Elbow Method for K-Means")

plt.xticks(list(k_values))

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_dir,
        "elbow_method.png"
    )
)

plt.close()


# ============================================================
# 8. Silhouette Score plot
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    list(k_values),
    silhouette_values,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")

plt.title("Silhouette Score for Different K Values")

plt.xticks(list(k_values))

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_dir,
        "silhouette_scores.png"
    )
)

plt.close()


# ============================================================
# 9. Display best silhouette K
# ============================================================

best_index = silhouette_values.index(
    max(silhouette_values)
)

best_k_silhouette = list(k_values)[best_index]

print("\n==============================================")
print("SILHOUETTE RESULT")
print("==============================================")

print(
    "Highest Silhouette Score K:",
    best_k_silhouette
)

print(
    "Highest Silhouette Score:",
    f"{silhouette_values[best_index]:.4f}"
)


# ============================================================
# 10. NOTE
# ============================================================

print("\n==============================================")
print("IMPORTANT")
print("==============================================")

print(
    "Do NOT automatically use the highest silhouette K."
)

print(
    "First inspect the Elbow Method and Silhouette results."
)

print(
    "We will select the final K after reviewing the results."
)