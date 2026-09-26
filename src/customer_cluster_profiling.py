import pandas as pd
import os


# ============================================================
# 1. File paths
# ============================================================

cluster_file = r"dataset\clustering\customer_clusters.csv"

customer_file = r"dataset\ml_ready\customer_segmentation_data.csv"

output_dir = r"dataset\clustering"

output_file = os.path.join(
    output_dir,
    "customer_cluster_profile.csv"
)


# ============================================================
# 2. Load datasets
# ============================================================

print("Loading cluster assignments...")

clusters = pd.read_csv(cluster_file)

print("Loading original customer features...")

customers = pd.read_csv(customer_file)


# ============================================================
# 3. Merge cluster information with original features
# ============================================================

df = customers.merge(
    clusters[
        [
            "CustomerID",
            "cluster"
        ]
    ],
    on="CustomerID",
    how="inner"
)


print("\n==============================================")
print("CLUSTER PROFILING")
print("==============================================")

print("Customers after merging:", len(df))


# ============================================================
# 4. Calculate cluster profile
# ============================================================

profile = df.groupby("cluster").agg(
    customer_count=("CustomerID", "count"),

    avg_recency=("recency", "mean"),
    median_recency=("recency", "median"),

    avg_frequency=("frequency", "mean"),
    median_frequency=("frequency", "median"),

    avg_monetary=("monetary", "mean"),
    median_monetary=("monetary", "median"),

    avg_order_value=("average_order_value", "mean"),
    median_order_value=("average_order_value", "median"),

    avg_total_quantity=("total_quantity", "mean"),
    median_total_quantity=("total_quantity", "median"),

    avg_unique_products=("unique_products", "mean"),
    median_unique_products=("unique_products", "median")
).reset_index()


# ============================================================
# 5. Calculate percentage of customers
# ============================================================

profile["customer_percentage"] = (
    profile["customer_count"]
    / len(df)
    * 100
)


# ============================================================
# 6. Display profile
# ============================================================

print("\n==============================================")
print("CUSTOMER CLUSTER PROFILE")
print("==============================================")

print(
    profile.to_string(index=False)
)


# ============================================================
# 7. Save profile
# ============================================================

profile.to_csv(
    output_file,
    index=False
)


# ============================================================
# 8. Display cluster comparison
# ============================================================

print("\n==============================================")
print("CLUSTER COMPARISON")
print("==============================================")


for _, row in profile.iterrows():

    print(
        f"\nCluster {int(row['cluster'])}"
    )

    print(
        f"Customers: {int(row['customer_count'])}"
    )

    print(
        f"Percentage: {row['customer_percentage']:.2f}%"
    )

    print(
        f"Average Recency: {row['avg_recency']:.2f} days"
    )

    print(
        f"Average Frequency: {row['avg_frequency']:.2f}"
    )

    print(
        f"Average Monetary: "
        f"{row['avg_monetary']:.2f}"
    )

    print(
        f"Average Order Value: "
        f"{row['avg_order_value']:.2f}"
    )

    print(
        f"Average Quantity: "
        f"{row['avg_total_quantity']:.2f}"
    )

    print(
        f"Average Unique Products: "
        f"{row['avg_unique_products']:.2f}"
    )


# ============================================================
# 9. Save merged customer-cluster dataset
# ============================================================

merged_output = os.path.join(
    output_dir,
    "customer_clusters_with_features.csv"
)

df.to_csv(
    merged_output,
    index=False
)


print("\n==============================================")
print("CLUSTER PROFILING COMPLETED")
print("==============================================")

print("\nProfile saved at:")
print(os.path.abspath(output_file))

print("\nDetailed customer-cluster dataset saved at:")
print(os.path.abspath(merged_output))