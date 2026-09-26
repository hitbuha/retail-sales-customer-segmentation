import pandas as pd
import matplotlib.pyplot as plt
import os


# ============================================================
# 1. File paths
# ============================================================

input_file = r"dataset\ml_ready\customer_segmentation_data.csv"

output_dir = r"dataset\eda\customer_segmentation"

os.makedirs(output_dir, exist_ok=True)


# ============================================================
# 2. Load dataset
# ============================================================

print("Loading customer segmentation dataset...")

df = pd.read_csv(input_file)

print("\nDataset loaded successfully.")


# ============================================================
# 3. Basic information
# ============================================================

print("\n==============================================")
print("CUSTOMER SEGMENTATION EDA")
print("==============================================")

print("\nNumber of customers:", len(df))
print("Number of features:", len(df.columns))

print("\nColumns:")
print(list(df.columns))


# ============================================================
# 4. Missing values
# ============================================================

print("\n==============================================")
print("MISSING VALUES")
print("==============================================")

print(df.isnull().sum())


# ============================================================
# 5. Duplicate records
# ============================================================

print("\n==============================================")
print("DUPLICATE RECORDS")
print("==============================================")

print("Duplicate rows:", df.duplicated().sum())


# ============================================================
# 6. Statistical summary
# ============================================================

print("\n==============================================")
print("STATISTICAL SUMMARY")
print("==============================================")

print(df.describe())


# ============================================================
# 7. Feature distributions
# ============================================================

features = [
    "recency",
    "frequency",
    "monetary",
    "average_order_value",
    "total_quantity",
    "unique_products"
]


# ============================================================
# 8. Histograms
# ============================================================

print("\nCreating feature distribution plots...")

for feature in features:

    plt.figure(figsize=(8, 5))

    plt.hist(df[feature], bins=50)

    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Number of Customers")

    plt.tight_layout()

    plt.savefig(
        os.path.join(output_dir, f"{feature}_distribution.png")
    )

    plt.close()


# ============================================================
# 9. Boxplots
# ============================================================

print("Creating boxplots...")

for feature in features:

    plt.figure(figsize=(8, 5))

    plt.boxplot(df[feature])

    plt.title(f"Boxplot of {feature}")
    plt.ylabel(feature)

    plt.tight_layout()

    plt.savefig(
        os.path.join(output_dir, f"{feature}_boxplot.png")
    )

    plt.close()


# ============================================================
# 10. Correlation matrix
# ============================================================

print("Creating correlation matrix...")

correlation = df[features].corr()

print("\nCorrelation Matrix:")
print(correlation)

plt.figure(figsize=(9, 7))

plt.imshow(correlation, cmap="coolwarm", interpolation="nearest")

plt.colorbar()

plt.xticks(
    range(len(features)),
    features,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(features)),
    features
)

plt.title("Customer Feature Correlation Matrix")

plt.tight_layout()

plt.savefig(
    os.path.join(output_dir, "correlation_matrix.png")
)

plt.close()


# ============================================================
# 11. Save correlation matrix
# ============================================================

correlation.to_csv(
    os.path.join(output_dir, "correlation_matrix.csv")
)


# ============================================================
# 12. Top customers by monetary value
# ============================================================

print("\n==============================================")
print("TOP 10 CUSTOMERS BY MONETARY VALUE")
print("==============================================")

print(
    df.nlargest(10, "monetary")[
        [
            "CustomerID",
            "monetary",
            "frequency",
            "recency",
            "average_order_value"
        ]
    ]
)


# ============================================================
# 13. Top customers by frequency
# ============================================================

print("\n==============================================")
print("TOP 10 CUSTOMERS BY FREQUENCY")
print("==============================================")

print(
    df.nlargest(10, "frequency")[
        [
            "CustomerID",
            "frequency",
            "monetary",
            "recency"
        ]
    ]
)


# ============================================================
# 14. Top customers by quantity
# ============================================================

print("\n==============================================")
print("TOP 10 CUSTOMERS BY TOTAL QUANTITY")
print("==============================================")

print(
    df.nlargest(10, "total_quantity")[
        [
            "CustomerID",
            "total_quantity",
            "monetary",
            "frequency"
        ]
    ]
)


# ============================================================
# 15. Save EDA summary
# ============================================================

summary = df[features].describe().T

summary.to_csv(
    os.path.join(output_dir, "customer_eda_summary.csv")
)


print("\n==============================================")
print("EDA COMPLETED")
print("==============================================")

print("EDA files saved in:")

print(
    os.path.abspath(output_dir)
)