import pandas as pd
import os

# ============================================================
# 1. File paths
# ============================================================

input_file = r"dataset\cleaned\online_retail_II_cleaned.csv"
output_file = r"dataset\ml_ready\customer_segmentation_data.csv"


# ============================================================
# 2. Load cleaned dataset
# ============================================================

print("Loading cleaned dataset...")

df = pd.read_csv(input_file)

print("Original cleaned records:", len(df))


# ============================================================
# 3. Convert InvoiceDate
# ============================================================

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])


# ============================================================
# 4. Remove transactions without CustomerID
# ============================================================

missing_customer = df["CustomerID"].isna().sum()

df = df.dropna(subset=["CustomerID"])

print("Records without CustomerID removed:", missing_customer)


# ============================================================
# 5. Keep only SALE transactions
# ============================================================

print("\nTransaction types before filtering:")
print(df["transaction_type"].value_counts())

df = df[df["transaction_type"] == "Sale"].copy()

print("\nRecords after keeping only Sale transactions:", len(df))


# ============================================================
# 6. Convert CustomerID to integer
# ============================================================

df["CustomerID"] = df["CustomerID"].astype(int)


# ============================================================
# 7. Reference date
# ============================================================

reference_date = df["InvoiceDate"].max()

print("Reference date:", reference_date)


# ============================================================
# 8. Create RFM + customer features
# ============================================================

customer_df = df.groupby("CustomerID").agg(
    last_purchase_date=("InvoiceDate", "max"),
    frequency=("InvoiceNo", "nunique"),
    monetary=("transaction_amount", "sum"),
    total_quantity=("Quantity", "sum"),
    unique_products=("StockCode", "nunique")
).reset_index()


# ============================================================
# 9. Calculate Recency
# ============================================================

customer_df["recency"] = (
    reference_date - customer_df["last_purchase_date"]
).dt.days


# ============================================================
# 10. Calculate Average Order Value
# ============================================================

customer_df["average_order_value"] = (
    customer_df["monetary"] / customer_df["frequency"]
)


# ============================================================
# 11. Select final columns
# ============================================================

customer_df = customer_df[
    [
        "CustomerID",
        "recency",
        "frequency",
        "monetary",
        "average_order_value",
        "total_quantity",
        "unique_products"
    ]
]


# ============================================================
# 12. Remove invalid values
# ============================================================

customer_df = customer_df.replace(
    [float("inf"), float("-inf")],
    pd.NA
)

customer_df = customer_df.dropna()


# ============================================================
# 13. Create output directory
# ============================================================

os.makedirs(os.path.dirname(output_file), exist_ok=True)


# ============================================================
# 14. Save dataset
# ============================================================

customer_df.to_csv(output_file, index=False)


# ============================================================
# 15. Display results
# ============================================================

print("\n==============================================")
print("CUSTOMER SEGMENTATION DATASET CREATED")
print("==============================================")

print("Number of customers:", len(customer_df))
print("Number of columns:", len(customer_df.columns))

print("\nColumns:")
print(list(customer_df.columns))

print("\nFirst 5 customers:")
print(customer_df.head())

print("\nDataset statistics:")
print(customer_df.describe())

print("\nMinimum Monetary:", customer_df["monetary"].min())
print("Minimum Average Order Value:",
      customer_df["average_order_value"].min())
print("Minimum Total Quantity:",
      customer_df["total_quantity"].min())
print("Minimum Recency:",
      customer_df["recency"].min())
print("Maximum Recency:",
      customer_df["recency"].max())

print("\nNegative monetary values:",
      (customer_df["monetary"] < 0).sum())

print("Negative average order values:",
      (customer_df["average_order_value"] < 0).sum())

print("\nSaved successfully at:")
print(os.path.abspath(output_file))