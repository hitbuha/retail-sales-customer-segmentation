import pandas as pd
import os


# ============================================================
# STEP 1: LOAD DATASETS
# ============================================================

file1 = r"D:\d\collage\ML\.venv\dataset\online_retail_09_10.csv"
file2 = r"D:\d\collage\ML\.venv\dataset\online_retail_10_11.csv"

print("=" * 60)
print("ONLINE RETAIL II - DATA CLEANING")
print("=" * 60)

print("\nLoading datasets...")

df1 = pd.read_csv(
    file1,
    encoding="latin1"
)

df2 = pd.read_csv(
    file2,
    encoding="latin1"
)

# Combine both datasets
df = pd.concat(
    [df1, df2],
    ignore_index=True
)

print("Original rows:", len(df))


# ============================================================
# STEP 2: FIX COLUMN ENCODING
# ============================================================

df.columns = [
    str(col)
    .replace("\ufeff", "")
    .replace("ï»¿", "")
    .strip()
    for col in df.columns
]

print("\nColumns after fixing encoding:")
print(df.columns.tolist())


# ============================================================
# STEP 3: CONVERT DATA TYPES
# ============================================================

# Invoice number as text
df["InvoiceNo"] = (
    df["InvoiceNo"]
    .astype(str)
    .str.strip()
)

# Stock code as text
df["StockCode"] = (
    df["StockCode"]
    .astype(str)
    .str.strip()
)

# Convert date
df["InvoiceDate"] = pd.to_datetime(
    df["InvoiceDate"],
    errors="coerce"
)

# Convert quantity
df["Quantity"] = pd.to_numeric(
    df["Quantity"],
    errors="coerce"
)

# Convert price
df["UnitPrice"] = pd.to_numeric(
    df["UnitPrice"],
    errors="coerce"
)

# Convert CustomerID
df["CustomerID"] = pd.to_numeric(
    df["CustomerID"],
    errors="coerce"
)


# ============================================================
# STEP 4: REMOVE EXACT DUPLICATES
# ============================================================

duplicate_count = df.duplicated().sum()

print("\nDuplicate records found:", duplicate_count)

df = df.drop_duplicates()

print(
    "Rows after removing duplicates:",
    len(df)
)


# ============================================================
# STEP 5: IDENTIFY CANCELLATIONS
# ============================================================

cancellation_count = (
    df["InvoiceNo"]
    .str.upper()
    .str.startswith("C")
    .sum()
)

print(
    "\nCancellation records:",
    cancellation_count
)


# ============================================================
# STEP 6: IDENTIFY RETURNS / NEGATIVE QUANTITIES
# ============================================================

return_count = (
    df["Quantity"] < 0
).sum()

print(
    "Negative quantity / return records:",
    return_count
)


# ============================================================
# STEP 7: HANDLE INVALID PRICES
# ============================================================

invalid_price_count = (
    df["UnitPrice"] <= 0
).sum()

print(
    "\nInvalid price records:",
    invalid_price_count
)

# Remove zero and negative prices
df = df[
    df["UnitPrice"] > 0
].copy()

print(
    "Rows after removing invalid prices:",
    len(df)
)


# ============================================================
# STEP 8: HANDLE MISSING DESCRIPTIONS
# ============================================================

missing_description = (
    df["Description"].isna().sum()
)

print(
    "\nMissing descriptions:",
    missing_description
)

# Keep the records and replace missing description
df["Description"] = (
    df["Description"]
    .fillna("Unknown")
)


# ============================================================
# STEP 9: CREATE TRANSACTION TYPE
# ============================================================

def classify_transaction(row):

    # Negative quantity is treated as Return
    if row["Quantity"] < 0:
        return "Return"

    # Invoice beginning with C is Cancellation
    elif row["InvoiceNo"].upper().startswith("C"):
        return "Cancellation"

    # Everything else is a normal sale
    else:
        return "Sale"


df["transaction_type"] = (
    df.apply(
        classify_transaction,
        axis=1
    )
)


# ============================================================
# STEP 10: CREATE TRANSACTION AMOUNT
# ============================================================

df["transaction_amount"] = (
    df["Quantity"] *
    df["UnitPrice"]
)


# ============================================================
# STEP 11: CREATE OUTPUT DIRECTORY
# ============================================================

output_folder = (
    "dataset/cleaned"
)

os.makedirs(
    output_folder,
    exist_ok=True
)


# ============================================================
# STEP 12: SAVE CLEANED DATASET
# ============================================================

output_file = (
    "dataset/cleaned/"
    "online_retail_II_cleaned.csv"
)

df.to_csv(
    output_file,
    index=False
)


# ============================================================
# STEP 13: FINAL ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("DATA CLEANING COMPLETED")
print("=" * 60)

print(
    "\nFinal rows:",
    len(df)
)

print(
    "Final columns:",
    len(df.columns)
)


# ============================================================
# FINAL COLUMNS
# ============================================================

print("\nFinal columns:")

for column in df.columns:
    print("-", column)


# ============================================================
# TRANSACTION TYPES
# ============================================================

print("\nTransaction types:")

print(
    df["transaction_type"]
    .value_counts()
)


# ============================================================
# MISSING CUSTOMER ID
# ============================================================

print("\nMissing CustomerID:")

print(
    df["CustomerID"].isna().sum()
)


# ============================================================
# MISSING DESCRIPTION
# ============================================================

print("\nMissing Description:")

print(
    df["Description"].isna().sum()
)


# ============================================================
# FINAL DUPLICATES CHECK
# ============================================================

print("\nRemaining duplicate records:")

print(
    df.duplicated().sum()
)


# ============================================================
# FINAL INVALID PRICE CHECK
# ============================================================

print("\nRemaining invalid prices:")

print(
    (df["UnitPrice"] <= 0).sum()
)


# ============================================================
# TRANSACTION AMOUNT SUMMARY
# ============================================================

print("\nTransaction Amount Summary:")

print(
    df["transaction_amount"]
    .describe()
)


# ============================================================
# DATE RANGE
# ============================================================

print("\nDate Range:")

print(
    "Start:",
    df["InvoiceDate"].min()
)

print(
    "End:",
    df["InvoiceDate"].max()
)


# ============================================================
# SAVE CONFIRMATION
# ============================================================

print("\n" + "=" * 60)

print(
    "Cleaned dataset saved successfully at:"
)

print(
    os.path.abspath(output_file)
)

print("=" * 60)