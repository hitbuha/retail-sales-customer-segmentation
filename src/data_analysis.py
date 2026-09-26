import pandas as pd

# -----------------------------
# 1. Load both datasets
# -----------------------------

df1 = pd.read_csv(
    "D:\d\collage\ML\.venv\dataset/online_retail_09_10.csv",
    encoding="latin1"
)

df2 = pd.read_csv(
    "D:\d\collage\ML\.venv\dataset/online_retail_10_11.csv",
    encoding="latin1"
)

# Combine datasets
df = pd.concat([df1, df2], ignore_index=True)

# Fix BOM in InvoiceNo column
df.columns = [
    str(col).lstrip("\ufeff").replace("ï»¿", "")
    for col in df.columns
]

print("=" * 60)
print("ONLINE RETAIL II - DATA ANALYSIS")
print("=" * 60)


# -----------------------------
# 2. Basic information
# -----------------------------

print("\n1. DATASET INFORMATION")
print("-" * 40)

print("Total Rows:", len(df))
print("Total Columns:", len(df.columns))

print("\nColumns:")
for col in df.columns:
    print("-", col)


# -----------------------------
# 3. Missing Values
# -----------------------------

print("\n2. MISSING VALUES")
print("-" * 40)

missing = df.isnull().sum()

for column, count in missing.items():
    percentage = (count / len(df)) * 100
    print(f"{column}: {count} ({percentage:.2f}%)")


# -----------------------------
# 4. Duplicate Records
# -----------------------------

print("\n3. DUPLICATE RECORDS")
print("-" * 40)

duplicates = df.duplicated().sum()

print("Duplicate Rows:", duplicates)


# -----------------------------
# 5. Cancellations
# -----------------------------

print("\n4. CANCELLATIONS")
print("-" * 40)

cancellations = df["InvoiceNo"].astype(str).str.startswith("C").sum()

print("Cancellation Transactions:", cancellations)


# -----------------------------
# 6. Negative Quantities
# -----------------------------

print("\n5. NEGATIVE QUANTITIES")
print("-" * 40)

negative_quantity = (df["Quantity"] < 0).sum()

print("Negative Quantity Records:", negative_quantity)


# -----------------------------
# 7. Invalid Prices
# -----------------------------

print("\n6. INVALID PRICES")
print("-" * 40)

invalid_prices = (df["UnitPrice"] <= 0).sum()

print("Zero / Negative Price Records:", invalid_prices)


# -----------------------------
# 8. Unique Customers
# -----------------------------

print("\n7. UNIQUE CUSTOMERS")
print("-" * 40)

unique_customers = df["CustomerID"].nunique()

print("Unique Customers:", unique_customers)


# -----------------------------
# 9. Unique Products
# -----------------------------

print("\n8. UNIQUE PRODUCTS")
print("-" * 40)

unique_products = df["StockCode"].nunique()

print("Unique Products:", unique_products)


# -----------------------------
# 10. Unique Invoices
# -----------------------------

print("\n9. UNIQUE INVOICES")
print("-" * 40)

unique_invoices = df["InvoiceNo"].nunique()

print("Unique Invoices:", unique_invoices)


# -----------------------------
# 11. Countries
# -----------------------------

print("\n10. COUNTRIES")
print("-" * 40)

unique_countries = df["Country"].nunique()

print("Number of Countries:", unique_countries)

print("\nCountry List:")
print(df["Country"].unique())


# -----------------------------
# 12. Date Range
# -----------------------------

print("\n11. DATE RANGE")
print("-" * 40)

df["InvoiceDate"] = pd.to_datetime(
    df["InvoiceDate"],
    dayfirst=False
)

print("Minimum Date:", df["InvoiceDate"].min())
print("Maximum Date:", df["InvoiceDate"].max())


# -----------------------------
# 13. Summary
# -----------------------------

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print("Total Records       :", len(df))
print("Duplicate Records   :", duplicates)
print("Missing Customer ID :", df["CustomerID"].isnull().sum())
print("Cancellations       :", cancellations)
print("Negative Quantity   :", negative_quantity)
print("Invalid Prices      :", invalid_prices)
print("Unique Customers    :", unique_customers)
print("Unique Products     :", unique_products)
print("Unique Invoices     :", unique_invoices)
print("Countries           :", unique_countries)
print("Date Range          :", df["InvoiceDate"].min(),
      "to", df["InvoiceDate"].max())