import pandas as pd
import os


# ============================================================
# STEP 1: LOAD CLEANED DATA
# ============================================================

input_file = (
    r"D:\d\collage\ML\dataset\cleaned"
    r"\online_retail_II_cleaned.csv"
)

print("=" * 60)
print("CREATING SALES PREDICTION DATASET")
print("=" * 60)

df = pd.read_csv(
    input_file
)

print("\nOriginal cleaned records:", len(df))


# ============================================================
# STEP 2: CONVERT DATE
# ============================================================

df["InvoiceDate"] = pd.to_datetime(
    df["InvoiceDate"],
    errors="coerce"
)


# ============================================================
# STEP 3: CREATE DATE COLUMN
# ============================================================

df["date"] = df["InvoiceDate"].dt.date


# ============================================================
# STEP 4: SEPARATE SALES AND RETURNS
# ============================================================

# For sales prediction, we calculate NET sales.
# Sale = positive amount
# Return = negative amount

daily_sales = df.groupby("date").agg(

    total_quantity=("Quantity", "sum"),

    transaction_count=("InvoiceNo", "nunique"),

    unique_customers=("CustomerID", "nunique"),

    unique_products=("StockCode", "nunique"),

    avg_unit_price=("UnitPrice", "mean"),

    total_sales=("transaction_amount", "sum")

).reset_index()


# ============================================================
# STEP 5: CONVERT DATE TO DATETIME
# ============================================================

daily_sales["date"] = pd.to_datetime(
    daily_sales["date"]
)


# ============================================================
# STEP 6: CREATE TIME FEATURES
# ============================================================

daily_sales["year"] = (
    daily_sales["date"].dt.year
)

daily_sales["month"] = (
    daily_sales["date"].dt.month
)

daily_sales["day"] = (
    daily_sales["date"].dt.day
)

daily_sales["day_of_week"] = (
    daily_sales["date"].dt.dayofweek
)

daily_sales["day_name"] = (
    daily_sales["date"].dt.day_name()
)


# ============================================================
# STEP 7: CREATE PREVIOUS SALES FEATURES
# ============================================================

daily_sales = daily_sales.sort_values(
    "date"
)

daily_sales["previous_day_sales"] = (
    daily_sales["total_sales"].shift(1)
)


# ============================================================
# STEP 8: REMOVE FIRST ROW WITH NO PREVIOUS SALES
# ============================================================

daily_sales = daily_sales.dropna(
    subset=["previous_day_sales"]
)


# ============================================================
# STEP 9: CREATE OUTPUT DIRECTORY
# ============================================================

output_folder = (
    r"D:\d\collage\ML\dataset\ml_ready"
)

os.makedirs(
    output_folder,
    exist_ok=True
)


# ============================================================
# STEP 10: SAVE SALES DATASET
# ============================================================

output_file = (
    r"D:\d\collage\ML\dataset\ml_ready"
    r"\sales_prediction_data.csv"
)

daily_sales.to_csv(
    output_file,
    index=False
)


# ============================================================
# STEP 11: DISPLAY RESULTS
# ============================================================

print("\nSales prediction dataset created.")

print(
    "Number of records:",
    len(daily_sales)
)

print(
    "Number of features:",
    len(daily_sales.columns)
)

print("\nColumns:")

for column in daily_sales.columns:
    print("-", column)

print("\nFirst 5 records:")
print(
    daily_sales.head()
)

print("\nSaved at:")
print(
    output_file
)

print("=" * 60)