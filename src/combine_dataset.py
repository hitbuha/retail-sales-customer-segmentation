import pandas as pd

# Read 2009-2010 dataset
df1 = pd.read_csv(
    "D:\d\collage\ML\.venv\dataset/online_retail_09_10.csv",
    encoding="latin1"
)

# Read 2010-2011 dataset
df2 = pd.read_csv(
    "D:\d\collage\ML\.venv\dataset/online_retail_10_11.csv",
    encoding="latin1"
)

# Combine both datasets
df = pd.concat([df1, df2], ignore_index=True)

# Display basic information
print("2009-2010 rows:", len(df1))
print("2010-2011 rows:", len(df2))
print("Combined rows:", len(df))

print("\nColumns:")
print(df.columns.tolist())

# Save combined dataset
df.to_csv(
    "D:\d\collage\ML\.venv\dataset/online_retail_II_combined.csv",
    index=False
)

print("\nCombined dataset created successfully!")