import pandas as pd
import matplotlib.pyplot as plt
import os


# ============================================================
# 1. File paths
# ============================================================

input_file = r"dataset\sales_prediction\sales_predictions.csv"

output_dir = r"dataset\sales_prediction\visualizations"

os.makedirs(output_dir, exist_ok=True)


# ============================================================
# 2. Load prediction data
# ============================================================

print("Loading sales prediction results...")

df = pd.read_csv(input_file)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values("date")


print("\nDataset loaded successfully.")

print("Number of test records:", len(df))


# ============================================================
# 3. Actual vs predicted sales
# ============================================================

plt.figure(figsize=(14, 6))

plt.plot(
    df["date"],
    df["actual_sales"],
    label="Actual Sales",
    linewidth=2
)

plt.plot(
    df["date"],
    df["simple_linear_prediction"],
    label="Simple Linear Regression",
    alpha=0.7
)

plt.plot(
    df["date"],
    df["multiple_linear_prediction"],
    label="Multiple Linear Regression",
    alpha=0.7
)

plt.plot(
    df["date"],
    df["polynomial_prediction"],
    label="Polynomial Regression",
    alpha=0.7
)

plt.xlabel("Date")
plt.ylabel("Sales")

plt.title(
    "Actual vs Predicted Daily Sales"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_dir,
        "actual_vs_predicted_sales.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 4. Polynomial Regression comparison
# ============================================================

plt.figure(figsize=(14, 6))

plt.plot(
    df["date"],
    df["actual_sales"],
    label="Actual Sales",
    linewidth=2
)

plt.plot(
    df["date"],
    df["polynomial_prediction"],
    label="Polynomial Prediction",
    linewidth=1.5
)

plt.xlabel("Date")
plt.ylabel("Sales")

plt.title(
    "Actual vs Polynomial Regression Predictions"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_dir,
        "actual_vs_polynomial.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 5. Prediction errors
# ============================================================

df["simple_error"] = (
    df["actual_sales"]
    - df["simple_linear_prediction"]
)

df["multiple_error"] = (
    df["actual_sales"]
    - df["multiple_linear_prediction"]
)

df["polynomial_error"] = (
    df["actual_sales"]
    - df["polynomial_prediction"]
)


# ============================================================
# 6. Polynomial residual plot
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    df["polynomial_prediction"],
    df["polynomial_error"],
    alpha=0.6
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Predicted Sales")

plt.ylabel("Residual")

plt.title(
    "Polynomial Regression Residuals"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_dir,
        "polynomial_residuals.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 7. Model performance comparison
# ============================================================

comparison_file = (
    r"dataset\sales_prediction"
    r"\regression_model_comparison.csv"
)

results = pd.read_csv(comparison_file)


# ============================================================
# 8. MAE comparison
# ============================================================

plt.figure(figsize=(9, 5))

plt.bar(
    results["Model"],
    results["MAE"]
)

plt.ylabel("Mean Absolute Error")

plt.title(
    "Regression Models - MAE Comparison"
)

plt.xticks(
    rotation=15,
    ha="right"
)

plt.grid(
    axis="y"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_dir,
        "mae_comparison.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 9. RMSE comparison
# ============================================================

plt.figure(figsize=(9, 5))

plt.bar(
    results["Model"],
    results["RMSE"]
)

plt.ylabel("Root Mean Squared Error")

plt.title(
    "Regression Models - RMSE Comparison"
)

plt.xticks(
    rotation=15,
    ha="right"
)

plt.grid(
    axis="y"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_dir,
        "rmse_comparison.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 10. R² comparison
# ============================================================

plt.figure(figsize=(9, 5))

plt.bar(
    results["Model"],
    results["R2"]
)

plt.ylabel("R² Score")

plt.title(
    "Regression Models - R² Comparison"
)

plt.xticks(
    rotation=15,
    ha="right"
)

plt.grid(
    axis="y"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_dir,
        "r2_comparison.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 11. Save errors
# ============================================================

df.to_csv(
    os.path.join(
        output_dir,
        "sales_predictions_with_errors.csv"
    ),
    index=False
)


# ============================================================
# 12. Display summary
# ============================================================

print("\n==============================================")
print("SALES VISUALIZATION COMPLETED")
print("==============================================")

print("\nVisualizations saved at:")

print(os.path.abspath(output_dir))

print("\nFiles created:")

print("1. actual_vs_predicted_sales.png")
print("2. actual_vs_polynomial.png")
print("3. polynomial_residuals.png")
print("4. mae_comparison.png")
print("5. rmse_comparison.png")
print("6. r2_comparison.png")
print("7. sales_predictions_with_errors.csv")