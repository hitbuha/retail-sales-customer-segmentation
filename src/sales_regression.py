import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# 1. File paths
# ============================================================

input_file = r"dataset\ml_ready\sales_prediction_data.csv"

output_dir = r"dataset\sales_prediction"

os.makedirs(output_dir, exist_ok=True)


# ============================================================
# 2. Load dataset
# ============================================================

print("Loading sales prediction dataset...")

df = pd.read_csv(input_file)

print("Dataset loaded successfully.")

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(list(df.columns))


# ============================================================
# 3. Convert date
# ============================================================

df["date"] = pd.to_datetime(df["date"])


# ============================================================
# 4. Sort chronologically
# ============================================================

df = df.sort_values("date").reset_index(drop=True)


# ============================================================
# 5. Define target
# ============================================================

target = "total_sales"


# ============================================================
# 6. Select features
# ============================================================

features = [
    "total_quantity",
    "transaction_count",
    "unique_customers",
    "unique_products",
    "avg_unit_price",
    "previous_day_sales",
    "month",
    "day_of_week"
]

X = df[features]

y = df[target]


print("\n==============================================")
print("REGRESSION DATA")
print("==============================================")

print("Features:")
print(features)

print("\nTarget:")
print(target)


# ============================================================
# 7. Check missing values
# ============================================================

print("\nMissing values:")

print(
    df[features + [target]].isnull().sum()
)


# ============================================================
# 8. Train-test split
# ============================================================

# Because this is time-based sales data,
# do NOT randomly shuffle the observations.

split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


print("\n==============================================")
print("TRAIN-TEST SPLIT")
print("==============================================")

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print(
    "Training period:",
    df["date"].iloc[0],
    "to",
    df["date"].iloc[split_index - 1]
)

print(
    "Testing period:",
    df["date"].iloc[split_index],
    "to",
    df["date"].iloc[-1]
)


# ============================================================
# 9. Evaluation function
# ============================================================

def evaluate_model(model_name, model, X_train, X_test, y_train, y_test):

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    mse = mean_squared_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_test,
        predictions
    )

    print("\n----------------------------------------------")
    print(model_name)
    print("----------------------------------------------")

    print(f"MAE  : {mae:.2f}")
    print(f"MSE  : {mse:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.4f}")

    return {
        "Model": model_name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "Predictions": predictions
    }


# ============================================================
# 10. Simple Linear Regression
# ============================================================

# Simple Linear Regression uses one predictor.
# Here previous_day_sales is used to predict today's sales.

simple_features = [
    "previous_day_sales"
]

simple_model = LinearRegression()

simple_result = evaluate_model(
    "Simple Linear Regression",
    simple_model,
    X_train[simple_features],
    X_test[simple_features],
    y_train,
    y_test
)


# ============================================================
# 11. Multiple Linear Regression
# ============================================================

multiple_model = LinearRegression()

multiple_result = evaluate_model(
    "Multiple Linear Regression",
    multiple_model,
    X_train,
    X_test,
    y_train,
    y_test
)


# ============================================================
# 12. Polynomial Regression
# ============================================================

polynomial_model = Pipeline([
    (
        "polynomial_features",
        PolynomialFeatures(
            degree=2,
            include_bias=False
        )
    ),
    (
        "linear_regression",
        LinearRegression()
    )
])


polynomial_result = evaluate_model(
    "Polynomial Regression",
    polynomial_model,
    X_train,
    X_test,
    y_train,
    y_test
)


# ============================================================
# 13. Create comparison table
# ============================================================

results = pd.DataFrame([
    {
        "Model": simple_result["Model"],
        "MAE": simple_result["MAE"],
        "MSE": simple_result["MSE"],
        "RMSE": simple_result["RMSE"],
        "R2": simple_result["R2"]
    },
    {
        "Model": multiple_result["Model"],
        "MAE": multiple_result["MAE"],
        "MSE": multiple_result["MSE"],
        "RMSE": multiple_result["RMSE"],
        "R2": multiple_result["R2"]
    },
    {
        "Model": polynomial_result["Model"],
        "MAE": polynomial_result["MAE"],
        "MSE": polynomial_result["MSE"],
        "RMSE": polynomial_result["RMSE"],
        "R2": polynomial_result["R2"]
    }
])


# ============================================================
# 14. Display results
# ============================================================

print("\n==============================================")
print("REGRESSION MODEL COMPARISON")
print("==============================================")

print(
    results.to_string(index=False)
)


# ============================================================
# 15. Save results
# ============================================================

results.to_csv(
    os.path.join(
        output_dir,
        "regression_model_comparison.csv"
    ),
    index=False
)


# ============================================================
# 16. Save predictions
# ============================================================

prediction_df = df.iloc[split_index:].copy()

prediction_df["actual_sales"] = y_test.values

prediction_df["simple_linear_prediction"] = (
    simple_result["Predictions"]
)

prediction_df["multiple_linear_prediction"] = (
    multiple_result["Predictions"]
)

prediction_df["polynomial_prediction"] = (
    polynomial_result["Predictions"]
)


prediction_df.to_csv(
    os.path.join(
        output_dir,
        "sales_predictions.csv"
    ),
    index=False
)


# ============================================================
# 17. Final message
# ============================================================

print("\n==============================================")
print("SALES REGRESSION COMPLETED")
print("==============================================")

print("\nFiles saved in:")

print(os.path.abspath(output_dir))