import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Retail ML Analytics Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dataset" / "final_output"

CUSTOMER_FILE = DATA_DIR / "final_customer_segments.csv"
PROFILE_FILE = DATA_DIR / "final_cluster_profiles.csv"
SALES_FILE = DATA_DIR / "final_sales_predictions.csv"
MODEL_FILE = DATA_DIR / "final_model_comparison.csv"


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    customers = pd.read_csv(CUSTOMER_FILE)
    profiles = pd.read_csv(PROFILE_FILE)
    sales = pd.read_csv(SALES_FILE)
    models = pd.read_csv(MODEL_FILE)

    sales["date"] = pd.to_datetime(sales["date"])

    return customers, profiles, sales, models


try:
    customers, profiles, sales, models = load_data()

except Exception as e:

    st.error("Unable to load dashboard data.")
    st.code(str(e))
    st.stop()


# =========================================================
# TITLE
# =========================================================

st.title("🛒 Retail Sales Prediction & Customer Segmentation")

st.markdown(
    """
    ### Machine Learning Analytics Dashboard

    This dashboard integrates two ML components:

    - **Customer Segmentation using PCA + K-Means**
    - **Sales Prediction using Regression Models**
    """
)

st.divider()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("📊 Dashboard Navigation")

page = st.sidebar.radio(
    "Select Section",
    [
        "Overview",
        "Customer Segmentation",
        "Sales Prediction",
        "Model Comparison",
        "Data Explorer"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    """
    **Project:** Retail Sales Prediction + Customer Segmentation

    **Dataset:** UCI Online Retail II

    **ML Techniques:**
    - Data Preprocessing
    - Feature Engineering
    - PCA
    - K-Means
    - Linear Regression
    - Polynomial Regression
    """
)


# =========================================================
# COMMON METRICS
# =========================================================

total_customers = len(customers)

cluster_counts = customers["cluster"].value_counts()

cluster_0_count = int(cluster_counts.get(0, 0))
cluster_1_count = int(cluster_counts.get(1, 0))

total_test_records = len(sales)

avg_daily_sales = sales["actual_sales"].mean()

total_test_sales = sales["actual_sales"].sum()


# =========================================================
# PAGE 1 — OVERVIEW
# =========================================================

if page == "Overview":

    st.header("📌 Project Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Customers",
        f"{total_customers:,}"
    )

    col2.metric(
        "Customer Clusters",
        "2"
    )

    col3.metric(
        "Sales Test Records",
        f"{total_test_records:,}"
    )

    col4.metric(
        "PCA Components",
        "3"
    )

    st.divider()

    # -----------------------------------------------------
    # CUSTOMER DISTRIBUTION
    # -----------------------------------------------------

    st.subheader("Customer Segment Distribution")

    segment_data = pd.DataFrame({
        "Segment": [
            "High-Engagement / High-Value",
            "Lower-Engagement / Lower-Value"
        ],
        "Customers": [
            cluster_0_count,
            cluster_1_count
        ]
    })

    col1, col2 = st.columns(2)

    with col1:

        fig = px.pie(
            segment_data,
            names="Segment",
            values="Customers",
            title="Customer Distribution",
            hole=0.45
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

    with col2:

        fig = px.bar(
            segment_data,
            x="Segment",
            y="Customers",
            title="Customers by Segment",
            text="Customers"
        )

        fig.update_traces(
            textposition="outside"
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

    # -----------------------------------------------------
    # SALES TREND
    # -----------------------------------------------------

    st.subheader("Sales Prediction Overview")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=sales["date"],
            y=sales["actual_sales"],
            mode="lines",
            name="Actual Sales"
        )
    )

    if "polynomial_prediction" in sales.columns:

        fig.add_trace(
            go.Scatter(
                x=sales["date"],
                y=sales["polynomial_prediction"],
                mode="lines",
                name="Polynomial Prediction"
            )
        )

    fig.update_layout(
        title="Actual vs Predicted Sales",
        xaxis_title="Date",
        yaxis_title="Sales"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )


# =========================================================
# PAGE 2 — CUSTOMER SEGMENTATION
# =========================================================

elif page == "Customer Segmentation":

    st.header("👥 Customer Segmentation")

    st.markdown(
        """
        Customer segmentation was performed using behavioral
        features followed by PCA dimensionality reduction and
        K-Means clustering.
        """
    )

    # -----------------------------------------------------
    # METRICS
    # -----------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Customers",
        f"{total_customers:,}"
    )

    col2.metric(
        "Cluster 0",
        f"{cluster_0_count:,}"
    )

    col3.metric(
        "Cluster 1",
        f"{cluster_1_count:,}"
    )

    col4.metric(
        "PCA Components",
        "3"
    )

    st.divider()

    # -----------------------------------------------------
    # CLUSTER PROFILE
    # -----------------------------------------------------

    st.subheader("Cluster Profiles")

    display_profiles = profiles.copy()

    st.dataframe(
        display_profiles,
        width="stretch",
        hide_index=True
    )

    # -----------------------------------------------------
    # RECENCY VS MONETARY
    # -----------------------------------------------------

    st.subheader("Customer Segmentation Visualization")

    fig = px.scatter(
        customers,
        x="recency",
        y="monetary",
        color="segment",
        hover_data=[
            "CustomerID",
            "frequency",
            "average_order_value",
            "unique_products"
        ],
        title="Recency vs Monetary Value"
    )

    fig.update_layout(
        xaxis_title="Recency (Days)",
        yaxis_title="Monetary Value"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

    # -----------------------------------------------------
    # FREQUENCY VS MONETARY
    # -----------------------------------------------------

    fig = px.scatter(
        customers,
        x="frequency",
        y="monetary",
        color="segment",
        hover_data=[
            "CustomerID",
            "recency",
            "average_order_value"
        ],
        title="Frequency vs Monetary Value"
    )

    fig.update_layout(
        xaxis_title="Purchase Frequency",
        yaxis_title="Monetary Value"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )


# =========================================================
# PAGE 3 — SALES PREDICTION
# =========================================================

elif page == "Sales Prediction":

    st.header("📈 Sales Prediction")

    st.markdown(
        """
        Sales prediction was evaluated using a chronological
        train/test split and three regression approaches.
        """
    )

    # -----------------------------------------------------
    # SALES METRICS
    # -----------------------------------------------------

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Test Records",
        f"{len(sales):,}"
    )

    col2.metric(
        "Average Actual Sales",
        f"{avg_daily_sales:,.2f}"
    )

    col3.metric(
        "Total Test Sales",
        f"{total_test_sales:,.2f}"
    )

    st.divider()

    # -----------------------------------------------------
    # MODEL SELECTION
    # -----------------------------------------------------

    model_options = list(sales.columns)

    prediction_columns = [
        col for col in model_options
        if "prediction" in col.lower()
    ]

    selected_model = st.selectbox(
        "Select Prediction Model",
        prediction_columns
    )

    # -----------------------------------------------------
    # ACTUAL VS PREDICTED
    # -----------------------------------------------------

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=sales["date"],
            y=sales["actual_sales"],
            mode="lines",
            name="Actual Sales"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=sales["date"],
            y=sales[selected_model],
            mode="lines",
            name=selected_model
        )
    )

    fig.update_layout(
        title=f"Actual vs {selected_model}",
        xaxis_title="Date",
        yaxis_title="Sales"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

    # -----------------------------------------------------
    # SALES TABLE
    # -----------------------------------------------------

    st.subheader("Sales Prediction Data")

    st.dataframe(
        sales[
            [
                "date",
                "actual_sales"
            ] + prediction_columns
        ],
        width="stretch",
        hide_index=True
    )


# =========================================================
# PAGE 4 — MODEL COMPARISON
# =========================================================

elif page == "Model Comparison":

    st.header("🤖 Regression Model Comparison")

    st.dataframe(
        models,
        width="stretch",
        hide_index=True
    )

    st.divider()

    # -----------------------------------------------------
    # MAE
    # -----------------------------------------------------

    st.subheader("MAE Comparison")

    fig = px.bar(
        models,
        x="Model",
        y="MAE",
        text="MAE",
        title="Mean Absolute Error"
    )

    fig.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

    # -----------------------------------------------------
    # RMSE
    # -----------------------------------------------------

    st.subheader("RMSE Comparison")

    fig = px.bar(
        models,
        x="Model",
        y="RMSE",
        text="RMSE",
        title="Root Mean Squared Error"
    )

    fig.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

    # -----------------------------------------------------
    # R2
    # -----------------------------------------------------

    st.subheader("R² Comparison")

    fig = px.bar(
        models,
        x="Model",
        y="R2",
        text="R2",
        title="R² Score"
    )

    fig.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )


# =========================================================
# PAGE 5 — DATA EXPLORER
# =========================================================

elif page == "Data Explorer":

    st.header("🔎 Data Explorer")

    st.subheader("Customer Segmentation Dataset")

    st.write(
        f"Rows: {customers.shape[0]:,} | "
        f"Columns: {customers.shape[1]}"
    )

    st.dataframe(
        customers,
        width="stretch",
        hide_index=True
    )

    st.divider()

    st.subheader("Sales Prediction Dataset")

    st.write(
        f"Rows: {sales.shape[0]:,} | "
        f"Columns: {sales.shape[1]}"
    )

    st.dataframe(
        sales,
        width="stretch",
        hide_index=True
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Retail Sales Prediction & Customer Segmentation | "
    "Machine Learning Project"
)
