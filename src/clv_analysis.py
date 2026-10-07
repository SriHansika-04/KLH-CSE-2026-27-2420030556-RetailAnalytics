# ============================================================
# CUSTOMER LIFETIME VALUE (CLV) ANALYSIS
# Retail Sales Analytics Project
# ============================================================

import os
import pandas as pd
import numpy as np

# ------------------------------------------------------------
# BLOCK 1: Define paths
# ------------------------------------------------------------

input_file = r"outputs\07_Customer_Intelligence\RFM_Customer_Value.xlsx"

output_folder = r"outputs\08_CLV"

os.makedirs(output_folder, exist_ok=True)

output_file = os.path.join(
    output_folder,
    "Customer_CLV.xlsx"
)

print("=" * 60)
print("CUSTOMER LIFETIME VALUE (CLV) ANALYSIS")
print("=" * 60)

# ------------------------------------------------------------
# BLOCK 2: Load RFM customer data
# ------------------------------------------------------------

df = pd.read_excel(
    input_file,
    sheet_name="RFM_Customer_Data"
)

print("\nRFM customer data loaded.")
print("Customers:", len(df))

# ------------------------------------------------------------
# BLOCK 3: Calculate average order value
# ------------------------------------------------------------

df["Average_Order_Value"] = (
    df["Total_Sales"] / df["Order_Count"]
)

# ------------------------------------------------------------
# BLOCK 4: Calculate purchase frequency
# ------------------------------------------------------------
# Frequency is already available from the RFM analysis.
#
# We use the customer's order count as the purchase frequency.

df["Purchase_Frequency"] = df["Order_Count"]

# ------------------------------------------------------------
# BLOCK 5: Calculate customer lifespan
# ------------------------------------------------------------
# Customer lifespan is estimated from Recency.
#
# To avoid zero values, a minimum lifespan of 1 year is used.

df["Estimated_Lifespan_Years"] = np.maximum(
    df["Recency"] / 365,
    1
)

# ------------------------------------------------------------
# BLOCK 6: Calculate CLV
# ------------------------------------------------------------
# A simple historical CLV model is used:
#
# CLV =
# Average Order Value
# × Purchase Frequency
# × Estimated Customer Lifespan
#
# This provides a transparent customer-value measure
# suitable for business intelligence analysis.

df["CLV"] = (
    df["Average_Order_Value"]
    * df["Purchase_Frequency"]
    * df["Estimated_Lifespan_Years"]
)

# ------------------------------------------------------------
# BLOCK 7: Create CLV categories
# ------------------------------------------------------------
# Customers are divided into three groups using CLV
# percentiles.

low_threshold = df["CLV"].quantile(0.33)
high_threshold = df["CLV"].quantile(0.67)


def clv_category(value):

    if value >= high_threshold:
        return "High CLV"

    elif value >= low_threshold:
        return "Medium CLV"

    else:
        return "Low CLV"


df["CLV_Category"] = df["CLV"].apply(
    clv_category
)

# ------------------------------------------------------------
# BLOCK 8: Create CLV summary
# ------------------------------------------------------------

clv_summary = df.groupby(
    "CLV_Category"
).agg(
    Customers=("Customer ID", "count"),
    Total_CLV=("CLV", "sum"),
    Average_CLV=("CLV", "mean"),
    Total_Sales=("Total_Sales", "sum"),
    Total_Profit=("Total_Profit", "sum")
).reset_index()

# ------------------------------------------------------------
# BLOCK 9: Overall CLV summary
# ------------------------------------------------------------

overall_summary = pd.DataFrame({
    "Metric": [
        "Total Customers",
        "Total CLV",
        "Average CLV",
        "Median CLV",
        "Maximum CLV",
        "Minimum CLV"
    ],

    "Value": [
        len(df),
        df["CLV"].sum(),
        df["CLV"].mean(),
        df["CLV"].median(),
        df["CLV"].max(),
        df["CLV"].min()
    ]
})

# ------------------------------------------------------------
# BLOCK 10: Save CLV results
# ------------------------------------------------------------

with pd.ExcelWriter(
    output_file,
    engine="openpyxl"
) as writer:

    df.to_excel(
        writer,
        sheet_name="Customer_CLV_Data",
        index=False
    )

    clv_summary.to_excel(
        writer,
        sheet_name="CLV_Category_Summary",
        index=False
    )

    overall_summary.to_excel(
        writer,
        sheet_name="CLV_Summary",
        index=False
    )

# ------------------------------------------------------------
# BLOCK 11: Display results
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("CLV ANALYSIS COMPLETED")
print("=" * 60)

print("\nCLV Category Distribution:")
print(
    df["CLV_Category"].value_counts()
)

print("\nCLV Summary:")
print(clv_summary)

print("\nOutput saved to:")
print(output_file)

print("\nTop 10 Customers by CLV:")

print(
    df.sort_values(
        "CLV",
        ascending=False
    )[[
        "Customer ID",
        "Total_Sales",
        "Total_Profit",
        "Order_Count",
        "Average_Order_Value",
        "CLV",
        "CLV_Category"
    ]].head(10)
)

print("\nDone!")