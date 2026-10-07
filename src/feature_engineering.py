# ============================================================
# RETAIL ANALYTICS PROJECT
# FEATURE ENGINEERING
# ============================================================
#
# Input:
#   outputs/01_Preprocessing/Preprocessed_Superstore.xlsx
#
# Output:
#   outputs/03_Feature_Engineering/
#
# This script:
#   1. Loads the preprocessed dataset
#   2. Validates the dataset
#   3. Creates date features
#   4. Creates shipping features
#   5. Creates sales features
#   6. Creates discount features
#   7. Creates customer-level features
#   8. Creates product-level features
#   9. Creates profit features
#  10. Creates a final feature-engineered dataset
#  11. Creates separate Excel files for each feature group
#  12. Creates a feature summary Excel file
#  13. Performs final validation
#
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os
import numpy as np
import pandas as pd


# ============================================================
# 2. PROJECT PATH
# ============================================================

BASE_PATH = (
    r"D:\OneDrive\Desktop\Specialisation\Retail-Analytics"
)


# ============================================================
# 3. INPUT PATH
# ============================================================

INPUT_PATH = os.path.join(
    BASE_PATH,
    "outputs",
    "01_Preprocessing",
    "Preprocessed_Global_Superstore.xlsx"
)


# ============================================================
# 4. OUTPUT PATHS
# ============================================================

OUTPUT_BASE = os.path.join(
    BASE_PATH,
    "outputs",
    "03_Feature_Engineering"
)


DATE_PATH = os.path.join(
    OUTPUT_BASE,
    "01_Date_Features"
)


SALES_PATH = os.path.join(
    OUTPUT_BASE,
    "02_Sales_Features"
)


CUSTOMER_PATH = os.path.join(
    OUTPUT_BASE,
    "03_Customer_Features"
)


PRODUCT_PATH = os.path.join(
    OUTPUT_BASE,
    "04_Product_Features"
)


PROFIT_PATH = os.path.join(
    OUTPUT_BASE,
    "05_Profit_Features"
)


# ============================================================
# 5. CREATE OUTPUT FOLDERS
# ============================================================

os.makedirs(
    DATE_PATH,
    exist_ok=True
)

os.makedirs(
    SALES_PATH,
    exist_ok=True
)

os.makedirs(
    CUSTOMER_PATH,
    exist_ok=True
)

os.makedirs(
    PRODUCT_PATH,
    exist_ok=True
)

os.makedirs(
    PROFIT_PATH,
    exist_ok=True
)


# ============================================================
# 6. START MESSAGE
# ============================================================

print()
print("============================================================")
print("RETAIL ANALYTICS - FEATURE ENGINEERING")
print("============================================================")


# ============================================================
# 7. CHECK INPUT FILE
# ============================================================

print()
print("===== CHECKING INPUT FILE =====")

print("Input path:")
print(INPUT_PATH)


if not os.path.exists(INPUT_PATH):

    print()
    print("ERROR: Input file was not found.")
    print()
    print("Expected file:")
    print(INPUT_PATH)
    print()

    raise FileNotFoundError(
        "Preprocessed Excel file was not found."
    )


print("Input file found successfully.")


# ============================================================
# 8. LOAD PREPROCESSED DATA
# ============================================================

print()
print("===== LOADING PREPROCESSED DATA =====")

df = pd.read_excel(
    INPUT_PATH
)


print()
print("Dataset loaded successfully.")

print(
    "Rows:",
    df.shape[0]
)

print(
    "Columns:",
    df.shape[1]
)


# ============================================================
# 9. DISPLAY ORIGINAL COLUMNS
# ============================================================

print()
print("===== ORIGINAL DATASET COLUMNS =====")

for column in df.columns:

    print(
        "-",
        column
    )


# ============================================================
# 10. REQUIRED COLUMNS
# ============================================================

required_columns = [

    "Order ID",

    "Order Date",

    "Ship Date",

    "Ship Mode",

    "Customer ID",

    "Customer Name",

    "Segment",

    "City",

    "State",

    "Country",

    "Market",

    "Region",

    "Product ID",

    "Category",

    "Sub-Category",

    "Product Name",

    "Order Priority",

    "Sales",

    "Quantity",

    "Discount",

    "Profit"

]


# ============================================================
# 11. CHECK REQUIRED COLUMNS
# ============================================================

print()
print("===== CHECKING REQUIRED COLUMNS =====")


missing_required_columns = [

    column

    for column in required_columns

    if column not in df.columns

]


if len(missing_required_columns) > 0:

    print()
    print(
        "ERROR: The following required columns are missing:"
    )

    for column in missing_required_columns:

        print(
            "-",
            column
        )

    print()

    raise ValueError(
        "Required columns are missing from the dataset."
    )


print(
    "All required columns are available."
)


# ============================================================
# 12. DATE CONVERSION
# ============================================================

print()
print("============================================================")
print("DATE CONVERSION")
print("============================================================")


df["Order Date"] = pd.to_datetime(
    df["Order Date"],
    errors="coerce"
)


df["Ship Date"] = pd.to_datetime(
    df["Ship Date"],
    errors="coerce"
)


print()
print("Order Date type:")
print(
    df["Order Date"].dtype
)


print()
print("Ship Date type:")
print(
    df["Ship Date"].dtype
)


print()
print("Invalid Order Dates:")
print(
    df["Order Date"].isna().sum()
)


print()
print("Invalid Ship Dates:")
print(
    df["Ship Date"].isna().sum()
)


# ============================================================
# 13. DATE FEATURES
# ============================================================

print()
print("============================================================")
print("BLOCK 1 - DATE FEATURES")
print("============================================================")


# Year

df["Order Year"] = (
    df["Order Date"].dt.year
)


# Month number

df["Order Month"] = (
    df["Order Date"].dt.month
)


# Month name

df["Order Month Name"] = (
    df["Order Date"].dt.month_name()
)


# Quarter

df["Order Quarter"] = (
    df["Order Date"].dt.quarter
)


# Week number

df["Order Week"] = (

    df["Order Date"]
    .dt.isocalendar()
    .week
    .astype("Int64")

)


# Day of week number

df["Order Day of Week"] = (
    df["Order Date"].dt.dayofweek
)


# Day name

df["Order Day Name"] = (
    df["Order Date"].dt.day_name()
)


# Weekend indicator

df["Is Weekend"] = (

    df["Order Day of Week"] >= 5

).astype(int)


# Shipping days

df["Shipping Days"] = (

    df["Ship Date"]
    -
    df["Order Date"]

).dt.days


print()
print("Date features created:")

date_features = [

    "Order Year",
    "Order Month",
    "Order Month Name",
    "Order Quarter",
    "Order Week",
    "Order Day of Week",
    "Order Day Name",
    "Is Weekend",
    "Shipping Days"

]


for feature in date_features:

    print(
        "-",
        feature
    )


# ============================================================
# 14. SAVE DATE FEATURES
# ============================================================

date_output = df[
    [
        "Order ID",
        "Order Date",
        "Ship Date",
        "Order Year",
        "Order Month",
        "Order Month Name",
        "Order Quarter",
        "Order Week",
        "Order Day of Week",
        "Order Day Name",
        "Is Weekend",
        "Shipping Days"
    ]
]


date_output_path = os.path.join(
    DATE_PATH,
    "Date_Features.xlsx"
)


date_output.to_excel(
    date_output_path,
    index=False
)


print()
print(
    "Date features saved:"
)

print(
    date_output_path
)


# ============================================================
# 15. SALES FEATURES
# ============================================================

print()
print("============================================================")
print("BLOCK 2 - SALES FEATURES")
print("============================================================")


# Sales per quantity

df["Sales per Quantity"] = np.where(

    df["Quantity"] != 0,

    df["Sales"] / df["Quantity"],

    0

)


# Discount amount

df["Discount Amount"] = (

    df["Sales"]
    *
    df["Discount"]

)


# Net sales

df["Net Sales"] = (

    df["Sales"]
    -
    df["Discount Amount"]

)


# Quantity category

df["Quantity Category"] = pd.cut(

    df["Quantity"],

    bins=[
        -np.inf,
        1,
        3,
        5,
        np.inf
    ],

    labels=[
        "Low",
        "Medium",
        "High",
        "Very High"
    ]

)


# Discount category

df["Discount Category"] = pd.cut(

    df["Discount"],

    bins=[
        -np.inf,
        0,
        0.10,
        0.30,
        np.inf
    ],

    labels=[
        "No Discount",
        "Low Discount",
        "Medium Discount",
        "High Discount"
    ]

)


print()
print("Sales features created:")

sales_features = [

    "Sales per Quantity",
    "Discount Amount",
    "Net Sales",
    "Quantity Category",
    "Discount Category"

]


for feature in sales_features:

    print(
        "-",
        feature
    )


# ============================================================
# 16. SAVE SALES FEATURES
# ============================================================

sales_output = df[
    [
        "Order ID",
        "Sales",
        "Quantity",
        "Discount",
        "Sales per Quantity",
        "Discount Amount",
        "Net Sales",
        "Quantity Category",
        "Discount Category"
    ]
]


sales_output_path = os.path.join(
    SALES_PATH,
    "Sales_Features.xlsx"
)


sales_output.to_excel(
    sales_output_path,
    index=False
)


print()
print(
    "Sales features saved:"
)

print(
    sales_output_path
)


# ============================================================
# 17. CUSTOMER FEATURES
# ============================================================

print()
print("============================================================")
print("BLOCK 3 - CUSTOMER FEATURES")
print("============================================================")


# Unique orders for each customer

customer_order_count = (

    df
    .groupby("Customer ID")["Order ID"]
    .transform("nunique")

)


df["Customer Order Count"] = (
    customer_order_count
)


# Total sales for each customer

customer_total_sales = (

    df
    .groupby("Customer ID")["Sales"]
    .transform("sum")

)


df["Customer Total Sales"] = (
    customer_total_sales
)


# Total profit for each customer

customer_total_profit = (

    df
    .groupby("Customer ID")["Profit"]
    .transform("sum")

)


df["Customer Total Profit"] = (
    customer_total_profit
)


# Average order value

df["Customer Average Order Value"] = np.where(

    df["Customer Order Count"] != 0,

    df["Customer Total Sales"]
    /
    df["Customer Order Count"],

    0

)


# Customer profitability

df["Customer Profitability"] = np.where(

    df["Customer Total Profit"] >= 0,

    "Profitable",

    "Loss Making"

)


print()
print("Customer features created:")

customer_features = [

    "Customer Order Count",
    "Customer Total Sales",
    "Customer Total Profit",
    "Customer Average Order Value",
    "Customer Profitability"

]


for feature in customer_features:

    print(
        "-",
        feature
    )


# ============================================================
# 18. SAVE CUSTOMER FEATURES
# ============================================================

customer_output = (

    df[
        [
            "Customer ID",
            "Customer Name",
            "Segment",
            "Customer Order Count",
            "Customer Total Sales",
            "Customer Total Profit",
            "Customer Average Order Value",
            "Customer Profitability"
        ]
    ]

    .drop_duplicates(
        subset=["Customer ID"]
    )

    .sort_values(
        "Customer Total Sales",
        ascending=False
    )

)


customer_output_path = os.path.join(
    CUSTOMER_PATH,
    "Customer_Features.xlsx"
)


customer_output.to_excel(
    customer_output_path,
    index=False
)


print()
print(
    "Customer features saved:"
)

print(
    customer_output_path
)


# ============================================================
# 19. PRODUCT FEATURES
# ============================================================

print()
print("============================================================")
print("BLOCK 4 - PRODUCT FEATURES")
print("============================================================")


# Number of unique orders for product

product_order_count = (

    df
    .groupby("Product ID")["Order ID"]
    .transform("nunique")

)


df["Product Order Count"] = (
    product_order_count
)


# Total product sales

product_total_sales = (

    df
    .groupby("Product ID")["Sales"]
    .transform("sum")

)


df["Product Total Sales"] = (
    product_total_sales
)


# Total product profit

product_total_profit = (

    df
    .groupby("Product ID")["Profit"]
    .transform("sum")

)


df["Product Total Profit"] = (
    product_total_profit
)


# Average product sales per order

df["Product Average Sales"] = np.where(

    df["Product Order Count"] != 0,

    df["Product Total Sales"]
    /
    df["Product Order Count"],

    0

)


# Product performance

df["Product Performance"] = np.where(

    df["Product Total Profit"] >= 0,

    "Profitable",

    "Loss Making"

)


print()
print("Product features created:")

product_features = [

    "Product Order Count",
    "Product Total Sales",
    "Product Total Profit",
    "Product Average Sales",
    "Product Performance"

]


for feature in product_features:

    print(
        "-",
        feature
    )


# ============================================================
# 20. SAVE PRODUCT FEATURES
# ============================================================

product_output = (

    df[
        [
            "Product ID",
            "Product Name",
            "Category",
            "Sub-Category",
            "Product Order Count",
            "Product Total Sales",
            "Product Total Profit",
            "Product Average Sales",
            "Product Performance"
        ]
    ]

    .drop_duplicates(
        subset=["Product ID"]
    )

    .sort_values(
        "Product Total Sales",
        ascending=False
    )

)


product_output_path = os.path.join(
    PRODUCT_PATH,
    "Product_Features.xlsx"
)


product_output.to_excel(
    product_output_path,
    index=False
)


print()
print(
    "Product features saved:"
)

print(
    product_output_path
)


# ============================================================
# 21. PROFIT FEATURES
# ============================================================

print()
print("============================================================")
print("BLOCK 5 - PROFIT FEATURES")
print("============================================================")


# Profit margin

df["Profit Margin"] = np.where(

    df["Sales"] != 0,

    (
        df["Profit"]
        /
        df["Sales"]
    )
    *
    100,

    0

)


# Profit status

df["Profit Status"] = np.where(

    df["Profit"] > 0,

    "Profit",

    np.where(

        df["Profit"] < 0,

        "Loss",

        "Break-even"

    )

)


# Profit category

df["Profit Category"] = pd.cut(

    df["Profit Margin"],

    bins=[
        -np.inf,
        0,
        10,
        25,
        np.inf
    ],

    labels=[
        "Loss",
        "Low Margin",
        "Medium Margin",
        "High Margin"
    ]

)


print()
print("Profit features created:")

profit_features = [

    "Profit Margin",
    "Profit Status",
    "Profit Category"

]


for feature in profit_features:

    print(
        "-",
        feature
    )


# ============================================================
# 22. SAVE PROFIT FEATURES
# ============================================================

profit_output = df[
    [
        "Order ID",
        "Sales",
        "Profit",
        "Profit Margin",
        "Profit Status",
        "Profit Category",
        "Category",
        "Sub-Category"
    ]
]


profit_output_path = os.path.join(
    PROFIT_PATH,
    "Profit_Features.xlsx"
)


profit_output.to_excel(
    profit_output_path,
    index=False
)


print()
print(
    "Profit features saved:"
)

print(
    profit_output_path
)


# ============================================================
# 23. FINAL FEATURE-ENGINEERED DATASET
# ============================================================

print()
print("============================================================")
print("FINAL FEATURE-ENGINEERED DATASET")
print("============================================================")


print()
print(
    "Rows:",
    df.shape[0]
)


print(
    "Columns:",
    df.shape[1]
)


# ============================================================
# 24. IDENTIFY NEW FEATURES
# ============================================================

engineered_columns = [

    column

    for column in df.columns

    if column not in required_columns

]


print()
print(
    "Number of engineered features:",
    len(engineered_columns)
)


print()
print("Engineered features:")


for column in engineered_columns:

    print(
        "-",
        column
    )


# ============================================================
# 25. CHECK MISSING VALUES
# ============================================================

print()
print("===== FINAL MISSING VALUE CHECK =====")


total_missing = (

    df
    .isnull()
    .sum()
    .sum()

)


print(
    "Total missing values:",
    total_missing
)


# ============================================================
# 26. CHECK DUPLICATES
# ============================================================

print()
print("===== FINAL DUPLICATE CHECK =====")


total_duplicates = (

    df
    .duplicated()
    .sum()

)


print(
    "Duplicate rows:",
    total_duplicates
)


# ============================================================
# 27. SAVE COMPLETE FEATURE-ENGINEERED DATASET
# ============================================================

final_output_path = os.path.join(

    OUTPUT_BASE,

    "Feature_Engineered_Superstore.xlsx"

)


df.to_excel(

    final_output_path,

    index=False

)


print()
print(
    "Complete feature-engineered dataset saved:"
)

print(
    final_output_path
)


# ============================================================
# 28. CREATE FEATURE SUMMARY
# ============================================================

print()
print("===== CREATING FEATURE SUMMARY =====")


feature_summary = pd.DataFrame({

    "Feature": [

        "Order Year",
        "Order Month",
        "Order Month Name",
        "Order Quarter",
        "Order Week",
        "Order Day of Week",
        "Order Day Name",
        "Is Weekend",
        "Shipping Days",

        "Sales per Quantity",
        "Discount Amount",
        "Net Sales",
        "Quantity Category",
        "Discount Category",

        "Customer Order Count",
        "Customer Total Sales",
        "Customer Total Profit",
        "Customer Average Order Value",
        "Customer Profitability",

        "Product Order Count",
        "Product Total Sales",
        "Product Total Profit",
        "Product Average Sales",
        "Product Performance",

        "Profit Margin",
        "Profit Status",
        "Profit Category"

    ],

    "Description": [

        "Year extracted from Order Date",

        "Month number extracted from Order Date",

        "Month name extracted from Order Date",

        "Quarter extracted from Order Date",

        "ISO week number of the order",

        "Numeric day of week",

        "Name of the day on which order was placed",

        "Indicates whether order was placed on weekend",

        "Number of days between order and shipment",

        "Sales generated per unit quantity",

        "Estimated discount amount",

        "Sales after estimated discount",

        "Categorical quantity level",

        "Categorical discount level",

        "Number of unique orders placed by customer",

        "Total sales generated by customer",

        "Total profit generated by customer",

        "Average sales value per customer order",

        "Customer classified based on total profit",

        "Number of unique orders containing product",

        "Total sales generated by product",

        "Total profit generated by product",

        "Average product sales per order",

        "Product classified based on total profit",

        "Profit as percentage of sales",

        "Profit, loss, or break-even classification",

        "Profit margin classification"

    ]

})


feature_summary_path = os.path.join(

    OUTPUT_BASE,

    "Feature_Summary.xlsx"

)


feature_summary.to_excel(

    feature_summary_path,

    index=False

)


print()
print(
    "Feature summary saved:"
)

print(
    feature_summary_path
)


# ============================================================
# 29. FINAL VALIDATION
# ============================================================

print()
print("============================================================")
print("FINAL FEATURE ENGINEERING VALIDATION")
print("============================================================")


print()
print(
    "Original row count:",
    df.shape[0]
)


print(
    "Final column count:",
    df.shape[1]
)


print(
    "Engineered feature count:",
    len(engineered_columns)
)


print(
    "Missing values:",
    total_missing
)


print(
    "Duplicate rows:",
    total_duplicates
)


# ============================================================
# 30. OUTPUT FILE CHECK
# ============================================================

print()
print("===== OUTPUT FILE CHECK =====")


output_files = [

    date_output_path,

    sales_output_path,

    customer_output_path,

    product_output_path,

    profit_output_path,

    final_output_path,

    feature_summary_path

]


for file_path in output_files:

    if os.path.exists(file_path):

        print(
            "[OK]",
            file_path
        )

    else:

        print(
            "[MISSING]",
            file_path
        )


# ============================================================
# 31. COMPLETION MESSAGE
# ============================================================

print()
print("============================================================")
print("FEATURE ENGINEERING COMPLETED SUCCESSFULLY")
print("============================================================")

print()
print("All feature-engineering Excel files have been saved.")

print()
print("Main final dataset:")
print(final_output_path)

print()
print("Feature summary:")
print(feature_summary_path)

print()
print("============================================================")