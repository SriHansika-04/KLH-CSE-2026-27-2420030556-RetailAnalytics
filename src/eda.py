# ============================================================
# BLOCK 1: LOAD PREPROCESSED GLOBAL SUPERSTORE DATASET
# ============================================================

# WHAT THIS BLOCK DOES:
# Loads the preprocessed Global Superstore Excel dataset
# created during the preprocessing stage.
#
# EXPECTED RESULT:
# The dataset should contain 51,290 rows and 24 columns.
# The Order Date and Ship Date columns are converted to
# datetime format for further EDA and time-based analysis.


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


# --------------------------------------------
# LOAD PREPROCESSED DATASET
# --------------------------------------------

input_path = (
    r"D:\OneDrive\Desktop\Specialisation\Retail-Analytics"
    r"\outputs\01_Preprocessing"
    r"\Preprocessed_Global_Superstore.xlsx"
)

df = pd.read_excel(input_path)


# --------------------------------------------
# CONVERT DATE COLUMNS
# --------------------------------------------

df["Order Date"] = pd.to_datetime(
    df["Order Date"]
)

df["Ship Date"] = pd.to_datetime(
    df["Ship Date"]
)


# --------------------------------------------
# DISPLAY DATASET INFORMATION
# --------------------------------------------

print("\n============================================")
print("EDA - DATASET LOADED")
print("============================================")

print("\nDataset loaded successfully!")

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nColumn names:")
print(df.columns.tolist())

print("\nDate columns:")
print(df[["Order Date", "Ship Date"]].dtypes)   

# ============================================================
# BLOCK 2: DATASET OVERVIEW & DATA QUALITY ASSESSMENT
# ============================================================

# WHAT THIS BLOCK DOES:
# Provides an overall understanding of the dataset by checking:
# 1. Dataset dimensions
# 2. Column names
# 3. Data types
# 4. Missing values
# 5. Unique values
# 6. Duplicate records
#
# EXPECTED RESULT:
# - Confirms the dataset contains 51,290 rows and 24 columns.
# - Identifies the type and number of values in each column.
# - Confirms whether missing values or duplicate records remain.
#
# This establishes the overall data quality before performing
# detailed EDA.


print("\n============================================")
print("BLOCK 2: DATASET OVERVIEW")
print("============================================")


# --------------------------------------------
# DATASET INFORMATION
# --------------------------------------------

print("\nDataset Shape:")
print(df.shape)


print("\nColumn Names:")
for column in df.columns:
    print("-", column)


print("\nData Types:")
print(df.dtypes)


# --------------------------------------------
# DATA QUALITY ASSESSMENT
# --------------------------------------------

print("\n============================================")
print("DATA QUALITY ASSESSMENT")
print("============================================")


# Missing values
missing_values = df.isnull().sum()

print("\nMissing Values:")
print(missing_values[missing_values > 0])


# Total missing values
total_missing = df.isnull().sum().sum()

print("\nTotal Missing Values:", total_missing)


# Duplicate records
duplicate_records = df.duplicated().sum()

print("\nDuplicate Records:", duplicate_records)


# Unique values
print("\nUnique Values Per Column:")
print(df.nunique())


# --------------------------------------------
# CREATE DATA QUALITY SUMMARY
# --------------------------------------------

data_quality_summary = pd.DataFrame({
    "Column": df.columns,
    "Data_Type": df.dtypes.astype(str).values,
    "Missing_Values": df.isnull().sum().values,
    "Unique_Values": df.nunique().values
})


print("\n===== DATA QUALITY SUMMARY =====")
print(data_quality_summary.to_string(index=False))

# ============================================================
# BLOCK 3: UNIVARIATE ANALYSIS
# ============================================================

# WHAT THIS BLOCK DOES:
# Analyzes individual variables separately to understand
# their distribution, frequency and overall pattern.
#
# Numerical variables:
# Sales, Quantity, Discount and Profit
# are analyzed using histograms and boxplots.
#
# Categorical variables:
# Segment, Category, Ship Mode and Order Priority
# are analyzed using frequency counts and bar charts.
#
# EXPECTED RESULT:
# The plots help identify common values, distribution patterns,
# skewness and unusual observations in the dataset.


# --------------------------------------------
# CREATE FIGURE OUTPUT FOLDER
# --------------------------------------------

figure_path = (
    r"D:\OneDrive\Desktop\Specialisation\Retail-Analytics"
    r"\outputs\02_EDA"
)

os.makedirs(figure_path, exist_ok=True)


# ============================================================
# 3.1 NUMERICAL VARIABLE DISTRIBUTIONS
# ============================================================

numerical_columns = [
    "Sales",
    "Quantity",
    "Discount",
    "Profit"
]


for column in numerical_columns:

    plt.figure(figsize=(10, 6))

    plt.hist(
        df[column],
        bins=30,
        edgecolor="black"
    )

    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            figure_path,
            f"Univariate_{column}.png"
        )
    )

    plt.show()


# ============================================================
# 3.2 CATEGORICAL VARIABLE ANALYSIS
# ============================================================

categorical_columns = [
    "Segment",
    "Category",
    "Ship Mode",
    "Order Priority"
]


for column in categorical_columns:

    value_counts = df[column].value_counts()

    print("\n============================================")
    print(f"{column.upper()} FREQUENCY")
    print("============================================")

    print(value_counts)

    plt.figure(figsize=(10, 6))

    value_counts.plot(
        kind="bar"
    )

    plt.title(f"{column} Distribution")
    plt.xlabel(column)
    plt.ylabel("Number of Records")

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            figure_path,
            f"Univariate_{column.replace(' ', '_')}.png"
        )
    )

    plt.show()


print("\n===== UNIVARIATE ANALYSIS COMPLETED =====")
# ============================================================
# BLOCK 4: BIVARIATE ANALYSIS
# ============================================================

# WHAT THIS BLOCK DOES:
# Bivariate analysis studies the relationship between two
# variables at a time.
#
# ANALYSES PERFORMED:
# 1. Sales vs Profit
# 2. Discount vs Profit
# 3. Sales by Category
# 4. Profit by Category
# 5. Sales by Segment
# 6. Profit by Segment
#
# EXPECTED RESULTS:
# - Sales vs Profit shows whether higher sales are associated
#   with higher or lower profit.
# - Discount vs Profit helps observe the effect of discounting
#   on profitability.
# - Category analysis identifies which categories generate
#   the highest sales and profit.
# - Segment analysis identifies which customer segments
#   contribute most to sales and profit.
#
# OUTPUT:
# Every chart is saved individually as a PNG file inside:
# outputs/02_EDA/02_Bivariate/


# ============================================================
# CREATE BIVARIATE OUTPUT FOLDER
# ============================================================

bivariate_path = (
    r"D:\OneDrive\Desktop\Specialisation\Retail-Analytics"
    r"\outputs\02_EDA\02_Bivariate"
)

os.makedirs(bivariate_path, exist_ok=True)


print("\n============================================")
print("BLOCK 4: BIVARIATE ANALYSIS")
print("============================================")


# ============================================================
# 4.1 SALES VS PROFIT
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    df["Sales"],
    df["Profit"],
    alpha=0.5
)

plt.title("Sales vs Profit")
plt.xlabel("Sales")
plt.ylabel("Profit")
plt.grid(True, alpha=0.2)

plt.tight_layout()

plt.savefig(
    os.path.join(
        bivariate_path,
        "01_Sales_vs_Profit.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# 4.2 DISCOUNT VS PROFIT
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    df["Discount"],
    df["Profit"],
    alpha=0.5
)

plt.title("Discount vs Profit")
plt.xlabel("Discount")
plt.ylabel("Profit")
plt.grid(True, alpha=0.2)

plt.tight_layout()

plt.savefig(
    os.path.join(
        bivariate_path,
        "02_Discount_vs_Profit.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# 4.3 SALES BY CATEGORY
# ============================================================

category_sales = (
    df.groupby("Category")["Sales"]
    .sum()
    .sort_values(ascending=False)
)

print("\n===== SALES BY CATEGORY =====")
print(category_sales)


plt.figure(figsize=(10, 6))

category_sales.plot(kind="bar")

plt.title("Total Sales by Category")
plt.xlabel("Category")
plt.ylabel("Total Sales")
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.2)

plt.tight_layout()

plt.savefig(
    os.path.join(
        bivariate_path,
        "03_Sales_by_Category.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# 4.4 PROFIT BY CATEGORY
# ============================================================

category_profit = (
    df.groupby("Category")["Profit"]
    .sum()
    .sort_values(ascending=False)
)

print("\n===== PROFIT BY CATEGORY =====")
print(category_profit)


plt.figure(figsize=(10, 6))

category_profit.plot(kind="bar")

plt.title("Total Profit by Category")
plt.xlabel("Category")
plt.ylabel("Total Profit")
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.2)

plt.tight_layout()

plt.savefig(
    os.path.join(
        bivariate_path,
        "04_Profit_by_Category.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# 4.5 SALES BY SEGMENT
# ============================================================

segment_sales = (
    df.groupby("Segment")["Sales"]
    .sum()
    .sort_values(ascending=False)
)

print("\n===== SALES BY SEGMENT =====")
print(segment_sales)


plt.figure(figsize=(10, 6))

segment_sales.plot(kind="bar")

plt.title("Total Sales by Customer Segment")
plt.xlabel("Customer Segment")
plt.ylabel("Total Sales")
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.2)

plt.tight_layout()

plt.savefig(
    os.path.join(
        bivariate_path,
        "05_Sales_by_Segment.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# 4.6 PROFIT BY SEGMENT
# ============================================================

segment_profit = (
    df.groupby("Segment")["Profit"]
    .sum()
    .sort_values(ascending=False)
)

print("\n===== PROFIT BY SEGMENT =====")
print(segment_profit)


plt.figure(figsize=(10, 6))

segment_profit.plot(kind="bar")

plt.title("Total Profit by Customer Segment")
plt.xlabel("Customer Segment")
plt.ylabel("Total Profit")
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.2)

plt.tight_layout()

plt.savefig(
    os.path.join(
        bivariate_path,
        "06_Profit_by_Segment.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# COMPLETION MESSAGE
# ============================================================

print("\n============================================")
print("BIVARIATE ANALYSIS COMPLETED")
print("============================================")

print("\nCharts saved individually in:")
print(bivariate_path)

# ============================================================
# BLOCK 5: MULTIVARIATE ANALYSIS
# ============================================================

# WHAT THIS BLOCK DOES:
# Multivariate analysis examines three or more variables
# together to understand more complex relationships.
#
# ANALYSES PERFORMED:
# 1. Sales vs Profit with Discount represented by point size
# 2. Sales vs Profit across Customer Segments
# 3. Sales vs Profit across Product Categories
# 4. Sales and Profit by Category and Segment
#
# EXPECTED RESULTS:
# - Identify how Sales, Profit and Discount interact.
# - Compare the relationship between Sales and Profit
#   across different customer segments.
# - Compare Sales and Profit patterns across categories.
# - Identify which Category-Segment combinations perform
#   better in terms of Sales and Profit.
#
# OUTPUT:
# Every visualization is saved individually inside:
# outputs/02_EDA/03_Multivariate/


# ============================================================
# CREATE MULTIVARIATE OUTPUT FOLDER
# ============================================================

multivariate_path = (
    r"D:\OneDrive\Desktop\Specialisation\Retail-Analytics"
    r"\outputs\02_EDA\03_Multivariate"
)

os.makedirs(multivariate_path, exist_ok=True)


print("\n============================================")
print("BLOCK 5: MULTIVARIATE ANALYSIS")
print("============================================")


# ============================================================
# 5.1 SALES VS PROFIT VS DISCOUNT
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    df["Sales"],
    df["Profit"],
    s=(df["Discount"] + 0.01) * 100,
    alpha=0.5
)

plt.title("Sales vs Profit with Discount")
plt.xlabel("Sales")
plt.ylabel("Profit")
plt.grid(True, alpha=0.2)

plt.tight_layout()

plt.savefig(
    os.path.join(
        multivariate_path,
        "01_Sales_Profit_Discount.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# 5.2 SALES VS PROFIT BY CUSTOMER SEGMENT
# ============================================================

plt.figure(figsize=(10, 6))

for segment in df["Segment"].dropna().unique():

    segment_data = df[df["Segment"] == segment]

    plt.scatter(
        segment_data["Sales"],
        segment_data["Profit"],
        alpha=0.4,
        label=segment
    )

plt.title("Sales vs Profit by Customer Segment")
plt.xlabel("Sales")
plt.ylabel("Profit")
plt.legend()
plt.grid(True, alpha=0.2)

plt.tight_layout()

plt.savefig(
    os.path.join(
        multivariate_path,
        "02_Sales_Profit_by_Segment.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# 5.3 SALES VS PROFIT BY CATEGORY
# ============================================================

plt.figure(figsize=(10, 6))

for category in df["Category"].dropna().unique():

    category_data = df[df["Category"] == category]

    plt.scatter(
        category_data["Sales"],
        category_data["Profit"],
        alpha=0.4,
        label=category
    )

plt.title("Sales vs Profit by Product Category")
plt.xlabel("Sales")
plt.ylabel("Profit")
plt.legend()
plt.grid(True, alpha=0.2)

plt.tight_layout()

plt.savefig(
    os.path.join(
        multivariate_path,
        "03_Sales_Profit_by_Category.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# 5.4 CATEGORY AND SEGMENT PERFORMANCE
# ============================================================

category_segment = (
    df.groupby(
        ["Category", "Segment"]
    )[["Sales", "Profit"]]
    .sum()
)

print("\n===== CATEGORY AND SEGMENT PERFORMANCE =====")
print(category_segment)


# Create a combined Sales + Profit visualization

category_segment_sales = (
    category_segment["Sales"]
    .unstack()
)

plt.figure(figsize=(12, 7))

category_segment_sales.plot(
    kind="bar",
    figsize=(12, 7)
)

plt.title("Sales by Category and Customer Segment")
plt.xlabel("Product Category")
plt.ylabel("Total Sales")
plt.xticks(rotation=0)
plt.legend(title="Customer Segment")
plt.grid(axis="y", alpha=0.2)

plt.tight_layout()

plt.savefig(
    os.path.join(
        multivariate_path,
        "04_Sales_Category_Segment.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# 5.5 PROFIT BY CATEGORY AND CUSTOMER SEGMENT
# ============================================================

category_segment_profit = (
    category_segment["Profit"]
    .unstack()
)

plt.figure(figsize=(12, 7))

category_segment_profit.plot(
    kind="bar",
    figsize=(12, 7)
)

plt.title("Profit by Category and Customer Segment")
plt.xlabel("Product Category")
plt.ylabel("Total Profit")
plt.xticks(rotation=0)
plt.legend(title="Customer Segment")
plt.grid(axis="y", alpha=0.2)

plt.tight_layout()

plt.savefig(
    os.path.join(
        multivariate_path,
        "05_Profit_Category_Segment.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# COMPLETION MESSAGE
# ============================================================

print("\n============================================")
print("MULTIVARIATE ANALYSIS COMPLETED")
print("============================================")

print("\nCharts saved individually in:")
print(multivariate_path)

# ============================================================
# BLOCK 6: CORRELATION ANALYSIS
# ============================================================

# WHAT THIS BLOCK DOES:
# This block measures the strength and direction of the
# relationships between numerical variables.
#
# VARIABLES ANALYZED:
# - Sales
# - Quantity
# - Discount
# - Profit
#
# EXPECTED RESULTS:
# - A correlation matrix showing correlation values
#   between numerical variables.
# - A correlation heatmap for easy visual interpretation.
#
# Correlation values range from -1 to +1:
# +1  -> strong positive relationship
#  0  -> little/no linear relationship
# -1  -> strong negative relationship
#
# OUTPUT:
# The heatmap is saved individually inside:
# outputs/02_EDA/04_Correlation/


# ============================================================
# CREATE CORRELATION OUTPUT FOLDER
# ============================================================

correlation_path = (
    r"D:\OneDrive\Desktop\Specialisation\Retail-Analytics"
    r"\outputs\02_EDA\04_Correlation"
)

os.makedirs(correlation_path, exist_ok=True)


print("\n============================================")
print("BLOCK 6: CORRELATION ANALYSIS")
print("============================================")


# ============================================================
# 6.1 SELECT NUMERICAL VARIABLES
# ============================================================

correlation_columns = [
    "Sales",
    "Quantity",
    "Discount",
    "Profit"
]

correlation_data = df[correlation_columns]


# ============================================================
# 6.2 CALCULATE CORRELATION MATRIX
# ============================================================

correlation_matrix = correlation_data.corr()

print("\n===== CORRELATION MATRIX =====")
print(correlation_matrix)


# ============================================================
# 6.3 CREATE CORRELATION HEATMAP
# ============================================================

plt.figure(figsize=(10, 7))

plt.imshow(
    correlation_matrix,
    interpolation="nearest",
    aspect="auto"
)

plt.colorbar(label="Correlation")

plt.xticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns
)

plt.title("Correlation Heatmap of Numerical Variables")

# Display correlation values inside the heatmap

for i in range(len(correlation_matrix.columns)):
    for j in range(len(correlation_matrix.columns)):

        plt.text(
            j,
            i,
            f"{correlation_matrix.iloc[i, j]:.2f}",
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig(
    os.path.join(
        correlation_path,
        "01_Correlation_Heatmap.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# 6.4 IDENTIFY STRONGEST CORRELATIONS
# ============================================================

correlation_pairs = (
    correlation_matrix
    .where(
        ~np.eye(
            correlation_matrix.shape[0],
            dtype=bool
        )
    )
    .stack()
    .sort_values(
        key=abs,
        ascending=False
    )
)

print("\n===== STRONGEST CORRELATION PAIRS =====")
print(correlation_pairs.head(10))


# ============================================================
# COMPLETION MESSAGE
# ============================================================

print("\n============================================")
print("CORRELATION ANALYSIS COMPLETED")
print("============================================")

print("\nCorrelation heatmap saved in:")
print(correlation_path)

# ============================================================
# BLOCK 7: SALES TREND ANALYSIS
# ============================================================

# WHAT THIS BLOCK DOES:
# This block analyzes how sales change over time.
#
# ANALYSIS PERFORMED:
# - Monthly sales trend
#
# EXPECTED RESULT:
# The line chart shows increasing, decreasing, and seasonal
# patterns in sales over the available order period.
#
# OUTPUT:
# The chart is saved individually inside:
# outputs/02_EDA/05_Sales_Trend/


# ============================================================
# CREATE OUTPUT FOLDER
# ============================================================

sales_trend_path = (
    r"D:\OneDrive\Desktop\Specialisation\Retail-Analytics"
    r"\outputs\02_EDA\05_Sales_Trend"
)

os.makedirs(sales_trend_path, exist_ok=True)


print("\n============================================")
print("BLOCK 7: SALES TREND ANALYSIS")
print("============================================")


# ============================================================
# 7.1 MONTHLY SALES
# ============================================================

monthly_sales = (
    df.groupby(
        df["Order Date"].dt.to_period("M")
    )["Sales"]
    .sum()
)

monthly_sales.index = monthly_sales.index.to_timestamp()


print("\n===== MONTHLY SALES =====")
print(monthly_sales)


# ============================================================
# 7.2 MONTHLY SALES TREND
# ============================================================

plt.figure(figsize=(14, 6))

plt.plot(
    monthly_sales.index,
    monthly_sales.values,
    marker="o"
)

plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.grid(True, alpha=0.2)

plt.tight_layout()

plt.savefig(
    os.path.join(
        sales_trend_path,
        "01_Monthly_Sales_Trend.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# 7.3 YEARLY SALES TREND
# ============================================================

yearly_sales = (
    df.groupby(
        df["Order Date"].dt.year
    )["Sales"]
    .sum()
)


print("\n===== YEARLY SALES =====")
print(yearly_sales)


plt.figure(figsize=(10, 6))

plt.plot(
    yearly_sales.index,
    yearly_sales.values,
    marker="o"
)

plt.title("Yearly Sales Trend")
plt.xlabel("Year")
plt.ylabel("Total Sales")
plt.grid(True, alpha=0.2)

plt.tight_layout()

plt.savefig(
    os.path.join(
        sales_trend_path,
        "02_Yearly_Sales_Trend.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# 7.4 MONTHLY SALES SUMMARY
# ============================================================

monthly_sales_summary = monthly_sales.reset_index()

monthly_sales_summary.columns = [
    "Month",
    "Total_Sales"
]

print("\n===== SALES TREND SUMMARY =====")
print(monthly_sales_summary.head())


# ============================================================
# COMPLETION MESSAGE
# ============================================================

print("\n============================================")
print("SALES TREND ANALYSIS COMPLETED")
print("============================================")

print("\nCharts saved individually in:")
print(sales_trend_path)
# ============================================================
# BLOCK 8: CUSTOMER BEHAVIOR ANALYSIS
# ============================================================

# WHAT THIS BLOCK DOES:
# Analyzes customer purchasing behavior.
#
# RESULTS:
# - Sales by customer segment
# - Profit by customer segment
# - Number of unique orders by segment
# - Top 10 customers by sales
# - Top 10 customers by profit
#
# All charts are saved individually in:
# outputs/02_EDA/06_Customer_Behavior/


customer_path = (
    r"D:\OneDrive\Desktop\Specialisation\Retail-Analytics"
    r"\outputs\02_EDA\06_Customer_Behavior"
)

os.makedirs(customer_path, exist_ok=True)

print("\n============================================")
print("BLOCK 8: CUSTOMER BEHAVIOR ANALYSIS")
print("============================================")


# 8.1 SALES BY CUSTOMER SEGMENT

segment_sales = (
    df.groupby("Segment")["Sales"]
    .sum()
    .sort_values(ascending=False)
)

print("\n===== SALES BY CUSTOMER SEGMENT =====")
print(segment_sales)

plt.figure(figsize=(10, 6))
segment_sales.plot(kind="bar")

plt.title("Sales by Customer Segment")
plt.xlabel("Customer Segment")
plt.ylabel("Total Sales")
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.2)

plt.tight_layout()
plt.savefig(
    os.path.join(
        customer_path,
        "01_Sales_by_Customer_Segment.png"
    ),
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()


# 8.2 PROFIT BY CUSTOMER SEGMENT

segment_profit = (
    df.groupby("Segment")["Profit"]
    .sum()
    .sort_values(ascending=False)
)

print("\n===== PROFIT BY CUSTOMER SEGMENT =====")
print(segment_profit)

plt.figure(figsize=(10, 6))
segment_profit.plot(kind="bar")

plt.title("Profit by Customer Segment")
plt.xlabel("Customer Segment")
plt.ylabel("Total Profit")
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.2)

plt.tight_layout()
plt.savefig(
    os.path.join(
        customer_path,
        "02_Profit_by_Customer_Segment.png"
    ),
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()


# 8.3 ORDERS BY CUSTOMER SEGMENT

segment_orders = (
    df.groupby("Segment")["Order ID"]
    .nunique()
    .sort_values(ascending=False)
)

print("\n===== ORDERS BY CUSTOMER SEGMENT =====")
print(segment_orders)

plt.figure(figsize=(10, 6))
segment_orders.plot(kind="bar")

plt.title("Number of Orders by Customer Segment")
plt.xlabel("Customer Segment")
plt.ylabel("Number of Unique Orders")
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.2)

plt.tight_layout()
plt.savefig(
    os.path.join(
        customer_path,
        "03_Orders_by_Customer_Segment.png"
    ),
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()


# 8.4 TOP 10 CUSTOMERS BY SALES

top_customers_sales = (
    df.groupby("Customer Name")["Sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("\n===== TOP 10 CUSTOMERS BY SALES =====")
print(top_customers_sales)

plt.figure(figsize=(12, 7))
top_customers_sales.sort_values().plot(kind="barh")

plt.title("Top 10 Customers by Sales")
plt.xlabel("Total Sales")
plt.ylabel("Customer Name")
plt.grid(axis="x", alpha=0.2)

plt.tight_layout()
plt.savefig(
    os.path.join(
        customer_path,
        "04_Top_10_Customers_by_Sales.png"
    ),
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()


# 8.5 TOP 10 CUSTOMERS BY PROFIT

top_customers_profit = (
    df.groupby("Customer Name")["Profit"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("\n===== TOP 10 CUSTOMERS BY PROFIT =====")
print(top_customers_profit)

plt.figure(figsize=(12, 7))
top_customers_profit.sort_values().plot(kind="barh")

plt.title("Top 10 Customers by Profit")
plt.xlabel("Total Profit")
plt.ylabel("Customer Name")
plt.grid(axis="x", alpha=0.2)

plt.tight_layout()
plt.savefig(
    os.path.join(
        customer_path,
        "05_Top_10_Customers_by_Profit.png"
    ),
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()


print("\nCUSTOMER BEHAVIOR ANALYSIS COMPLETED.")


# ============================================================
# BLOCK 9: PRODUCT PERFORMANCE ANALYSIS
# ============================================================

# WHAT THIS BLOCK DOES:
# Analyzes product and sub-category performance.
#
# RESULTS:
# - Sales and profit by category
# - Sales and profit by sub-category
# - Top 10 products by sales
# - Top 10 products by profit
#
# All charts are saved individually in:
# outputs/02_EDA/07_Product_Performance/


product_path = (
    r"D:\OneDrive\Desktop\Specialisation\Retail-Analytics"
    r"\outputs\02_EDA\07_Product_Performance"
)

os.makedirs(product_path, exist_ok=True)

print("\n============================================")
print("BLOCK 9: PRODUCT PERFORMANCE ANALYSIS")
print("============================================")


# 9.1 SALES BY CATEGORY

category_sales = (
    df.groupby("Category")["Sales"]
    .sum()
    .sort_values(ascending=False)
)

print("\n===== SALES BY CATEGORY =====")
print(category_sales)

plt.figure(figsize=(10, 6))
category_sales.plot(kind="bar")

plt.title("Sales by Product Category")
plt.xlabel("Category")
plt.ylabel("Total Sales")
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.2)

plt.tight_layout()
plt.savefig(
    os.path.join(
        product_path,
        "01_Sales_by_Category.png"
    ),
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()


# 9.2 PROFIT BY CATEGORY

category_profit = (
    df.groupby("Category")["Profit"]
    .sum()
    .sort_values(ascending=False)
)

print("\n===== PROFIT BY CATEGORY =====")
print(category_profit)

plt.figure(figsize=(10, 6))
category_profit.plot(kind="bar")

plt.title("Profit by Product Category")
plt.xlabel("Category")
plt.ylabel("Total Profit")
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.2)

plt.tight_layout()
plt.savefig(
    os.path.join(
        product_path,
        "02_Profit_by_Category.png"
    ),
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()


# 9.3 SALES BY SUB-CATEGORY

subcategory_sales = (
    df.groupby("Sub-Category")["Sales"]
    .sum()
    .sort_values(ascending=False)
)

print("\n===== SALES BY SUB-CATEGORY =====")
print(subcategory_sales)

plt.figure(figsize=(12, 7))
subcategory_sales.sort_values().plot(kind="barh")

plt.title("Sales by Sub-Category")
plt.xlabel("Total Sales")
plt.ylabel("Sub-Category")
plt.grid(axis="x", alpha=0.2)

plt.tight_layout()
plt.savefig(
    os.path.join(
        product_path,
        "03_Sales_by_Sub_Category.png"
    ),
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()


# 9.4 PROFIT BY SUB-CATEGORY

subcategory_profit = (
    df.groupby("Sub-Category")["Profit"]
    .sum()
    .sort_values(ascending=False)
)

print("\n===== PROFIT BY SUB-CATEGORY =====")
print(subcategory_profit)

plt.figure(figsize=(12, 7))
subcategory_profit.sort_values().plot(kind="barh")

plt.title("Profit by Sub-Category")
plt.xlabel("Total Profit")
plt.ylabel("Sub-Category")
plt.grid(axis="x", alpha=0.2)

plt.tight_layout()
plt.savefig(
    os.path.join(
        product_path,
        "04_Profit_by_Sub_Category.png"
    ),
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()


# 9.5 TOP 10 PRODUCTS BY SALES

top_products_sales = (
    df.groupby("Product Name")["Sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("\n===== TOP 10 PRODUCTS BY SALES =====")
print(top_products_sales)

plt.figure(figsize=(12, 7))
top_products_sales.sort_values().plot(kind="barh")

plt.title("Top 10 Products by Sales")
plt.xlabel("Total Sales")
plt.ylabel("Product Name")
plt.grid(axis="x", alpha=0.2)

plt.tight_layout()
plt.savefig(
    os.path.join(
        product_path,
        "05_Top_10_Products_by_Sales.png"
    ),
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()


# 9.6 TOP 10 PRODUCTS BY PROFIT

top_products_profit = (
    df.groupby("Product Name")["Profit"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("\n===== TOP 10 PRODUCTS BY PROFIT =====")
print(top_products_profit)

plt.figure(figsize=(12, 7))
top_products_profit.sort_values().plot(kind="barh")

plt.title("Top 10 Products by Profit")
plt.xlabel("Total Profit")
plt.ylabel("Product Name")
plt.grid(axis="x", alpha=0.2)

plt.tight_layout()
plt.savefig(
    os.path.join(
        product_path,
        "06_Top_10_Products_by_Profit.png"
    ),
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()


print("\nPRODUCT PERFORMANCE ANALYSIS COMPLETED.")


# ============================================================
# BLOCK 10: REGIONAL PERFORMANCE ANALYSIS
# ============================================================

# WHAT THIS BLOCK DOES:
# Examines sales and profit performance across geographical
# regions and markets.
#
# RESULTS:
# - Sales by Region
# - Profit by Region
# - Sales by Market
# - Profit by Market
#
# All charts are saved individually in:
# outputs/02_EDA/08_Regional_Performance/


regional_path = (
    r"D:\OneDrive\Desktop\Specialisation\Retail-Analytics"
    r"\outputs\02_EDA\08_Regional_Performance"
)

os.makedirs(regional_path, exist_ok=True)

print("\n============================================")
print("BLOCK 10: REGIONAL PERFORMANCE ANALYSIS")
print("============================================")


# 10.1 SALES BY REGION

region_sales = (
    df.groupby("Region")["Sales"]
    .sum()
    .sort_values(ascending=False)
)

print("\n===== SALES BY REGION =====")
print(region_sales)

plt.figure(figsize=(12, 7))
region_sales.plot(kind="bar")

plt.title("Sales by Region")
plt.xlabel("Region")
plt.ylabel("Total Sales")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", alpha=0.2)

plt.tight_layout()
plt.savefig(
    os.path.join(
        regional_path,
        "01_Sales_by_Region.png"
    ),
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()


# 10.2 PROFIT BY REGION

region_profit = (
    df.groupby("Region")["Profit"]
    .sum()
    .sort_values(ascending=False)
)

print("\n===== PROFIT BY REGION =====")
print(region_profit)

plt.figure(figsize=(12, 7))
region_profit.plot(kind="bar")

plt.title("Profit by Region")
plt.xlabel("Region")
plt.ylabel("Total Profit")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", alpha=0.2)

plt.tight_layout()
plt.savefig(
    os.path.join(
        regional_path,
        "02_Profit_by_Region.png"
    ),
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()


# 10.3 SALES BY MARKET

market_sales = (
    df.groupby("Market")["Sales"]
    .sum()
    .sort_values(ascending=False)
)

print("\n===== SALES BY MARKET =====")
print(market_sales)

plt.figure(figsize=(10, 6))
market_sales.plot(kind="bar")

plt.title("Sales by Market")
plt.xlabel("Market")
plt.ylabel("Total Sales")
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.2)

plt.tight_layout()
plt.savefig(
    os.path.join(
        regional_path,
        "03_Sales_by_Market.png"
    ),
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()


# 10.4 PROFIT BY MARKET

market_profit = (
    df.groupby("Market")["Profit"]
    .sum()
    .sort_values(ascending=False)
)

print("\n===== PROFIT BY MARKET =====")
print(market_profit)

plt.figure(figsize=(10, 6))
market_profit.plot(kind="bar")

plt.title("Profit by Market")
plt.xlabel("Market")
plt.ylabel("Total Profit")
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.2)

plt.tight_layout()
plt.savefig(
    os.path.join(
        regional_path,
        "04_Profit_by_Market.png"
    ),
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()


print("\nREGIONAL PERFORMANCE ANALYSIS COMPLETED.")


# ============================================================
# BLOCK 11: PROFIT ANALYSIS
# ============================================================

# WHAT THIS BLOCK DOES:
# Examines overall profitability and identifies profitable
# and loss-making areas.
#
# RESULTS:
# - Profit distribution
# - Profit by category
# - Profit by sub-category
# - Loss-making sub-categories
#
# All charts are saved individually in:
# outputs/02_EDA/09_Profit_Analysis/


profit_path = (
    r"D:\OneDrive\Desktop\Specialisation\Retail-Analytics"
    r"\outputs\02_EDA\09_Profit_Analysis"
)

os.makedirs(profit_path, exist_ok=True)

print("\n============================================")
print("BLOCK 11: PROFIT ANALYSIS")
print("============================================")


# 11.1 PROFIT DISTRIBUTION

plt.figure(figsize=(10, 6))

plt.hist(
    df["Profit"],
    bins=50
)

plt.title("Profit Distribution")
plt.xlabel("Profit")
plt.ylabel("Frequency")
plt.grid(axis="y", alpha=0.2)

plt.tight_layout()
plt.savefig(
    os.path.join(
        profit_path,
        "01_Profit_Distribution.png"
    ),
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()


# 11.2 PROFIT BY CATEGORY

profit_category = (
    df.groupby("Category")["Profit"]
    .sum()
    .sort_values(ascending=False)
)

print("\n===== PROFIT BY CATEGORY =====")
print(profit_category)

plt.figure(figsize=(10, 6))
profit_category.plot(kind="bar")

plt.title("Total Profit by Category")
plt.xlabel("Category")
plt.ylabel("Total Profit")
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.2)

plt.tight_layout()
plt.savefig(
    os.path.join(
        profit_path,
        "02_Profit_by_Category.png"
    ),
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()


# 11.3 PROFIT BY SUB-CATEGORY

profit_subcategory = (
    df.groupby("Sub-Category")["Profit"]
    .sum()
    .sort_values()
)

print("\n===== PROFIT BY SUB-CATEGORY =====")
print(profit_subcategory)

plt.figure(figsize=(12, 7))
profit_subcategory.plot(kind="barh")

plt.title("Profit by Sub-Category")
plt.xlabel("Total Profit")
plt.ylabel("Sub-Category")
plt.grid(axis="x", alpha=0.2)

plt.tight_layout()
plt.savefig(
    os.path.join(
        profit_path,
        "03_Profit_by_Sub_Category.png"
    ),
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()


# 11.4 LOSS-MAKING SUB-CATEGORIES

loss_subcategories = (
    df.groupby("Sub-Category")["Profit"]
    .sum()
)

loss_subcategories = (
    loss_subcategories[
        loss_subcategories < 0
    ]
    .sort_values()
)

print("\n===== LOSS-MAKING SUB-CATEGORIES =====")
print(loss_subcategories)

if not loss_subcategories.empty:

    plt.figure(figsize=(10, 6))

    loss_subcategories.plot(kind="bar")

    plt.title("Loss-Making Sub-Categories")
    plt.xlabel("Sub-Category")
    plt.ylabel("Total Loss")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            profit_path,
            "04_Loss_Making_Sub_Categories.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()
    plt.close()


print("\nPROFIT ANALYSIS COMPLETED.")


# ============================================================
# BLOCK 12: DISCOUNT ANALYSIS
# ============================================================

# WHAT THIS BLOCK DOES:
# Examines how discount levels relate to sales and profit.
#
# RESULTS:
# - Discount distribution
# - Average profit at each discount level
# - Total sales at each discount level
# - Total profit at each discount level
#
# All charts are saved individually in:
# outputs/02_EDA/10_Discount_Analysis/


discount_path = (
    r"D:\OneDrive\Desktop\Specialisation\Retail-Analytics"
    r"\outputs\02_EDA\10_Discount_Analysis"
)

os.makedirs(discount_path, exist_ok=True)

print("\n============================================")
print("BLOCK 12: DISCOUNT ANALYSIS")
print("============================================")


# 12.1 DISCOUNT DISTRIBUTION

plt.figure(figsize=(10, 6))

plt.hist(
    df["Discount"],
    bins=20
)

plt.title("Discount Distribution")
plt.xlabel("Discount")
plt.ylabel("Frequency")
plt.grid(axis="y", alpha=0.2)

plt.tight_layout()
plt.savefig(
    os.path.join(
        discount_path,
        "01_Discount_Distribution.png"
    ),
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()


# 12.2 AVERAGE PROFIT BY DISCOUNT

discount_profit = (
    df.groupby("Discount")["Profit"]
    .mean()
    .sort_index()
)

print("\n===== AVERAGE PROFIT BY DISCOUNT =====")
print(discount_profit)

plt.figure(figsize=(10, 6))

plt.plot(
    discount_profit.index,
    discount_profit.values,
    marker="o"
)

plt.title("Average Profit by Discount Level")
plt.xlabel("Discount")
plt.ylabel("Average Profit")
plt.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(
    os.path.join(
        discount_path,
        "02_Average_Profit_by_Discount.png"
    ),
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()


# 12.3 TOTAL SALES BY DISCOUNT

discount_sales = (
    df.groupby("Discount")["Sales"]
    .sum()
    .sort_index()
)

print("\n===== TOTAL SALES BY DISCOUNT =====")
print(discount_sales)

plt.figure(figsize=(10, 6))

discount_sales.plot(kind="bar")

plt.title("Total Sales by Discount Level")
plt.xlabel("Discount")
plt.ylabel("Total Sales")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", alpha=0.2)

plt.tight_layout()
plt.savefig(
    os.path.join(
        discount_path,
        "03_Total_Sales_by_Discount.png"
    ),
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()


# 12.4 TOTAL PROFIT BY DISCOUNT

discount_total_profit = (
    df.groupby("Discount")["Profit"]
    .sum()
    .sort_index()
)

print("\n===== TOTAL PROFIT BY DISCOUNT =====")
print(discount_total_profit)

plt.figure(figsize=(10, 6))

discount_total_profit.plot(kind="bar")

plt.title("Total Profit by Discount Level")
plt.xlabel("Discount")
plt.ylabel("Total Profit")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", alpha=0.2)

plt.tight_layout()
plt.savefig(
    os.path.join(
        discount_path,
        "04_Total_Profit_by_Discount.png"
    ),
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()


print("\nDISCOUNT ANALYSIS COMPLETED.")


# ============================================================
# BLOCK 13: CUSTOMER SEGMENTATION ANALYSIS
# ============================================================

# WHAT THIS BLOCK DOES:
# Compares customer segments using Sales, Profit,
# Order Count and Average Order Value.
#
# RESULTS:
# - Sales comparison between segments
# - Profit comparison between segments
# - Number of orders by segment
# - Average sales per order by segment
#
# All charts are saved individually in:
# outputs/02_EDA/11_Customer_Segmentation/


segmentation_path = (
    r"D:\OneDrive\Desktop\Specialisation\Retail-Analytics"
    r"\outputs\02_EDA\11_Customer_Segmentation"
)

os.makedirs(segmentation_path, exist_ok=True)

print("\n============================================")
print("BLOCK 13: CUSTOMER SEGMENTATION ANALYSIS")
print("============================================")


# ============================================================
# 13.1 SEGMENT SALES
# ============================================================

segmentation_sales = (
    df.groupby("Segment")["Sales"]
    .sum()
    .sort_values(ascending=False)
)

print("\n===== SEGMENT SALES =====")
print(segmentation_sales)

plt.figure(figsize=(10, 6))

segmentation_sales.plot(kind="bar")

plt.title("Sales by Customer Segment")
plt.xlabel("Customer Segment")
plt.ylabel("Total Sales")
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.2)

plt.tight_layout()
plt.savefig(
    os.path.join(
        segmentation_path,
        "01_Sales_by_Segment.png"
    ),
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()


# ============================================================
# 13.2 SEGMENT PROFIT
# ============================================================

segmentation_profit = (
    df.groupby("Segment")["Profit"]
    .sum()
    .sort_values(ascending=False)
)

print("\n===== SEGMENT PROFIT =====")
print(segmentation_profit)

plt.figure(figsize=(10, 6))

segmentation_profit.plot(kind="bar")

plt.title("Profit by Customer Segment")
plt.xlabel("Customer Segment")
plt.ylabel("Total Profit")
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.2)

plt.tight_layout()
plt.savefig(
    os.path.join(
        segmentation_path,
        "02_Profit_by_Segment.png"
    ),
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()


# ============================================================
# 13.3 NUMBER OF ORDERS BY SEGMENT
# ============================================================

segmentation_orders = (
    df.groupby("Segment")["Order ID"]
    .nunique()
    .sort_values(ascending=False)
)

print("\n===== ORDERS BY SEGMENT =====")
print(segmentation_orders)

plt.figure(figsize=(10, 6))

segmentation_orders.plot(kind="bar")

plt.title("Number of Orders by Customer Segment")
plt.xlabel("Customer Segment")
plt.ylabel("Number of Unique Orders")
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.2)

plt.tight_layout()
plt.savefig(
    os.path.join(
        segmentation_path,
        "03_Orders_by_Segment.png"
    ),
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()


# ============================================================
# 13.4 AVERAGE ORDER VALUE BY SEGMENT
# ============================================================

segment_aov = (
    df.groupby("Segment")
    .apply(
        lambda x: x["Sales"].sum()
        / x["Order ID"].nunique(),
        include_groups=False
    )
    .sort_values(ascending=False)
)

print("\n===== AVERAGE ORDER VALUE BY SEGMENT =====")
print(segment_aov)

plt.figure(figsize=(10, 6))

segment_aov.plot(kind="bar")

plt.title("Average Order Value by Customer Segment")
plt.xlabel("Customer Segment")
plt.ylabel("Average Order Value")
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.2)

plt.tight_layout()
plt.savefig(
    os.path.join(
        segmentation_path,
        "04_Average_Order_Value_by_Segment.png"
    ),
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()


# ============================================================
# FINAL CUSTOMER SEGMENT SUMMARY
# ============================================================

segment_summary = pd.DataFrame({
    "Total_Sales": segmentation_sales,
    "Total_Profit": segmentation_profit,
    "Unique_Orders": segmentation_orders,
    "Average_Order_Value": segment_aov
})

print("\n===== CUSTOMER SEGMENT SUMMARY =====")
print(segment_summary)


print("\n============================================")
print("CUSTOMER SEGMENTATION ANALYSIS COMPLETED")
print("============================================")

print("\nAll EDA blocks 6 to 11 completed.")
# ============================================================
# BLOCK 14: EXPORT EDA RESULTS TO EXCEL
# ============================================================

# This block saves the important EDA summary tables into
# ONE Excel workbook.
#
# Each analysis is stored in a separate Excel sheet.
#
# OUTPUT:
# outputs/02_EDA/EDA_Results.xlsx


eda_excel_path = (
    r"D:\OneDrive\Desktop\Specialisation\Retail-Analytics"
    r"\outputs\02_EDA\EDA_Results.xlsx"
)

print("\n============================================")
print("BLOCK 14: EXPORTING EDA RESULTS TO EXCEL")
print("============================================")


# Make sure the output directory exists
os.makedirs(
    os.path.dirname(eda_excel_path),
    exist_ok=True
)


# ============================================================
# CREATE EXCEL WORKBOOK
# ============================================================

with pd.ExcelWriter(
    eda_excel_path,
    engine="openpyxl"
) as writer:

    # --------------------------------------------------------
    # 1. CORRELATION MATRIX
    # --------------------------------------------------------

    correlation_matrix.to_excel(
        writer,
        sheet_name="Correlation"
    )


    # --------------------------------------------------------
    # 2. MONTHLY SALES
    # --------------------------------------------------------

    monthly_sales_summary.to_excel(
        writer,
        sheet_name="Monthly_Sales",
        index=False
    )


    # --------------------------------------------------------
    # 3. CUSTOMER BEHAVIOR
    # --------------------------------------------------------

    segment_sales.rename(
        "Total_Sales"
    ).to_frame().to_excel(
        writer,
        sheet_name="Segment_Sales"
    )


    segment_profit.rename(
        "Total_Profit"
    ).to_frame().to_excel(
        writer,
        sheet_name="Segment_Profit"
    )


    segment_orders.rename(
        "Unique_Orders"
    ).to_frame().to_excel(
        writer,
        sheet_name="Segment_Orders"
    )


    top_customers_sales.rename(
        "Total_Sales"
    ).to_frame().to_excel(
        writer,
        sheet_name="Top_Customers_Sales"
    )


    top_customers_profit.rename(
        "Total_Profit"
    ).to_frame().to_excel(
        writer,
        sheet_name="Top_Customers_Profit"
    )


    # --------------------------------------------------------
    # 4. PRODUCT PERFORMANCE
    # --------------------------------------------------------

    category_sales.rename(
        "Total_Sales"
    ).to_frame().to_excel(
        writer,
        sheet_name="Category_Sales"
    )


    category_profit.rename(
        "Total_Profit"
    ).to_frame().to_excel(
        writer,
        sheet_name="Category_Profit"
    )


    subcategory_sales.rename(
        "Total_Sales"
    ).to_frame().to_excel(
        writer,
        sheet_name="Subcategory_Sales"
    )


    subcategory_profit.rename(
        "Total_Profit"
    ).to_frame().to_excel(
        writer,
        sheet_name="Subcategory_Profit"
    )


    top_products_sales.rename(
        "Total_Sales"
    ).to_frame().to_excel(
        writer,
        sheet_name="Top_Products_Sales"
    )


    top_products_profit.rename(
        "Total_Profit"
    ).to_frame().to_excel(
        writer,
        sheet_name="Top_Products_Profit"
    )


    # --------------------------------------------------------
    # 5. REGIONAL PERFORMANCE
    # --------------------------------------------------------

    region_sales.rename(
        "Total_Sales"
    ).to_frame().to_excel(
        writer,
        sheet_name="Region_Sales"
    )


    region_profit.rename(
        "Total_Profit"
    ).to_frame().to_excel(
        writer,
        sheet_name="Region_Profit"
    )


    market_sales.rename(
        "Total_Sales"
    ).to_frame().to_excel(
        writer,
        sheet_name="Market_Sales"
    )


    market_profit.rename(
        "Total_Profit"
    ).to_frame().to_excel(
        writer,
        sheet_name="Market_Profit"
    )


    # --------------------------------------------------------
    # 6. PROFIT ANALYSIS
    # --------------------------------------------------------

    profit_category.rename(
        "Total_Profit"
    ).to_frame().to_excel(
        writer,
        sheet_name="Profit_Category"
    )


    profit_subcategory.rename(
        "Total_Profit"
    ).to_frame().to_excel(
        writer,
        sheet_name="Profit_Subcategory"
    )


    loss_subcategories.rename(
        "Total_Loss"
    ).to_frame().to_excel(
        writer,
        sheet_name="Loss_Subcategories"
    )


    # --------------------------------------------------------
    # 7. DISCOUNT ANALYSIS
    # --------------------------------------------------------

    discount_profit.rename(
        "Average_Profit"
    ).to_frame().to_excel(
        writer,
        sheet_name="Avg_Profit_Discount"
    )


    discount_sales.rename(
        "Total_Sales"
    ).to_frame().to_excel(
        writer,
        sheet_name="Sales_Discount"
    )


    discount_total_profit.rename(
        "Total_Profit"
    ).to_frame().to_excel(
        writer,
        sheet_name="Profit_Discount"
    )


    # --------------------------------------------------------
    # 8. CUSTOMER SEGMENTATION
    # --------------------------------------------------------

    segment_summary.to_excel(
        writer,
        sheet_name="Customer_Segmentation"
    )


print("\n============================================")
print("EDA EXCEL EXPORT COMPLETED")
print("============================================")

print("\nExcel file saved at:")
print(eda_excel_path)