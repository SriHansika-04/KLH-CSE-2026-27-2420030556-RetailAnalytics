# ============================================
# GLOBAL SUPERSTORE RETAIL ANALYTICS
# DATA PREPROCESSING
# ============================================

import pandas as pd
import numpy as np
import os

# ============================================
# BLOCK 1: LOAD GLOBAL SUPERSTORE DATASET
# ============================================

# WHAT THIS BLOCK DOES:
# Loads the raw Global Superstore CSV into a Pandas DataFrame.
# Latin-1 encoding is used because the CSV contains special
# characters that may not be read correctly using UTF-8.

df = pd.read_csv(
    "data/Global_Superstore.csv",
    encoding="latin1"
)

print("Dataset loaded successfully!")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


# ============================================
# BLOCK 2: DISPLAY COLUMN NAMES
# ============================================

# WHAT THIS BLOCK DOES:
# Displays the exact column names present in the dataset.
# We use these names to avoid errors in later processing.

print("\nColumn Names:")
print(df.columns.tolist())

# ============================================
# BLOCK 3: DATASET OVERVIEW
# ============================================

# WHAT THIS BLOCK DOES:
# Displays the first and last few records and the exact
# structure of the dataset.
#
# EXPECTED RESULT:
# We will see sample records, confirm the 24 columns,
# and understand the type of information available
# before starting the preprocessing steps.


print("\n===== FIRST 5 RECORDS =====")
print(df.head())


print("\n===== LAST 5 RECORDS =====")
print(df.tail())


print("\n===== DATASET INFORMATION =====")
df.info()


print("\n===== COLUMN NAMES =====")
print(df.columns.tolist())


# ============================================
# BLOCK 4: DATA QUALITY ASSESSMENT
# ============================================

# WHAT THIS BLOCK DOES:
# Checks the overall quality and structure of the dataset.
# We examine data types, non-null values, unique values,
# and basic information for every column.
#
# EXPECTED RESULT:
# This will help us identify:
# - Numerical and categorical columns
# - Columns containing missing values
# - Columns with very few or many unique values
# - Columns that may require data type conversion
#
# NOTE:
# No data is removed or modified in this block.
# It is only an assessment of the raw dataset.


print("\n===== DATA TYPES =====")
print(df.dtypes)


print("\n===== NON-NULL VALUES =====")
print(df.notnull().sum())


print("\n===== UNIQUE VALUES =====")
print(df.nunique())


print("\n===== DATASET SHAPE =====")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


print("\n===== DATA QUALITY SUMMARY =====")

quality_summary = pd.DataFrame({
    "Data_Type": df.dtypes.astype(str),
    "Non_Null_Count": df.notnull().sum(),
    "Missing_Count": df.isnull().sum(),
    "Unique_Count": df.nunique()
})

print(quality_summary)

# ============================================
# BLOCK 5: DESCRIPTIVE STATISTICS
# ============================================

# WHAT THIS BLOCK DOES:
# Calculates statistical measures for the numerical columns
# in the dataset.
#
# The main measures include:
# - Count
# - Mean
# - Standard deviation
# - Minimum
# - 25th percentile
# - Median
# - 75th percentile
# - Maximum
#
# EXPECTED RESULT:
# This helps us understand the central tendency, spread,
# and range of numerical variables such as Sales, Profit,
# Quantity and Discount.
#
# NOTE:
# No data is removed or modified in this block.


print("\n===== DESCRIPTIVE STATISTICS =====")

descriptive_stats = df.describe()

print(descriptive_stats)

# ============================================
# BLOCK 6: MISSING VALUE ANALYSIS
# ============================================

# WHAT THIS BLOCK DOES:
# Checks every column for missing values and calculates
# both the number and percentage of missing records.
#
# EXPECTED RESULT:
# We will identify which columns contain missing values,
# how many values are missing, and their percentage of
# the complete dataset.
#
# NOTE:
# No values are changed or removed in this block.
# We only identify the missing data first.


missing_count = df.isnull().sum()

missing_percentage = (
    df.isnull().sum() / len(df) * 100
)

missing_summary = pd.DataFrame({
    "Missing_Count": missing_count,
    "Missing_Percentage": missing_percentage
})

# Display only columns that contain missing values
missing_summary = missing_summary[
    missing_summary["Missing_Count"] > 0
].sort_values(
    by="Missing_Count",
    ascending=False
)

print("\n===== MISSING VALUE ANALYSIS =====")

if missing_summary.empty:
    print("No missing values found in the dataset.")
else:
    print(missing_summary)

    # ============================================
# BLOCK 7: DUPLICATE RECORD DETECTION
# ============================================

# WHAT THIS BLOCK DOES:
# Checks whether the dataset contains completely duplicated
# rows and counts how many duplicate records are present.
#
# EXPECTED RESULT:
# The output shows the number of duplicate rows found.
# Duplicate rows will be removed in the next step so that
# the same transaction is not counted more than once.
#
# NOTE:
# This block only identifies duplicates. It does not remove
# anything yet.


duplicate_count = df.duplicated().sum()

print("\n===== DUPLICATE RECORD DETECTION =====")
print("Duplicate records found:", duplicate_count)

# ============================================
# BLOCK 8: DUPLICATE RECORD REMOVAL
# ============================================

# WHAT THIS BLOCK DOES:
# Removes completely duplicated rows identified in Block 7.
#
# EXPECTED RESULT:
# The number of rows after removal will be displayed.
# If there were no duplicates, the number of rows will remain
# unchanged.


rows_before = df.shape[0]

df = df.drop_duplicates()

rows_after = df.shape[0]

print("\n===== DUPLICATE RECORD REMOVAL =====")
print("Rows before removal:", rows_before)
print("Rows removed:", rows_before - rows_after)
print("Rows after removal:", rows_after)

# ============================================
# BLOCK 9: DATA CLEANING
# ============================================

# WHAT THIS BLOCK DOES:
# Cleans text-based columns by removing unnecessary spaces
# from the beginning and end of values.
#
# EXPECTED RESULT:
# Text values become consistent, reducing problems caused
# by accidental spaces or inconsistent formatting.
#
# NOTE:
# We are NOT removing the Postal Code column or any useful
# dataset column in this block.


# Identify all text/categorical columns
text_columns = df.select_dtypes(include="object").columns

for column in text_columns:
    df[column] = df[column].astype(str).str.strip()


print("\n===== DATA CLEANING =====")
print("Text columns cleaned successfully.")

print("\nText columns processed:")
print(text_columns.tolist())

# ============================================
# BLOCK 10: DATA TYPE CONVERSION
# ============================================

# WHAT THIS BLOCK DOES:
# Converts date columns from text/object format into
# Pandas datetime format so that we can perform
# time-based analysis later.
#
# EXPECTED RESULT:
# Order Date and Ship Date should become datetime64[ns].
# Invalid date values, if any, will be converted to NaT.
#
# NOTE:
# We are using the actual column names after checking
# the dataset structure.


# Convert date columns
df["Order Date"] = pd.to_datetime(
    df["Order Date"],
    errors="coerce"
)

df["Ship Date"] = pd.to_datetime(
    df["Ship Date"],
    errors="coerce"
)

# ============================================
# BLOCK 10: DATA TYPE CONVERSION
# ============================================

# WHAT THIS BLOCK DOES:
# Converts Order Date and Ship Date from text format
# into Pandas datetime format.
#
# EXPECTED RESULT:
# Both date columns should become datetime64[ns].
# This allows monthly/quarterly sales analysis and
# delivery-time calculation later.
#
# Invalid date values, if any, are converted to NaT.


df["Order Date"] = pd.to_datetime(
    df["Order Date"],
    errors="coerce"
)

df["Ship Date"] = pd.to_datetime(
    df["Ship Date"],
    errors="coerce"
)


print("\n===== DATA TYPE CONVERSION =====")

print("\nDate column data types:")
print(df[["Order Date", "Ship Date"]].dtypes)


print("\nInvalid/missing dates after conversion:")

print(
    "Order Date:",
    df["Order Date"].isna().sum()
)

print(
    "Ship Date:",
    df["Ship Date"].isna().sum()
)

# ============================================
# BLOCK 11: MISSING VALUE TREATMENT
# ============================================

# WHAT THIS BLOCK DOES:
# Handles missing values based on the type of column.
#
# Numerical columns:
# Missing values are replaced with the median.
#
# Categorical/text columns:
# Missing values are replaced with the mode (most frequent value).
#
# Date columns:
# Missing dates are NOT filled using median/mode.
# They will be handled separately if any are present.
#
# EXPECTED RESULT:
# The number of missing values should decrease after
# this treatment while useful records are retained.
#
# NOTE:
# No columns are deleted, including Postal Code.


# --------------------------------------------
# 1. NUMERICAL COLUMNS
# --------------------------------------------

numeric_columns = df.select_dtypes(
    include=np.number
).columns

for column in numeric_columns:
    if df[column].isnull().sum() > 0:
        df[column] = df[column].fillna(
            df[column].median()
        )


# --------------------------------------------
# 2. CATEGORICAL / TEXT COLUMNS
# --------------------------------------------

categorical_columns = df.select_dtypes(
    include="object"
).columns

for column in categorical_columns:
    if df[column].isnull().sum() > 0:
        df[column] = df[column].fillna(
            df[column].mode()[0]
        )


# --------------------------------------------
# 3. DISPLAY REMAINING MISSING VALUES
# --------------------------------------------

remaining_missing = df.isnull().sum()

remaining_missing = remaining_missing[
    remaining_missing > 0
].sort_values(ascending=False)


print("\n===== MISSING VALUE TREATMENT =====")

if remaining_missing.empty:
    print("No missing values remain.")
else:
    print("Remaining missing values:")
    print(remaining_missing)

    # ============================================
# BLOCK 12: OUTLIER DETECTION
# ============================================

# WHAT THIS BLOCK DOES:
# Detects potential outliers in important numerical variables
# using the Interquartile Range (IQR) method.
#
# EXPECTED RESULT:
# The output shows Q1, Q3, IQR, lower limit, upper limit,
# and the number of potential outliers for each variable.
#
# NOTE:
# This block ONLY detects outliers.
# No records or values are changed here.


# Numerical variables selected for outlier analysis
outlier_columns = [
    "Sales",
    "Quantity",
    "Discount",
    "Profit"
]


print("\n===== OUTLIER DETECTION USING IQR =====")

outlier_summary = []

for column in outlier_columns:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outlier_count = (
        (df[column] < lower_limit) |
        (df[column] > upper_limit)
    ).sum()

    outlier_summary.append({
        "Column": column,
        "Q1": Q1,
        "Q3": Q3,
        "IQR": IQR,
        "Lower_Limit": lower_limit,
        "Upper_Limit": upper_limit,
        "Outlier_Count": outlier_count
    })


outlier_summary = pd.DataFrame(outlier_summary)

print(outlier_summary.to_string(index=False))

# ============================================
# BLOCK 13: OUTLIER TREATMENT - IQR CAPPING
# ============================================

# WHAT THIS BLOCK DOES:
# Caps extreme values in Sales, Quantity, Discount and Profit
# using the IQR lower and upper limits.
#
# Instead of deleting complete transactions, values outside
# the IQR limits are replaced by the corresponding boundary.
#
# EXPECTED RESULT:
# Extreme numerical values are reduced while all transaction
# records are retained.
#
# NOTE:
# The number of rows should NOT change after this block.


outlier_columns = [
    "Sales",
    "Quantity",
    "Discount",
    "Profit"
]

rows_before = len(df)

for column in outlier_columns:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    df[column] = df[column].clip(
        lower=lower_limit,
        upper=upper_limit
    )


rows_after = len(df)

print("\n===== OUTLIER TREATMENT =====")

print("Rows before treatment:", rows_before)
print("Rows after treatment:", rows_after)

if rows_before == rows_after:
    print("All transaction records were retained.")
else:
    print("Warning: Number of rows changed.")


    # ============================================
# BLOCK 14: DATA TRANSFORMATION & PREPARATION
# ============================================

# WHAT THIS BLOCK DOES:
# Performs basic transformations needed after cleaning:
# 1. Ensures numerical columns have numeric data types.
# 2. Ensures date columns are in datetime format.
# 3. Creates a clean copy of the dataset for the next stage.
#
# EXPECTED RESULT:
# The cleaned dataset will have consistent numerical and
# datetime data types and will be ready for EDA.
#
# NOTE:
# Categorical variables are NOT encoded here because their
# original categories are required for EDA and feature
# engineering. Encoding will be performed later when needed
# for machine learning.


# --------------------------------------------
# 1. ENSURE NUMERICAL COLUMNS ARE NUMERIC
# --------------------------------------------

numeric_columns = [
    "Sales",
    "Quantity",
    "Discount",
    "Profit"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# --------------------------------------------
# 2. ENSURE DATE COLUMNS ARE DATETIME
# --------------------------------------------

df["Order Date"] = pd.to_datetime(
    df["Order Date"],
    errors="coerce"
)

df["Ship Date"] = pd.to_datetime(
    df["Ship Date"],
    errors="coerce"
)


# --------------------------------------------
# 3. RESET DATAFRAME INDEX
# --------------------------------------------

df = df.reset_index(drop=True)


# --------------------------------------------
# 4. FINAL PREPROCESSING CHECK
# --------------------------------------------

print("\n===== DATA TRANSFORMATION =====")

print("\nNumerical column data types:")
print(df[numeric_columns].dtypes)

print("\nDate column data types:")
print(df[["Order Date", "Ship Date"]].dtypes)

print("\nFinal dataset shape:")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nRemaining missing values:")
print(df.isnull().sum().sum())

# --------------------------------------------
# OUTPUT FILE PATH
# --------------------------------------------

output_path = (
    r"D:\OneDrive\Desktop\Specialisation\Retail-Analytics"
    r"\outputs\01_Preprocessing"
    r"\Preprocessed_Global_Superstore.xlsx"
)

# Create the output folder if it doesn't exist
os.makedirs(
    os.path.dirname(output_path),
    exist_ok=True
)

# Save Excel file
df.to_excel(
    output_path,
    index=False
)

print("\n===== PREPROCESSING COMPLETED =====")
print("Preprocessed dataset saved successfully!")
print("Output path:")
print(output_path)