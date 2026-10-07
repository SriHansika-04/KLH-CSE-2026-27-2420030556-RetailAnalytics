# ============================================================
# RETAIL ANALYTICS PROJECT
# FEATURE SELECTION
# ============================================================
#
# Input:
#   outputs/03_Feature_Engineering/
#   Feature_Engineered_Superstore.xlsx
#
# Output:
#   outputs/04_Feature_Selection/
#
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
    "03_Feature_Engineering",
    "Feature_Engineered_Superstore.xlsx"
)


# ============================================================
# 4. OUTPUT PATH
# ============================================================

OUTPUT_BASE = os.path.join(
    BASE_PATH,
    "outputs",
    "04_Feature_Selection"
)


# ============================================================
# 5. OUTPUT FOLDERS
# ============================================================

CORRELATION_PATH = os.path.join(
    OUTPUT_BASE,
    "01_Correlation"
)

NUMERICAL_PATH = os.path.join(
    OUTPUT_BASE,
    "02_Numerical_Features"
)

CATEGORICAL_PATH = os.path.join(
    OUTPUT_BASE,
    "03_Categorical_Features"
)

FINAL_PATH = os.path.join(
    OUTPUT_BASE,
    "04_Final_Features"
)


# ============================================================
# 6. CREATE FOLDERS
# ============================================================

os.makedirs(
    CORRELATION_PATH,
    exist_ok=True
)

os.makedirs(
    NUMERICAL_PATH,
    exist_ok=True
)

os.makedirs(
    CATEGORICAL_PATH,
    exist_ok=True
)

os.makedirs(
    FINAL_PATH,
    exist_ok=True
)


# ============================================================
# 7. START
# ============================================================

print()
print("============================================================")
print("RETAIL ANALYTICS - FEATURE SELECTION")
print("============================================================")


# ============================================================
# 8. CHECK INPUT FILE
# ============================================================

print()
print("===== CHECKING INPUT FILE =====")

print(
    "Input path:"
)

print(
    INPUT_PATH
)


if not os.path.exists(INPUT_PATH):

    print()
    print("ERROR: Input file was not found.")
    print()
    print("Expected:")
    print(INPUT_PATH)
    print()

    raise FileNotFoundError(
        "Feature-engineered dataset was not found."
    )


print()
print("Input file found successfully.")


# ============================================================
# 9. LOAD DATA
# ============================================================

print()
print("===== LOADING FEATURE-ENGINEERED DATA =====")


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
# 10. DISPLAY COLUMNS
# ============================================================

print()
print("===== DATASET COLUMNS =====")

for column in df.columns:

    print(
        "-",
        column
    )


# ============================================================
# 11. BASIC VALIDATION
# ============================================================

print()
print("===== BASIC VALIDATION =====")

print(
    "Missing values:",
    df.isnull().sum().sum()
)

print(
    "Duplicate rows:",
    df.duplicated().sum()
)


# ============================================================
# 12. IDENTIFY NUMERICAL FEATURES
# ============================================================

print()
print("============================================================")
print("1. NUMERICAL FEATURE IDENTIFICATION")
print("============================================================")


numerical_columns = (

    df
    .select_dtypes(
        include=["number"]
    )
    .columns
    .tolist()

)


print()
print(
    "Number of numerical features:",
    len(numerical_columns)
)


print()
print("Numerical features:")


for column in numerical_columns:

    print(
        "-",
        column
    )


# ============================================================
# 13. SAVE NUMERICAL FEATURE LIST
# ============================================================

numerical_summary = pd.DataFrame({

    "Feature": numerical_columns,

    "Data Type": [

        str(df[column].dtype)

        for column in numerical_columns

    ],

    "Missing Values": [

        df[column].isnull().sum()

        for column in numerical_columns

    ],

    "Unique Values": [

        df[column].nunique()

        for column in numerical_columns

    ],

    "Mean": [

        df[column].mean()

        for column in numerical_columns

    ],

    "Median": [

        df[column].median()

        for column in numerical_columns

    ],

    "Minimum": [

        df[column].min()

        for column in numerical_columns

    ],

    "Maximum": [

        df[column].max()

        for column in numerical_columns

    ],

})


numerical_output_path = os.path.join(

    NUMERICAL_PATH,

    "Numerical_Feature_Summary.xlsx"

)


numerical_summary.to_excel(

    numerical_output_path,

    index=False

)


print()
print(
    "Numerical feature summary saved:"
)

print(
    numerical_output_path
)


# ============================================================
# 14. IDENTIFY CATEGORICAL FEATURES
# ============================================================

print()
print("============================================================")
print("2. CATEGORICAL FEATURE IDENTIFICATION")
print("============================================================")


categorical_columns = (

    df
    .select_dtypes(
        include=["object", "string", "category"]
    )
    .columns
    .tolist()

)


print()
print(
    "Number of categorical features:",
    len(categorical_columns)
)


print()
print("Categorical features:")


for column in categorical_columns:

    print(
        "-",
        column
    )


# ============================================================
# 15. SAVE CATEGORICAL FEATURE SUMMARY
# ============================================================

categorical_summary = pd.DataFrame({

    "Feature": categorical_columns,

    "Data Type": [

        str(df[column].dtype)

        for column in categorical_columns

    ],

    "Missing Values": [

        df[column].isnull().sum()

        for column in categorical_columns

    ],

    "Unique Values": [

        df[column].nunique()

        for column in categorical_columns

    ]

})


categorical_output_path = os.path.join(

    CATEGORICAL_PATH,

    "Categorical_Feature_Summary.xlsx"

)


categorical_summary.to_excel(

    categorical_output_path,

    index=False

)


print()
print(
    "Categorical feature summary saved:"
)

print(
    categorical_output_path
)


# ============================================================
# 16. CORRELATION ANALYSIS
# ============================================================

print()
print("============================================================")
print("3. CORRELATION ANALYSIS")
print("============================================================")


# Select numerical data

numerical_data = df[
    numerical_columns
].copy()


# Calculate correlation matrix

correlation_matrix = (

    numerical_data
    .corr()

)


print()
print("Correlation matrix calculated successfully.")


# ============================================================
# 17. SAVE CORRELATION MATRIX TO EXCEL
# ============================================================

correlation_excel_path = os.path.join(

    CORRELATION_PATH,

    "Correlation_Matrix.xlsx"

)


correlation_matrix.to_excel(

    correlation_excel_path

)


print()
print(
    "Correlation matrix saved:"
)

print(
    correlation_excel_path
)


# ============================================================
# 18. CREATE CORRELATION HEATMAP
# ============================================================

print()
print("===== CREATING CORRELATION HEATMAP =====")


plt.figure(
    figsize=(18, 14)
)


plt.imshow(
    correlation_matrix,
    interpolation="nearest",
    aspect="auto"
)


plt.colorbar(
    label="Correlation"
)


plt.xticks(

    range(
        len(correlation_matrix.columns)
    ),

    correlation_matrix.columns,

    rotation=90

)


plt.yticks(

    range(
        len(correlation_matrix.columns)
    ),

    correlation_matrix.columns

)


plt.title(
    "Correlation Matrix of Numerical Features"
)


plt.tight_layout()


heatmap_path = os.path.join(

    CORRELATION_PATH,

    "Correlation_Heatmap.png"

)


plt.savefig(

    heatmap_path,

    dpi=300,

    bbox_inches="tight"

)


plt.close()


print()
print(
    "Correlation heatmap saved:"
)

print(
    heatmap_path
)


# ============================================================
# 19. FIND HIGHLY CORRELATED FEATURES
# ============================================================

print()
print("===== HIGH CORRELATION ANALYSIS =====")


correlation_pairs = []


for i in range(
    len(correlation_matrix.columns)
):

    for j in range(
        i + 1,
        len(correlation_matrix.columns)
    ):

        feature_1 = (
            correlation_matrix.columns[i]
        )

        feature_2 = (
            correlation_matrix.columns[j]
        )

        correlation_value = (
            correlation_matrix.iloc[i, j]
        )


        if not pd.isna(
            correlation_value
        ):

            correlation_pairs.append({

                "Feature 1":
                    feature_1,

                "Feature 2":
                    feature_2,

                "Correlation":
                    correlation_value,

                "Absolute Correlation":
                    abs(correlation_value)

            })


correlation_pairs_df = pd.DataFrame(
    correlation_pairs
)


if not correlation_pairs_df.empty:

    correlation_pairs_df = (

        correlation_pairs_df
        .sort_values(
            "Absolute Correlation",
            ascending=False
        )

    )


# ============================================================
# 20. SAVE CORRELATION PAIRS
# ============================================================

correlation_pairs_path = os.path.join(

    CORRELATION_PATH,

    "Correlation_Pairs.xlsx"

)


correlation_pairs_df.to_excel(

    correlation_pairs_path,

    index=False

)


print()
print(
    "Correlation pairs saved:"
)

print(
    correlation_pairs_path
)


# ============================================================
# 21. HIGH CORRELATION PAIRS
# ============================================================

high_correlation_threshold = 0.80


high_correlation_pairs = (

    correlation_pairs_df[

        correlation_pairs_df[
            "Absolute Correlation"
        ]
        >= high_correlation_threshold

    ]

)


print()
print(
    "High correlation threshold:",
    high_correlation_threshold
)


print()
print(
    "Number of highly correlated pairs:",
    len(high_correlation_pairs)
)


if len(high_correlation_pairs) > 0:

    print()
    print("Highly correlated features:")

    print(
        high_correlation_pairs[
            [
                "Feature 1",
                "Feature 2",
                "Correlation"
            ]
        ].to_string(
            index=False
        )
    )

else:

    print()
    print(
        "No feature pairs exceeded the threshold."
    )


# ============================================================
# 22. SAVE HIGH CORRELATION FEATURES
# ============================================================

high_correlation_path = os.path.join(

    CORRELATION_PATH,

    "High_Correlation_Features.xlsx"

)


high_correlation_pairs.to_excel(

    high_correlation_path,

    index=False

)


print()
print(
    "High correlation analysis saved:"
)

print(
    high_correlation_path
)


# ============================================================
# 23. FEATURE IMPORTANCE / VARIABILITY SCREENING
# ============================================================

print()
print("============================================================")
print("4. NUMERICAL FEATURE SCREENING")
print("============================================================")


numerical_screening = pd.DataFrame({

    "Feature": numerical_columns,

    "Mean": [

        df[column].mean()

        for column in numerical_columns

    ],

    "Std Dev": [

        df[column].std()

        for column in numerical_columns

    ],

    "Variance": [

        df[column].var()

        for column in numerical_columns

    ],

    "Minimum": [

        df[column].min()

        for column in numerical_columns

    ],

    "Maximum": [

        df[column].max()

        for column in numerical_columns

    ],

    "Unique Values": [

        df[column].nunique()

        for column in numerical_columns

    ]

})


# ============================================================
# 24. REMOVE CONSTANT FEATURES
# ============================================================

constant_features = [

    column

    for column in numerical_columns

    if df[column].nunique() <= 1

]


print()
print(
    "Constant numerical features:",
    len(constant_features)
)


if len(constant_features) > 0:

    for column in constant_features:

        print(
            "-",
            column
        )

else:

    print(
        "No constant numerical features found."
    )


# ============================================================
# 25. SAVE NUMERICAL SCREENING
# ============================================================

numerical_screening_path = os.path.join(

    NUMERICAL_PATH,

    "Numerical_Feature_Screening.xlsx"

)


numerical_screening.to_excel(

    numerical_screening_path,

    index=False

)


print()
print(
    "Numerical screening saved:"
)

print(
    numerical_screening_path
)


# ============================================================
# 26. CATEGORICAL SCREENING
# ============================================================

print()
print("============================================================")
print("5. CATEGORICAL FEATURE SCREENING")
print("============================================================")


categorical_screening = pd.DataFrame({

    "Feature": categorical_columns,

    "Unique Values": [

        df[column].nunique()

        for column in categorical_columns

    ],

    "Missing Values": [

        df[column].isnull().sum()

        for column in categorical_columns

    ],

    "Missing Percentage": [

        (
            df[column].isnull().sum()
            /
            len(df)
        )
        *
        100

        for column in categorical_columns

    ]

})


# ============================================================
# 27. SAVE CATEGORICAL SCREENING
# ============================================================

categorical_screening_path = os.path.join(

    CATEGORICAL_PATH,

    "Categorical_Feature_Screening.xlsx"

)


categorical_screening.to_excel(

    categorical_screening_path,

    index=False

)


print()
print(
    "Categorical screening saved:"
)

print(
    categorical_screening_path
)


# ============================================================
# 28. SELECT FINAL NUMERICAL FEATURES
# ============================================================

print()
print("============================================================")
print("6. FINAL FEATURE SELECTION")
print("============================================================")


# Important business-analysis numerical features

preferred_numerical_features = [

    "Sales",

    "Quantity",

    "Discount",

    "Profit",

    "Shipping Cost",

    "Sales per Quantity",

    "Discount Amount",

    "Net Sales",

    "Shipping Days",

    "Profit Margin",

    "Customer Order Count",

    "Customer Total Sales",

    "Customer Total Profit",

    "Customer Average Order Value",

    "Product Order Count",

    "Product Total Sales",

    "Product Total Profit",

    "Product Average Sales"

]


# Keep only features that actually exist

selected_numerical_features = [

    column

    for column in preferred_numerical_features

    if column in df.columns

]


# Remove constant features

selected_numerical_features = [

    column

    for column in selected_numerical_features

    if column not in constant_features

]


print()
print(
    "Selected numerical features:"
)


for column in selected_numerical_features:

    print(
        "-",
        column
    )


# ============================================================
# 29. SELECT CATEGORICAL FEATURES
# ============================================================

preferred_categorical_features = [

    "Ship Mode",

    "Segment",

    "City",

    "State",

    "Country",

    "Market",

    "Region",

    "Category",

    "Sub-Category",

    "Order Priority",

    "Order Month Name",

    "Order Day Name",

    "Quantity Category",

    "Discount Category",

    "Customer Profitability",

    "Product Performance",

    "Profit Status",

    "Profit Category"

]


selected_categorical_features = [

    column

    for column in preferred_categorical_features

    if column in df.columns

]


print()
print(
    "Selected categorical features:"
)


for column in selected_categorical_features:

    print(
        "-",
        column
    )


# ============================================================
# 30. SELECT IMPORTANT IDENTIFIER COLUMNS
# ============================================================

identifier_features = [

    "Order ID",

    "Order Date",

    "Ship Date",

    "Customer ID",

    "Customer Name",

    "Product ID",

    "Product Name"

]


selected_identifier_features = [

    column

    for column in identifier_features

    if column in df.columns

]


# ============================================================
# 31. CREATE FINAL FEATURE LIST
# ============================================================

final_feature_list = (

    selected_identifier_features

    +

    selected_categorical_features

    +

    selected_numerical_features

)


# Remove duplicates while preserving order

final_feature_list = list(
    dict.fromkeys(
        final_feature_list
    )
)


print()
print(
    "Total final selected features:",
    len(final_feature_list)
)


print()
print("Final feature list:")


for column in final_feature_list:

    print(
        "-",
        column
    )


# ============================================================
# 32. CREATE FINAL DATASET
# ============================================================

final_dataset = df[
    final_feature_list
].copy()


print()
print("Final selected dataset shape:")

print(
    "Rows:",
    final_dataset.shape[0]
)

print(
    "Columns:",
    final_dataset.shape[1]
)


# ============================================================
# 33. SAVE FINAL DATASET
# ============================================================

final_dataset_path = os.path.join(

    FINAL_PATH,

    "Final_Selected_Features.xlsx"

)


final_dataset.to_excel(

    final_dataset_path,

    index=False

)


print()
print(
    "Final selected dataset saved:"
)

print(
    final_dataset_path
)


# ============================================================
# 34. SAVE FINAL FEATURE LIST
# ============================================================

final_feature_summary = pd.DataFrame({

    "Feature": final_feature_list,

    "Data Type": [

        str(
            df[column].dtype
        )

        for column in final_feature_list

    ],

    "Feature Type": [

        (
            "Numerical"
            if column in selected_numerical_features
            else
            "Categorical"
            if column in selected_categorical_features
            else
            "Identifier/Date"
        )

        for column in final_feature_list

    ],

    "Unique Values": [

        df[column].nunique()

        for column in final_feature_list

    ],

    "Missing Values": [

        df[column].isnull().sum()

        for column in final_feature_list

    ]

})


final_feature_summary_path = os.path.join(

    FINAL_PATH,

    "Final_Feature_Summary.xlsx"

)


final_feature_summary.to_excel(

    final_feature_summary_path,

    index=False

)


print()
print(
    "Final feature summary saved:"
)

print(
    final_feature_summary_path
)


# ============================================================
# 35. SAVE REMOVED FEATURES
# ============================================================

removed_features = [

    column

    for column in df.columns

    if column not in final_feature_list

]


removed_feature_summary = pd.DataFrame({

    "Removed Feature": removed_features,

    "Reason": [

        (
            "Not selected for final analytical dataset"
        )

        for column in removed_features

    ]

})


removed_features_path = os.path.join(

    FINAL_PATH,

    "Removed_Features.xlsx"

)


removed_feature_summary.to_excel(

    removed_features_path,

    index=False

)


print()
print(
    "Removed feature list saved:"
)

print(
    removed_features_path
)


# ============================================================
# 36. FINAL VALIDATION
# ============================================================

print()
print("============================================================")
print("FINAL FEATURE SELECTION VALIDATION")
print("============================================================")


print()
print(
    "Original dataset rows:",
    df.shape[0]
)


print(
    "Original dataset columns:",
    df.shape[1]
)


print(
    "Final selected rows:",
    final_dataset.shape[0]
)


print(
    "Final selected columns:",
    final_dataset.shape[1]
)


print(
    "Selected numerical features:",
    len(selected_numerical_features)
)


print(
    "Selected categorical features:",
    len(selected_categorical_features)
)


print(
    "Identifier/date features:",
    len(selected_identifier_features)
)


print(
    "Removed features:",
    len(removed_features)
)


print(
    "Missing values in final dataset:",
    final_dataset.isnull().sum().sum()
)


print(
    "Duplicate rows in final dataset:",
    final_dataset.duplicated().sum()
)


# ============================================================
# 37. OUTPUT FILE CHECK
# ============================================================

print()
print("===== OUTPUT FILE CHECK =====")


output_files = [

    numerical_output_path,

    numerical_screening_path,

    categorical_output_path,

    categorical_screening_path,

    correlation_excel_path,

    correlation_pairs_path,

    high_correlation_path,

    heatmap_path,

    final_dataset_path,

    final_feature_summary_path,

    removed_features_path

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
# 38. COMPLETION MESSAGE
# ============================================================

print()
print("============================================================")
print("FEATURE SELECTION COMPLETED SUCCESSFULLY")
print("============================================================")

print()
print("Main final dataset:")
print(
    final_dataset_path
)

print()
print("Feature summary:")
print(
    final_feature_summary_path
)

print()
print("Correlation heatmap:")
print(
    heatmap_path
)

print()
print("============================================================")