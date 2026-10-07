# ============================================================
# QUANTITY SCENARIO ANALYSIS
# Retail Sales Analytics Project
# ============================================================
# Purpose:
# This script studies how changing the order quantity affects
# XGBoost-predicted sales.
#
# The already-trained XGBoost model is used.
# No model retraining is required.
# ============================================================

import os
import pandas as pd
import joblib


# ------------------------------------------------------------
# BLOCK 1: Define file paths
# ------------------------------------------------------------
# This block defines where the trained XGBoost model,
# feature-engineered dataset, and scenario output will be stored.

model_file = r"outputs\05_Machine_Learning\08_Final_Model\Final_XGBoost_Sales_Model.pkl"

input_file = r"outputs\03_Feature_Engineering\Feature_Engineered_Superstore.xlsx"

output_folder = r"outputs\09_Scenario_Analysis"

os.makedirs(output_folder, exist_ok=True)

output_file = os.path.join(
    output_folder,
    "Quantity_Scenario.xlsx"
)

print("=" * 60)
print("QUANTITY SCENARIO ANALYSIS")
print("=" * 60)


# ------------------------------------------------------------
# BLOCK 2: Load the trained XGBoost model and dataset
# ------------------------------------------------------------
# The saved XGBoost pipeline is loaded.
# The feature-engineered Superstore dataset is also loaded.

model = joblib.load(model_file)

df = pd.read_excel(input_file)

print("\nModel loaded successfully.")
print("Dataset loaded successfully.")
print("Total rows:", len(df))


# ------------------------------------------------------------
# BLOCK 3: Select a sample of transactions
# ------------------------------------------------------------
# We use up to 1,000 actual transactions for scenario analysis.
# This keeps the analysis realistic and computationally efficient.

sample_size = min(1000, len(df))

scenario_df = df.sample(
    n=sample_size,
    random_state=42
).copy()

print("Rows selected for scenario analysis:", sample_size)


# ------------------------------------------------------------
# BLOCK 4: Define the same features used by the ML model
# ------------------------------------------------------------
# These should match the features used during XGBoost training.

features = [
    "Quantity",
    "Discount",
    "Shipping Cost",
    "Shipping Days",
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
    "Order Year",
    "Order Month",
    "Order Quarter",
    "Order Week",
    "Order Day of Week",
    "Is Weekend",
    "Quantity Category",
    "Discount Category"
]


# ------------------------------------------------------------
# BLOCK 5: Check that required features exist
# ------------------------------------------------------------
# This prevents confusing errors if a column is missing.

missing_features = [
    col for col in features
    if col not in scenario_df.columns
]

if missing_features:

    print("\nERROR: The following required features are missing:")

    for col in missing_features:
        print("-", col)

    raise ValueError(
        "Required ML features are missing from the dataset."
    )


# ------------------------------------------------------------
# BLOCK 6: Create the baseline prediction
# ------------------------------------------------------------
# The baseline represents the predicted sales using the
# original quantity values from the actual transactions.

baseline_data = scenario_df[features].copy()

baseline_prediction = model.predict(
    baseline_data
)

scenario_df["Baseline_Predicted_Sales"] = (
    baseline_prediction
)

baseline_total = baseline_prediction.sum()

baseline_average = baseline_prediction.mean()

print("\nBaseline prediction calculated.")

print(
    "Baseline Average Predicted Sales:",
    round(baseline_average, 2)
)

print(
    "Baseline Total Predicted Sales:",
    round(baseline_total, 2)
)


# ------------------------------------------------------------
# BLOCK 7: Define quantity scenarios
# ------------------------------------------------------------
# We test several different quantity levels.
#
# This allows us to study how predicted sales change when
# quantity is systematically increased or decreased.

quantity_levels = [
    1,
    2,
    3,
    4,
    5,
    6,
    8,
    10,
    12
]


# ------------------------------------------------------------
# BLOCK 8: Run the quantity scenarios
# ------------------------------------------------------------
# For every quantity level:
#
# 1. Copy the original transaction features.
# 2. Replace Quantity with the scenario quantity.
# 3. Update Quantity Category.
# 4. Generate XGBoost predictions.
# 5. Store the results.

results = []


def quantity_category(quantity):

    if quantity <= 2:
        return "Low Quantity"

    elif quantity <= 5:
        return "Medium Quantity"

    else:
        return "High Quantity"


for quantity in quantity_levels:

    test_data = scenario_df[features].copy()

    # Change quantity.

    test_data["Quantity"] = quantity

    # Update the derived Quantity Category.

    test_data["Quantity Category"] = (
        test_data["Quantity"]
        .apply(quantity_category)
    )

    # Generate predictions.

    predictions = model.predict(
        test_data
    )

    total_predicted_sales = predictions.sum()

    average_predicted_sales = predictions.mean()

    # Compare against baseline.

    change_from_baseline = (
        total_predicted_sales
        - baseline_total
    )

    percentage_change = (
        change_from_baseline
        / baseline_total
    ) * 100

    results.append({

        "Quantity": quantity,

        "Average_Predicted_Sales":
            average_predicted_sales,

        "Total_Predicted_Sales":
            total_predicted_sales,

        "Change_from_Baseline":
            change_from_baseline,

        "Percentage_Change":
            percentage_change
    })


# ------------------------------------------------------------
# BLOCK 9: Create the final scenario results table
# ------------------------------------------------------------

results_df = pd.DataFrame(results)


# ------------------------------------------------------------
# BLOCK 10: Round values for easier presentation
# ------------------------------------------------------------
# The raw prediction values can contain many decimal places.
# Rounded values make the Excel output easier to read.

results_df["Average_Predicted_Sales"] = (
    results_df["Average_Predicted_Sales"]
    .round(2)
)

results_df["Total_Predicted_Sales"] = (
    results_df["Total_Predicted_Sales"]
    .round(2)
)

results_df["Change_from_Baseline"] = (
    results_df["Change_from_Baseline"]
    .round(2)
)

results_df["Percentage_Change"] = (
    results_df["Percentage_Change"]
    .round(2)
)


# ------------------------------------------------------------
# BLOCK 11: Save results to Excel
# ------------------------------------------------------------
# Two sheets are created:
#
# 1. Quantity_Scenarios
#    Contains the main scenario analysis.
#
# 2. Baseline_Data
#    Contains the sampled transactions and their baseline
#    predicted sales.

with pd.ExcelWriter(
    output_file,
    engine="openpyxl"
) as writer:

    results_df.to_excel(
        writer,
        sheet_name="Quantity_Scenarios",
        index=False
    )

    scenario_df.to_excel(
        writer,
        sheet_name="Baseline_Data",
        index=False
    )


# ------------------------------------------------------------
# BLOCK 12: Display the final results
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("QUANTITY SCENARIO RESULTS")
print("=" * 60)

print(
    results_df.to_string(
        index=False
    )
)


# ------------------------------------------------------------
# BLOCK 13: Display output location
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("OUTPUT")
print("=" * 60)

print("Excel file saved successfully:")
print(output_file)

print("\nQuantity Scenario Analysis completed successfully!")