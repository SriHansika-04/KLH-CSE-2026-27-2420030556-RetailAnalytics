# ============================================================
# DISCOUNT SCENARIO ANALYSIS
# Retail Sales Analytics Project
# ============================================================

import os
import pandas as pd
import numpy as np
import joblib

# ------------------------------------------------------------
# BLOCK 1: Paths
# ------------------------------------------------------------

model_file = r"outputs\05_Machine_Learning\08_Final_Model\Final_XGBoost_Sales_Model.pkl"

input_file = r"outputs\03_Feature_Engineering\Feature_Engineered_Superstore.xlsx"

output_folder = r"outputs\09_Scenario_Analysis"

os.makedirs(output_folder, exist_ok=True)

output_file = os.path.join(
    output_folder,
    "Discount_Scenario.xlsx"
)

print("=" * 60)
print("DISCOUNT SCENARIO ANALYSIS")
print("=" * 60)

# ------------------------------------------------------------
# BLOCK 2: Load model and data
# ------------------------------------------------------------

model = joblib.load(model_file)

df = pd.read_excel(input_file)

print("\nModel loaded successfully.")
print("Dataset loaded successfully.")
print("Rows:", len(df))

# ------------------------------------------------------------
# BLOCK 3: Select rows for scenario analysis
# ------------------------------------------------------------
# We use a sample of actual transactions as the baseline.
# This keeps the scenario realistic while avoiding excessive
# prediction time.

sample_size = min(1000, len(df))

scenario_df = df.sample(
    n=sample_size,
    random_state=42
).copy()

# ------------------------------------------------------------
# BLOCK 4: Define model features
# ------------------------------------------------------------
# These are the same features used during ML sales prediction.

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

# Keep only features that exist in the dataset.

features = [
    col for col in features
    if col in scenario_df.columns
]

# ------------------------------------------------------------
# BLOCK 5: Create baseline prediction
# ------------------------------------------------------------

baseline_data = scenario_df[features].copy()

baseline_prediction = model.predict(
    baseline_data
)

scenario_df["Baseline_Predicted_Sales"] = (
    baseline_prediction
)

# ------------------------------------------------------------
# BLOCK 6: Test different discount levels
# ------------------------------------------------------------
# We test discount values from 0% to 50%.

discount_levels = [
    0.00,
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50
]

results = []

for discount in discount_levels:

    test_data = scenario_df[features].copy()

    # Change discount.

    test_data["Discount"] = discount

    # Update discount category because the model uses it.

    def discount_category(x):

        if x == 0:
            return "No Discount"

        elif x <= 0.20:
            return "Low Discount"

        elif x <= 0.40:
            return "Medium Discount"

        else:
            return "High Discount"

    test_data["Discount Category"] = (
        test_data["Discount"]
        .apply(discount_category)
    )

    predictions = model.predict(
        test_data
    )

    results.append({
        "Discount": discount,
        "Average_Predicted_Sales": predictions.mean(),
        "Total_Predicted_Sales": predictions.sum()
    })

# ------------------------------------------------------------
# BLOCK 7: Create results table
# ------------------------------------------------------------

results_df = pd.DataFrame(results)

results_df["Change_from_0_Discount"] = (
    results_df["Total_Predicted_Sales"]
    - results_df.loc[0, "Total_Predicted_Sales"]
)

results_df["Percentage_Change"] = (
    results_df["Change_from_0_Discount"]
    / results_df.loc[0, "Total_Predicted_Sales"]
) * 100

# ------------------------------------------------------------
# BLOCK 8: Save results
# ------------------------------------------------------------

with pd.ExcelWriter(
    output_file,
    engine="openpyxl"
) as writer:

    results_df.to_excel(
        writer,
        sheet_name="Discount_Scenarios",
        index=False
    )

    scenario_df.to_excel(
        writer,
        sheet_name="Baseline_Data",
        index=False
    )

# ------------------------------------------------------------
# BLOCK 9: Display results
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("DISCOUNT SCENARIO RESULTS")
print("=" * 60)

print(results_df)

print("\nOutput saved to:")
print(output_file)

print("\nDone!")