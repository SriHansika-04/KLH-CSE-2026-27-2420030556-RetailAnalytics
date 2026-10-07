# ============================================================
# RETAIL SALES ANALYTICS
# MODEL EXPLAINABILITY
# ============================================================
# Techniques:
# 1. SHAP
# 2. LIME
#
# Model:
# XGBoost Sales Prediction Model
#
# Target:
# Sales
#
# Output:
# outputs/10_Model_Explainability/
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os
import warnings
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import shap
from lime.lime_tabular import LimeTabularExplainer

from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


# ============================================================
# 2. PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


ML_PREPARED_FILE = os.path.join(
    BASE_DIR,
    "outputs",
    "05_Machine_Learning",
    "01_Data_Split",
    "ML_Prepared_Dataset.xlsx"
)


MODEL_FILE = os.path.join(
    BASE_DIR,
    "outputs",
    "05_Machine_Learning",
    "08_Final_Model",
    "Final_XGBoost_Sales_Model.pkl"
)


EXPLAINABILITY_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "10_Model_Explainability"
)


os.makedirs(
    EXPLAINABILITY_DIR,
    exist_ok=True
)


# ============================================================
# 3. START
# ============================================================

print("=" * 70)
print("RETAIL SALES ANALYTICS - MODEL EXPLAINABILITY")
print("=" * 70)


# ============================================================
# 4. LOAD XGBOOST MODEL
# ============================================================

print("\n[1] Loading saved XGBoost model...")


if not os.path.exists(MODEL_FILE):

    raise FileNotFoundError(
        f"\nXGBoost model not found:\n{MODEL_FILE}"
    )


xgb_pipeline = joblib.load(
    MODEL_FILE
)


print(
    "XGBoost model loaded successfully."
)


# ============================================================
# 5. LOAD ML DATASET
# ============================================================

print(
    "\n[2] Loading prepared ML dataset..."
)


if not os.path.exists(ML_PREPARED_FILE):

    raise FileNotFoundError(
        f"\nPrepared ML dataset not found:\n"
        f"{ML_PREPARED_FILE}"
    )


ml_df = pd.read_excel(
    ML_PREPARED_FILE
)


print(
    "Prepared ML dataset loaded successfully."
)

print(
    "Rows    :",
    ml_df.shape[0]
)

print(
    "Columns :",
    ml_df.shape[1]
)


# ============================================================
# 6. FEATURES AND TARGET
# ============================================================

target = "Sales"


candidate_features = [

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


selected_features = [
    feature
    for feature in candidate_features
    if feature in ml_df.columns
]


print(
    "\nFeatures used for explainability:"
)

for feature in selected_features:
    print(
        " -",
        feature
    )


# ============================================================
# 7. CREATE X AND y
# ============================================================

X = ml_df[
    selected_features
].copy()


y = ml_df[
    target
].copy()


# ============================================================
# 8. TRAIN-TEST SPLIT
# ============================================================

print(
    "\n[3] Recreating train-test split..."
)


X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42

)


X_train = X_train.reset_index(
    drop=True
)

X_test = X_test.reset_index(
    drop=True
)

y_train = y_train.reset_index(
    drop=True
)

y_test = y_test.reset_index(
    drop=True
)


print(
    "Training rows :",
    len(X_train)
)

print(
    "Testing rows  :",
    len(X_test)
)


# ============================================================
# 9. PREDICTIONS
# ============================================================

print(
    "\n[4] Generating XGBoost predictions..."
)


test_predictions = xgb_pipeline.predict(
    X_test
)


print(
    "Predictions generated successfully."
)


# ============================================================
# 10. ACCESS PIPELINE
# ============================================================

preprocessor = (
    xgb_pipeline.named_steps[
        "preprocessor"
    ]
)


xgb_model = (
    xgb_pipeline.named_steps[
        "model"
    ]
)


# ============================================================
# 11. SHAP
# ============================================================

print("\n")
print("=" * 70)
print("SHAP EXPLAINABILITY")
print("=" * 70)


print(
    "\n[5] Transforming test data for SHAP..."
)


X_test_transformed = (
    preprocessor.transform(
        X_test
    )
)


if hasattr(
    X_test_transformed,
    "toarray"
):

    X_test_transformed = (
        X_test_transformed.toarray()
    )


feature_names = (
    preprocessor
    .get_feature_names_out()
)


print(
    "Transformed feature count:",
    len(feature_names)
)


# ============================================================
# 12. SHAP SAMPLE
# ============================================================

SHAP_SAMPLE_SIZE = min(
    1000,
    X_test_transformed.shape[0]
)


X_shap = (
    X_test_transformed[
        :SHAP_SAMPLE_SIZE
    ]
)


print(
    "SHAP sample size:",
    SHAP_SAMPLE_SIZE
)


# ============================================================
# 13. CALCULATE SHAP
# ============================================================

print(
    "\n[6] Calculating SHAP values..."
)


explainer = shap.TreeExplainer(
    xgb_model
)


shap_values = (
    explainer.shap_values(
        X_shap
    )
)


print(
    "SHAP values calculated successfully."
)


# ============================================================
# 14. SHAP FEATURE IMPORTANCE
# ============================================================

print(
    "\n[7] Creating SHAP feature importance..."
)


mean_abs_shap = (
    np.abs(
        shap_values
    ).mean(
        axis=0
    )
)


shap_importance = pd.DataFrame({

    "Feature":
        feature_names,

    "Mean Absolute SHAP":
        mean_abs_shap

})


shap_importance = (
    shap_importance
    .sort_values(
        "Mean Absolute SHAP",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)


shap_importance.insert(
    0,
    "Rank",
    range(
        1,
        len(shap_importance) + 1
    )
)


# Save Excel

shap_importance.to_excel(

    os.path.join(
        EXPLAINABILITY_DIR,
        "SHAP_Feature_Importance.xlsx"
    ),

    index=False
)


# Save CSV

shap_importance.to_csv(

    os.path.join(
        EXPLAINABILITY_DIR,
        "SHAP_Feature_Importance.csv"
    ),

    index=False
)


print(
    "SHAP feature importance table saved."
)


# ============================================================
# 15. SHAP BAR PLOT
# ============================================================

print(
    "\n[8] Creating SHAP feature importance plot..."
)


TOP_N = min(
    20,
    len(shap_importance)
)


top_features = (
    shap_importance
    .head(TOP_N)
    .sort_values(
        "Mean Absolute SHAP",
        ascending=True
    )
)


plt.figure(
    figsize=(10, 8)
)


plt.barh(

    top_features["Feature"],

    top_features[
        "Mean Absolute SHAP"
    ]

)


plt.xlabel(
    "Mean Absolute SHAP Value"
)


plt.ylabel(
    "Feature"
)


plt.title(
    "SHAP Feature Importance - XGBoost Sales Prediction"
)


plt.tight_layout()


plt.savefig(

    os.path.join(
        EXPLAINABILITY_DIR,
        "SHAP_Feature_Importance.png"
    ),

    dpi=300,

    bbox_inches="tight"

)


plt.close()


print(
    "SHAP feature importance plot saved."
)


# ============================================================
# 16. SHAP SUMMARY PLOT
# ============================================================

print(
    "\n[9] Creating SHAP summary plot..."
)


shap.summary_plot(

    shap_values,

    X_shap,

    feature_names=feature_names,

    show=False,

    max_display=20

)


plt.title(
    "SHAP Summary Plot - XGBoost Sales Prediction"
)


plt.tight_layout()


plt.savefig(

    os.path.join(
        EXPLAINABILITY_DIR,
        "SHAP_Summary_Plot.png"
    ),

    dpi=300,

    bbox_inches="tight"

)


plt.close()


print(
    "SHAP summary plot saved."
)


# ============================================================
# 17. POWER BI SHAP DATASET
# ============================================================

print(
    "\n[10] Creating Power BI SHAP dataset..."
)


powerbi_shap = (
    shap_importance
    .head(20)
    .copy()
)


powerbi_shap.to_excel(

    os.path.join(
        EXPLAINABILITY_DIR,
        "PowerBI_SHAP_Feature_Importance.xlsx"
    ),

    index=False

)


powerbi_shap.to_csv(

    os.path.join(
        EXPLAINABILITY_DIR,
        "PowerBI_SHAP_Feature_Importance.csv"
    ),

    index=False

)


print(
    "Power BI SHAP dataset saved."
)
# ============================================================
# 18. LIME EXPLAINABILITY
# ============================================================

print("\n")
print("=" * 70)
print("LIME EXPLAINABILITY")
print("=" * 70)


# ============================================================
# 18.1 DEFINE FEATURE TYPES
# ============================================================
#
# IMPORTANT:
#
# The saved XGBoost pipeline was trained with categorical
# variables and numerical variables.
#
# LIME requires categorical variables to be represented using
# integer codes internally.
#
# Therefore, we explicitly define the feature types here
# instead of relying on dataframe dtypes or the ColumnTransformer.
#
# This avoids the previous problem where:
#
#     Categorical features: 0
#
# was detected and LIME treated everything as numerical.
# ============================================================


# ------------------------------------------------------------
# CATEGORICAL FEATURES
# ------------------------------------------------------------

categorical_features = [

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
    "Quantity Category",
    "Discount Category"

]


# ------------------------------------------------------------
# NUMERICAL FEATURES
# ------------------------------------------------------------

numeric_features = [

    "Quantity",
    "Discount",
    "Shipping Cost",
    "Shipping Days",
    "Order Year",
    "Order Month",
    "Order Quarter",
    "Order Week",
    "Order Day of Week",
    "Is Weekend"

]


# ------------------------------------------------------------
# Keep only features actually available in the dataset
# ------------------------------------------------------------

categorical_features = [

    feature

    for feature in categorical_features

    if feature in selected_features

]


numeric_features = [

    feature

    for feature in numeric_features

    if feature in selected_features

]


print(
    "\nCategorical features:",
    len(categorical_features)
)


for feature in categorical_features:

    print(
        " -",
        feature
    )


print(
    "\nNumerical features:",
    len(numeric_features)
)


for feature in numeric_features:

    print(
        " -",
        feature
    )


# ------------------------------------------------------------
# Safety validation
# ------------------------------------------------------------

all_detected_features = (
    categorical_features
    + numeric_features
)


missing_type_features = [

    feature

    for feature in selected_features

    if feature not in all_detected_features

]


if len(missing_type_features) > 0:

    print(
        "\nWARNING: Some features were not assigned a type:"
    )

    for feature in missing_type_features:

        print(
            " -",
            feature
        )


# ============================================================
# 19. LIME DATA PREPARATION
# ============================================================

print(
    "\n[11] Preparing data for LIME..."
)


lime_categories = {}

lime_mappings = {}


X_lime_train_encoded = pd.DataFrame(
    index=X_train.index
)


X_lime_test_encoded = pd.DataFrame(
    index=X_test.index
)


# ============================================================
# 19.1 ENCODE FEATURES FOR LIME
# ============================================================

for feature in selected_features:

    # ========================================================
    # CATEGORICAL FEATURES
    # ========================================================

    if feature in categorical_features:

        # ----------------------------------------------------
        # Get unique original category values
        # ----------------------------------------------------

        categories = (

            X[feature]
            .dropna()
            .astype(str)
            .unique()
            .tolist()

        )


        # ----------------------------------------------------
        # Safety check
        # ----------------------------------------------------

        if len(categories) == 0:

            categories = [
                "Unknown"
            ]


        # ----------------------------------------------------
        # Create:
        #
        # Original category -> integer code
        #
        # Example:
        #
        # Consumer  -> 0
        # Corporate -> 1
        # Home Office -> 2
        # ----------------------------------------------------

        mapping = {

            value: index

            for index, value
            in enumerate(categories)

        }


        lime_categories[
            feature
        ] = categories


        lime_mappings[
            feature
        ] = mapping


        # ----------------------------------------------------
        # Encode training data
        # ----------------------------------------------------

        X_lime_train_encoded[
            feature
        ] = (

            X_train[
                feature
            ]
            .astype(str)
            .map(mapping)
            .fillna(0)
            .astype(float)

        )


        # ----------------------------------------------------
        # Encode testing data
        # ----------------------------------------------------

        X_lime_test_encoded[
            feature
        ] = (

            X_test[
                feature
            ]
            .astype(str)
            .map(mapping)
            .fillna(0)
            .astype(float)

        )


    # ========================================================
    # NUMERICAL FEATURES
    # ========================================================

    else:

        X_lime_train_encoded[
            feature
        ] = pd.to_numeric(

            X_train[
                feature
            ],

            errors="coerce"

        )


        X_lime_test_encoded[
            feature
        ] = pd.to_numeric(

            X_test[
                feature
            ],

            errors="coerce"

        )


# ============================================================
# 19.2 CLEAN LIME DATA
# ============================================================

print(
    "\nCleaning LIME data..."
)


# ------------------------------------------------------------
# Replace infinite values with NaN
# ------------------------------------------------------------

X_lime_train_encoded = (

    X_lime_train_encoded
    .replace(
        [np.inf, -np.inf],
        np.nan
    )

)


X_lime_test_encoded = (

    X_lime_test_encoded
    .replace(
        [np.inf, -np.inf],
        np.nan
    )

)


# ============================================================
# 19.3 CALCULATE NUMERICAL MEDIANS
# ============================================================

lime_medians = {}


for feature in numeric_features:

    median_value = (

        pd.to_numeric(

            X_train[
                feature
            ],

            errors="coerce"

        )

        .replace(

            [np.inf, -np.inf],

            np.nan

        )

        .median()

    )


    if pd.isna(
        median_value
    ):

        median_value = 0.0


    lime_medians[
        feature
    ] = median_value


# ============================================================
# 19.4 FILL NUMERICAL MISSING VALUES
# ============================================================

for feature in numeric_features:

    X_lime_train_encoded[
        feature
    ] = (

        X_lime_train_encoded[
            feature
        ]

        .fillna(
            lime_medians[
                feature
            ]
        )

    )


    X_lime_test_encoded[
        feature
    ] = (

        X_lime_test_encoded[
            feature
        ]

        .fillna(
            lime_medians[
                feature
            ]
        )

    )


# ============================================================
# 19.5 FINAL NaN CLEANING
# ============================================================

X_lime_train_encoded = (

    X_lime_train_encoded
    .fillna(0)

)


X_lime_test_encoded = (

    X_lime_test_encoded
    .fillna(0)

)


# ============================================================
# 19.6 FINAL FINITE-VALUE SAFETY
# ============================================================

X_lime_train_encoded = pd.DataFrame(

    np.nan_to_num(

        X_lime_train_encoded.values,

        nan=0.0,

        posinf=0.0,

        neginf=0.0

    ),

    columns=selected_features

)


X_lime_test_encoded = pd.DataFrame(

    np.nan_to_num(

        X_lime_test_encoded.values,

        nan=0.0,

        posinf=0.0,

        neginf=0.0

    ),

    columns=selected_features

)


print(
    "LIME data preparation completed."
)


print(
    "Training NaN count:",
    X_lime_train_encoded.isna().sum().sum()
)


print(
    "Testing NaN count:",
    X_lime_test_encoded.isna().sum().sum()
)


print(
    "Training infinite values:",
    np.isinf(
        X_lime_train_encoded.values
    ).sum()
)


print(
    "Testing infinite values:",
    np.isinf(
        X_lime_test_encoded.values
    ).sum()
)


# ============================================================
# 20. LIME CATEGORICAL INFORMATION
# ============================================================

categorical_indices = [

    selected_features.index(
        feature
    )

    for feature
    in categorical_features

]


categorical_names = {

    selected_features.index(
        feature
    ):

    lime_categories[
        feature
    ]

    for feature
    in categorical_features

}


print(
    "\nLIME categorical indices:",
    categorical_indices
)


print(
    "Number of LIME categorical features:",
    len(categorical_indices)
)


# ============================================================
# 21. LIME PREDICTION FUNCTION
# ============================================================
#
# LIME works with numerical category codes.
#
# Example:
#
#     Consumer  -> 0
#     Corporate -> 1
#     Home Office -> 2
#
# But the original XGBoost pipeline expects:
#
#     Consumer
#     Corporate
#     Home Office
#
# Therefore this function converts the LIME codes back into
# the original categorical strings before prediction.
# ============================================================


def lime_predict(data):


    # ========================================================
    # Convert incoming LIME data to NumPy
    # ========================================================

    data = np.asarray(
        data,
        dtype=float
    )


    # ========================================================
    # Create dataframe with original feature names
    # ========================================================

    converted = pd.DataFrame(

        data,

        columns=selected_features

    )


    # ========================================================
    # NUMERICAL FEATURES
    # ========================================================

    for feature in numeric_features:

        converted[
            feature
        ] = pd.to_numeric(

            converted[
                feature
            ],

            errors="coerce"

        )


        # ----------------------------------------------------
        # Replace infinite values
        # ----------------------------------------------------

        converted[
            feature
        ] = (

            converted[
                feature
            ]

            .replace(

                [np.inf, -np.inf],

                np.nan

            )

        )


        # ----------------------------------------------------
        # Fill missing values with training median
        # ----------------------------------------------------

        converted[
            feature
        ] = (

            converted[
                feature
            ]

            .fillna(

                lime_medians[
                    feature
                ]

            )

        )


        # ----------------------------------------------------
        # Final numerical fallback
        # ----------------------------------------------------

        converted[
            feature
        ] = (

            converted[
                feature
            ]

            .fillna(0)

        )


    # ========================================================
    # CATEGORICAL FEATURES
    # ========================================================

    for feature in categorical_features:

        mapping = lime_mappings[
            feature
        ]


        # ----------------------------------------------------
        # Reverse mapping
        #
        # Integer code -> Original category
        # ----------------------------------------------------

        reverse_mapping = {

            index: value

            for value, index
            in mapping.items()

        }


        # ----------------------------------------------------
        # Convert LIME numerical category codes back to
        # original categorical values
        # ----------------------------------------------------

        converted[
            feature
        ] = (

            converted[
                feature
            ]

            .round()

            .clip(

                lower=0,

                upper=len(mapping) - 1

            )

            .astype(int)

            .map(
                reverse_mapping
            )

        )


        # ----------------------------------------------------
        # Safety fallback
        # ----------------------------------------------------

        first_category = (

            list(
                mapping.keys()
            )[0]

        )


        converted[
            feature
        ] = (

            converted[
                feature
            ]

            .fillna(
                first_category
            )

            .astype(str)

        )


    # ========================================================
    # FINAL MODEL INPUT
    # ========================================================

    prediction_data = converted[
        selected_features
    ].copy()


    # ========================================================
    # Ensure numerical columns are numeric
    # ========================================================

    for feature in numeric_features:

        prediction_data[
            feature
        ] = pd.to_numeric(

            prediction_data[
                feature
            ],

            errors="coerce"

        )


        prediction_data[
            feature
        ] = (

            prediction_data[
                feature
            ]

            .replace(

                [np.inf, -np.inf],

                np.nan

            )

            .fillna(

                lime_medians[
                    feature
                ]

            )

            .fillna(0)

        )


    # ========================================================
    # Ensure categorical columns are strings
    # ========================================================

    for feature in categorical_features:

        prediction_data[
            feature
        ] = (

            prediction_data[
                feature
            ]
            .astype(str)

        )


    # ========================================================
    # FINAL PREDICTION
    # ========================================================

    return xgb_pipeline.predict(
        prediction_data
    )


# ============================================================
# 22. CREATE LIME EXPLAINER
# ============================================================

print(
    "\n[12] Creating LIME explainer..."
)


lime_explainer = LimeTabularExplainer(

    training_data=
        X_lime_train_encoded.values,

    feature_names=
        selected_features,

    categorical_features=
        categorical_indices,

    categorical_names=
        categorical_names,

    mode="regression",

    # IMPORTANT:
    #
    # Keep False.
    #
    # This avoids the previous scipy truncnorm
    # domain error.
    #

    discretize_continuous=False,

    random_state=42

)


print(
    "LIME explainer created."
)


# ============================================================
# 23. GENERATE LIME EXPLANATIONS
# ============================================================

print(
    "\n[13] Generating LIME explanations..."
)


lime_results = []


NUMBER_OF_EXAMPLES = min(

    3,

    len(
        X_lime_test_encoded
    )

)


for i in range(
    NUMBER_OF_EXAMPLES
):


    print(

        f"Generating LIME explanation "
        f"{i + 1}/{NUMBER_OF_EXAMPLES}..."

    )


    # ========================================================
    # Get test instance
    # ========================================================

    instance = (

        X_lime_test_encoded
        .iloc[i]
        .values
        .astype(float)

    )


    # ========================================================
    # Generate explanation
    # ========================================================

    explanation = (

        lime_explainer.explain_instance(

            instance,

            lime_predict,

            num_features=10

        )

    )


    # ========================================================
    # Save HTML explanation
    # ========================================================

    explanation.save_to_file(

        os.path.join(

            EXPLAINABILITY_DIR,

            f"LIME_Explanation_{i + 1}.html"

        )

    )


    # ========================================================
    # Store explanation results
    # ========================================================

    for feature, weight in (

        explanation.as_list()

    ):

        lime_results.append({

            "Example":
                i + 1,

            "Actual Sales":
                y_test.iloc[i],

            "Predicted Sales":
                test_predictions[i],

            "Feature":
                feature,

            "LIME Weight":
                weight

        })


print(
    "LIME explanations generated."
)


# ============================================================
# 24. SAVE LIME RESULTS
# ============================================================

lime_results_df = pd.DataFrame(
    lime_results
)


# ============================================================
# Save Excel
# ============================================================

lime_results_df.to_excel(

    os.path.join(

        EXPLAINABILITY_DIR,

        "LIME_Local_Explanations.xlsx"

    ),

    index=False

)


# ============================================================
# Save CSV
# ============================================================

lime_results_df.to_csv(

    os.path.join(

        EXPLAINABILITY_DIR,

        "LIME_Local_Explanations.csv"

    ),

    index=False

)


print(
    "LIME results saved."
)


# ============================================================
# 25. CREATE EXPLAINABILITY SUMMARY
# ============================================================

summary = pd.DataFrame({

    "Item": [

        "Model",

        "Explainability Method 1",

        "Explainability Method 2",

        "SHAP Sample Size",

        "LIME Examples",

        "Target Variable",

        "Categorical Features",

        "Numerical Features"

    ],

    "Value": [

        "XGBoost",

        "SHAP",

        "LIME",

        SHAP_SAMPLE_SIZE,

        NUMBER_OF_EXAMPLES,

        "Sales",

        len(categorical_features),

        len(numeric_features)

    ]

})


summary.to_excel(

    os.path.join(

        EXPLAINABILITY_DIR,

        "Explainability_Summary.xlsx"

    ),

    index=False

)


# ============================================================
# 26. COMPLETION
# ============================================================

print("\n")
print("=" * 70)
print("MODEL EXPLAINABILITY COMPLETED")
print("=" * 70)


print(
    "\nOutput folder:"
)


print(
    EXPLAINABILITY_DIR
)


print(
    "\nSHAP outputs created:"
)


print(
    " - SHAP_Feature_Importance.png"
)


print(
    " - SHAP_Summary_Plot.png"
)


print(
    " - SHAP_Feature_Importance.xlsx"
)


print(
    " - SHAP_Feature_Importance.csv"
)


print(
    " - PowerBI_SHAP_Feature_Importance.xlsx"
)


print(
    " - PowerBI_SHAP_Feature_Importance.csv"
)


print(
    "\nLIME outputs created:"
)


print(
    " - LIME_Explanation_1.html"
)


print(
    " - LIME_Explanation_2.html"
)


print(
    " - LIME_Explanation_3.html"
)


print(
    " - LIME_Local_Explanations.xlsx"
)


print(
    " - LIME_Local_Explanations.csv"
)


print(
    " - Explainability_Summary.xlsx"
)


print(
    "\nSHAP and LIME explainability completed successfully."
)


print("=" * 70)