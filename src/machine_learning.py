# ============================================================
# RETAIL SALES ANALYTICS
# MACHINE LEARNING + DEEP LEARNING + HYBRID MODEL
# ============================================================
#
# SALES PREDICTION MODELS
#
# 1. Linear Regression          - ML Baseline
# 2. Decision Tree              - ML Baseline
# 3. Random Forest              - ML Baseline
# 4. XGBoost                    - Advanced ML Baseline
# 5. LSTM                       - Deep Learning Baseline
# 6. GRU                        - Deep Learning Baseline
# 7. LSTM + XGBoost             - Hybrid Experiment
# 8. LSTM + Random Forest       - Hybrid Experiment
# 9. GRU + XGBoost              - Hybrid Experiment
# 10. GRU + Random Forest       - Hybrid Experiment
#
# CUSTOMER ANALYSIS
#
# 11. K-Means Customer Segmentation
#
# TARGET
# Sales
#
# REGRESSION METRICS
# - MAE
# - MSE
# - RMSE
# - R2
# - MAPE
# - SMAPE
#
# ============================================================
# PROPOSED HYBRID ARCHITECTURE
# ============================================================
#
# Historical Monthly Sales
#          |
#          v
#        LSTM
#          |
#          v
# LSTM Temporal Prediction
#          |
#          v
# Transaction Features + LSTM Temporal Feature
#          |
#          v
#      Final XGBoost
#          |
#          v
# Final Hybrid Sales Prediction
#
# The LSTM therefore contributes an actual learned temporal
# feature to the final XGBoost model.
#
# There is NO alpha/gamma weighting.
# There is NO residual correction.
#
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os

# Reduce TensorFlow messages BEFORE importing TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import warnings
import joblib

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.cluster import KMeans

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score
)

from xgboost import XGBRegressor

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    LSTM,
    GRU,
    Dense,
    Dropout
)

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import Huber


warnings.filterwarnings("ignore")

np.random.seed(42)
tf.random.set_seed(42)


# ============================================================
# 2. PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


INPUT_FILE = os.path.join(
    BASE_DIR,
    "outputs",
    "03_Feature_Engineering",
    "Feature_Engineered_Superstore.xlsx"
)


ML_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "05_Machine_Learning"
)


DATA_SPLIT_DIR = os.path.join(
    ML_DIR,
    "01_Data_Split"
)

LINEAR_DIR = os.path.join(
    ML_DIR,
    "02_Linear_Regression"
)

TREE_DIR = os.path.join(
    ML_DIR,
    "03_Decision_Tree"
)

RF_DIR = os.path.join(
    ML_DIR,
    "04_Random_Forest"
)

XGB_DIR = os.path.join(
    ML_DIR,
    "05_XGBoost"
)

COMPARISON_DIR = os.path.join(
    ML_DIR,
    "06_Model_Comparison"
)

CLUSTER_DIR = os.path.join(
    ML_DIR,
    "07_Customer_Segmentation"
)

FINAL_DIR = os.path.join(
    ML_DIR,
    "08_Final_Model"
)

LSTM_DIR = os.path.join(
    ML_DIR,
    "09_LSTM"
)

HYBRID_DIR = os.path.join(
    ML_DIR,
    "10_Hybrid_Model"
)

GRU_DIR = os.path.join(
    ML_DIR,
    "11_GRU"
)

HYBRID_LSTM_RF_DIR = os.path.join(
    ML_DIR,
    "12_Hybrid_LSTM_RF"
)

HYBRID_GRU_XGB_DIR = os.path.join(
    ML_DIR,
    "13_Hybrid_GRU_XGB"
)

HYBRID_GRU_RF_DIR = os.path.join(
    ML_DIR,
    "14_Hybrid_GRU_RF"
)


for folder in [
    DATA_SPLIT_DIR,
    LINEAR_DIR,
    TREE_DIR,
    RF_DIR,
    XGB_DIR,
    COMPARISON_DIR,
    CLUSTER_DIR,
    FINAL_DIR,
    LSTM_DIR,
    HYBRID_DIR,
    GRU_DIR,
    HYBRID_LSTM_RF_DIR,
    HYBRID_GRU_XGB_DIR,
    HYBRID_GRU_RF_DIR
]:

    os.makedirs(
        folder,
        exist_ok=True
    )


print("=" * 80)
print("RETAIL SALES ANALYTICS")
print("MACHINE LEARNING + DEEP LEARNING + HYBRID MODEL")
print("=" * 80)


# ============================================================
# 3. LOAD DATA
# ============================================================

print("\n[1] Loading Feature Engineered Dataset...")


if not os.path.exists(INPUT_FILE):

    raise FileNotFoundError(
        f"\nInput file not found:\n{INPUT_FILE}"
    )


df = pd.read_excel(
    INPUT_FILE
)


print("Dataset loaded successfully.")
print("Rows    :", df.shape[0])
print("Columns :", df.shape[1])


# ============================================================
# 4. BASIC VALIDATION
# ============================================================

print("\n[2] Dataset Validation")

print(
    "Missing values:",
    df.isnull().sum().sum()
)

print(
    "Duplicate rows :",
    df.duplicated().sum()
)


if "Sales" not in df.columns:

    raise ValueError(
        "Sales column was not found."
    )


if "Order Date" not in df.columns:

    raise ValueError(
        "Order Date column is required."
    )


print("Target variable: Sales")
print("Time variable  : Order Date")


# ============================================================
# 5. PREPARE SALES PREDICTION FEATURES
# ============================================================
#
# Sales-derived features are deliberately excluded.
#
# This prevents target leakage.
#
# ============================================================

print("\n[3] Preparing ML Features...")


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

    col

    for col in candidate_features

    if col in df.columns

]


print(
    "\nFeatures selected:"
)


for feature in selected_features:

    print(
        " -",
        feature
    )


if len(selected_features) == 0:

    raise ValueError(
        "No valid ML features were found."
    )


# ============================================================
# 6. CREATE ML DATASET
# ============================================================

ml_df = df[
    selected_features + [target]
].copy()


ml_df = ml_df.dropna(
    subset=[target]
)


# Numeric features
numeric_features = [

    col

    for col in selected_features

    if pd.api.types.is_numeric_dtype(
        ml_df[col]
    )

]


# Categorical features
categorical_features = [

    col

    for col in selected_features

    if col not in numeric_features

]


# Fill numeric missing values
for col in numeric_features:

    ml_df[col] = ml_df[col].fillna(
        ml_df[col].median()
    )


# Fill categorical missing values
for col in categorical_features:

    ml_df[col] = ml_df[col].fillna(
        "Unknown"
    )


print(
    "\nML Dataset Shape:",
    ml_df.shape
)


ml_df.to_excel(

    os.path.join(
        DATA_SPLIT_DIR,
        "ML_Prepared_Dataset.xlsx"
    ),

    index=False

)


# ============================================================
# 7. PREPARE MONTHLY SALES FIRST
# ============================================================
#
# The monthly dataset determines the common chronological
# test period.
#
# ML test period
#       =
# LSTM test period
#       =
# Hybrid test period
#
# ============================================================

print("\n[4] Preparing Monthly Sales Dataset...")


lstm_df = df[
    [
        "Order Date",
        "Sales"
    ]
].copy()


lstm_df["Order Date"] = pd.to_datetime(
    lstm_df["Order Date"],
    errors="coerce"
)


lstm_df = lstm_df.dropna(
    subset=[
        "Order Date",
        "Sales"
    ]
)


lstm_df["Year_Month"] = (
    lstm_df["Order Date"]
    .dt.to_period("M")
)


monthly_sales = (

    lstm_df

    .groupby(
        "Year_Month"
    )["Sales"]

    .sum()

    .reset_index()

)


monthly_sales["Date"] = (
    monthly_sales["Year_Month"]
    .dt.to_timestamp()
)


monthly_sales = monthly_sales[
    [
        "Date",
        "Sales"
    ]
].sort_values(
    "Date"
).reset_index(
    drop=True
)


# Create complete monthly index

complete_months = pd.date_range(

    start=monthly_sales["Date"].min(),

    end=monthly_sales["Date"].max(),

    freq="MS"

)


monthly_sales = (

    monthly_sales

    .set_index("Date")

    .reindex(
        complete_months,
        fill_value=0
    )

    .rename_axis("Date")

    .reset_index()

)


print(
    "\nMonthly observations:",
    len(monthly_sales)
)


print(
    "Monthly date range:",
    monthly_sales["Date"].min(),
    "to",
    monthly_sales["Date"].max()
)


monthly_sales.to_excel(

    os.path.join(
        LSTM_DIR,
        "LSTM_Monthly_Sales_Dataset.xlsx"
    ),

    index=False
)


# ============================================================
# 8. DEFINE COMMON TIME SPLIT
# ============================================================

LOOKBACK = 6


monthly_values = (
    monthly_sales["Sales"]
    .values
    .astype("float32")
)


monthly_dates = (
    monthly_sales["Date"]
    .values
)


monthly_split = int(
    len(monthly_values) * 0.80
)


lstm_train_values = (
    monthly_values[:monthly_split]
)


lstm_test_values = (
    monthly_values[monthly_split:]
)


lstm_train_dates = (
    monthly_dates[:monthly_split]
)


lstm_test_dates = (
    monthly_dates[monthly_split:]
)


hybrid_test_start_date = pd.to_datetime(
    lstm_test_dates[0]
)


print(
    "\nCommon chronological split:"
)


print(
    "Training months:",
    len(lstm_train_values)
)


print(
    "Testing months :",
    len(lstm_test_values)
)


print(
    "Test starts    :",
    hybrid_test_start_date
)


# ============================================================
# 9. CREATE TRANSACTION-LEVEL COMMON SPLIT
# ============================================================

print("\n[5] Creating Common Transaction Train-Test Split...")


model_df = df[
    selected_features
    +
    [
        target,
        "Order Date"
    ]
].copy()


model_df["Order Date"] = pd.to_datetime(
    model_df["Order Date"],
    errors="coerce"
)


model_df = model_df.dropna(
    subset=[
        "Order Date",
        target
    ]
)


model_df = model_df.sort_values(
    "Order Date"
).reset_index(
    drop=True
)


# Fill missing values
for col in numeric_features:

    model_df[col] = model_df[col].fillna(
        ml_df[col].median()
    )


for col in categorical_features:

    model_df[col] = model_df[col].fillna(
        "Unknown"
    )


# Same date cutoff as LSTM

train_df = model_df[
    model_df["Order Date"]
    <
    hybrid_test_start_date
].copy()


test_df = model_df[
    model_df["Order Date"]
    >=
    hybrid_test_start_date
].copy()


X_train = train_df[
    selected_features
]

X_test = test_df[
    selected_features
]

y_train = train_df[
    target
]

y_test = test_df[
    target
]


print(
    "Training rows:",
    len(train_df)
)


print(
    "Testing rows :",
    len(test_df)
)


print(
    "Training period:",
    train_df["Order Date"].min(),
    "to",
    train_df["Order Date"].max()
)


print(
    "Testing period :",
    test_df["Order Date"].min(),
    "to",
    test_df["Order Date"].max()
)


# Save split summary

split_summary = pd.DataFrame({

    "Dataset": [

        "Complete Dataset",
        "Training Dataset",
        "Testing Dataset"

    ],

    "Rows": [

        len(model_df),
        len(train_df),
        len(test_df)

    ],

    "Start Date": [

        model_df["Order Date"].min(),
        train_df["Order Date"].min(),
        test_df["Order Date"].min()

    ],

    "End Date": [

        model_df["Order Date"].max(),
        train_df["Order Date"].max(),
        test_df["Order Date"].max()

    ]

})


split_summary.to_excel(

    os.path.join(
        DATA_SPLIT_DIR,
        "Train_Test_Split_Summary.xlsx"
    ),

    index=False

)


# ============================================================
# 10. PREPROCESSING PIPELINE
# ============================================================

preprocessor = ColumnTransformer(

    transformers=[

        (
            "numeric",

            StandardScaler(),

            numeric_features

        ),

        (
            "categorical",

            OneHotEncoder(
                handle_unknown="ignore"
            ),

            categorical_features

        )

    ]

)


# ============================================================
# 11. REGRESSION METRICS
# ============================================================

def calculate_mape(
    actual,
    predicted
):

    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    mask = actual != 0

    if mask.sum() == 0:

        return np.nan

    return (

        np.mean(

            np.abs(

                (
                    actual[mask]
                    -
                    predicted[mask]
                )
                /
                actual[mask]

            )

        )
        * 100

    )


def calculate_smape(
    actual,
    predicted
):

    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    denominator = (

        np.abs(actual)
        +
        np.abs(predicted)

    )

    mask = denominator != 0

    if mask.sum() == 0:

        return np.nan

    return (

        np.mean(

            2
            *
            np.abs(
                actual[mask]
                -
                predicted[mask]
            )
            /
            denominator[mask]

        )
        * 100

    )


def calculate_metrics(
    actual,
    predicted
):

    mae = mean_absolute_error(
        actual,
        predicted
    )

    mse = mean_squared_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mse
    )

    r2 = r2_score(
        actual,
        predicted
    )

    mape = calculate_mape(
        actual,
        predicted
    )

    smape = calculate_smape(
        actual,
        predicted
    )

    return {

        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape,
        "SMAPE": smape

    }


# ============================================================
# 12. MODEL EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model_name,
    model,
    X_train,
    X_test,
    y_train,
    y_test
):

    print("\n" + "-" * 80)
    print(model_name)
    print("-" * 80)


    model.fit(
        X_train,
        y_train
    )


    predictions = model.predict(
        X_test
    )


    metrics = calculate_metrics(
        y_test,
        predictions
    )


    print(
        "MAE   :",
        round(metrics["MAE"], 4)
    )

    print(
        "MSE   :",
        round(metrics["MSE"], 4)
    )

    print(
        "RMSE  :",
        round(metrics["RMSE"], 4)
    )

    print(
        "R²    :",
        round(metrics["R2"], 4)
    )

    print(
        "MAPE  :",
        round(metrics["MAPE"], 4),
        "%"
    )

    print(
        "SMAPE :",
        round(metrics["SMAPE"], 4),
        "%"
    )


    return (
        model,
        predictions,
        metrics
    )


# ============================================================
# 13. SAVE PREDICTIONS FUNCTION
# ============================================================

def save_prediction_results(
    output_dir,
    filename,
    dates,
    actual,
    predicted
):

    result_df = pd.DataFrame({

        "Order Date":
            pd.to_datetime(
                dates
            ).values,

        "Actual Sales":
            np.asarray(actual),

        "Predicted Sales":
            np.asarray(predicted),

        "Error":
            (
                np.asarray(actual)
                -
                np.asarray(predicted)
            ),

        "Absolute Error":
            np.abs(
                np.asarray(actual)
                -
                np.asarray(predicted)
            )

    })


    result_df.to_excel(

        os.path.join(
            output_dir,
            filename
        ),

        index=False

    )


# ============================================================
# 14. LINEAR REGRESSION
# ============================================================

linear_model = Pipeline(

    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            LinearRegression()
        )

    ]

)


(
    linear_model,
    linear_predictions,
    linear_metrics
) = evaluate_model(

    "LINEAR REGRESSION - ML BASELINE",

    linear_model,

    X_train,
    X_test,

    y_train,
    y_test

)


save_prediction_results(

    LINEAR_DIR,

    "Linear_Regression_Predictions.xlsx",

    test_df["Order Date"],

    y_test,

    linear_predictions

)


# ============================================================
# 15. DECISION TREE
# ============================================================

tree_model = Pipeline(

    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",

            DecisionTreeRegressor(

                max_depth=12,

                random_state=42

            )

        )

    ]

)


(
    tree_model,
    tree_predictions,
    tree_metrics
) = evaluate_model(

    "DECISION TREE - ML BASELINE",

    tree_model,

    X_train,
    X_test,

    y_train,
    y_test

)


save_prediction_results(

    TREE_DIR,

    "Decision_Tree_Predictions.xlsx",

    test_df["Order Date"],

    y_test,

    tree_predictions

)


# ============================================================
# 16. RANDOM FOREST
# ============================================================

rf_model = Pipeline(

    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",

            RandomForestRegressor(

                n_estimators=150,

                max_depth=15,

                random_state=42,

                n_jobs=-1

            )

        )

    ]

)


(
    rf_model,
    rf_predictions,
    rf_metrics
) = evaluate_model(

    "RANDOM FOREST - ML BASELINE",

    rf_model,

    X_train,
    X_test,

    y_train,
    y_test

)


save_prediction_results(

    RF_DIR,

    "Random_Forest_Predictions.xlsx",

    test_df["Order Date"],

    y_test,

    rf_predictions

)


# ============================================================
# 17. XGBOOST ADVANCED ML BASELINE
# ============================================================

xgb_model = Pipeline(

    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",

            XGBRegressor(

                n_estimators=300,

                learning_rate=0.05,

                max_depth=6,

                subsample=0.8,

                colsample_bytree=0.8,

                objective="reg:squarederror",

                eval_metric="rmse",

                random_state=42,

                n_jobs=-1

            )

        )

    ]

)


(
    xgb_model,
    xgb_predictions,
    xgb_metrics
) = evaluate_model(

    "XGBOOST - ADVANCED ML BASELINE",

    xgb_model,

    X_train,
    X_test,

    y_train,
    y_test

)


save_prediction_results(

    XGB_DIR,

    "XGBoost_Predictions.xlsx",

    test_df["Order Date"],

    y_test,

    xgb_predictions

)


xgb_mae = xgb_metrics["MAE"]
xgb_mse = xgb_metrics["MSE"]
xgb_rmse = xgb_metrics["RMSE"]
xgb_r2 = xgb_metrics["R2"]
xgb_mape = xgb_metrics["MAPE"]
xgb_smape = xgb_metrics["SMAPE"]


joblib.dump(

    xgb_model,

    os.path.join(
        XGB_DIR,
        "XGBoost_Advanced_ML_Baseline.pkl"
    )

)


# ============================================================
# 18. LSTM BASELINE
# ============================================================

print("\n" + "=" * 80)
print("LSTM DEEP LEARNING BASELINE")
print("=" * 80)


# Log transform

log_monthly_values = np.log1p(
    monthly_values
)


lstm_scaler = StandardScaler()


lstm_scaler.fit(

    log_monthly_values[
        :monthly_split
    ].reshape(
        -1,
        1
    )

)


scaled_log_values = (
    lstm_scaler
    .transform(
        log_monthly_values.reshape(
            -1,
            1
        )
    )
    .flatten()
)


# ------------------------------------------------------------
# Sequence function
# ------------------------------------------------------------

def create_sequences(
    values,
    lookback
):

    X_seq = []
    y_seq = []

    for i in range(
        lookback,
        len(values)
    ):

        X_seq.append(
            values[
                i - lookback:i
            ]
        )

        y_seq.append(
            values[i]
        )

    return (
        np.array(X_seq),
        np.array(y_seq)
    )


X_lstm_train, y_lstm_train = create_sequences(

    scaled_log_values[
        :monthly_split
    ],

    LOOKBACK

)


test_start = (
    monthly_split - LOOKBACK
)


X_lstm_test, y_lstm_test = create_sequences(

    scaled_log_values[
        test_start:
    ],

    LOOKBACK

)


X_lstm_train = X_lstm_train.reshape(

    X_lstm_train.shape[0],

    X_lstm_train.shape[1],

    1

)


X_lstm_test = X_lstm_test.reshape(

    X_lstm_test.shape[0],

    X_lstm_test.shape[1],

    1

)


print(
    "LSTM training sequence:",
    X_lstm_train.shape
)


print(
    "LSTM testing sequence :",
    X_lstm_test.shape
)


# ============================================================
# 19. BUILD LSTM BASELINE
# ============================================================

lstm_model = Sequential([

    Input(
        shape=(
            LOOKBACK,
            1
        )
    ),

    LSTM(
        16
    ),

    Dropout(
        0.15
    ),

    Dense(
        8,
        activation="relu"
    ),

    Dense(
        1
    )

])


lstm_model.compile(

    optimizer="adam",

    loss=Huber()

)


lstm_early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=15,

    restore_best_weights=True

)


history = lstm_model.fit(

    X_lstm_train,

    y_lstm_train,

    epochs=120,

    batch_size=4,

    validation_split=0.20,

    shuffle=False,

    callbacks=[
        lstm_early_stopping
    ],

    verbose=0

)


print(
    "LSTM baseline training completed."
)


# ============================================================
# 20. LSTM PREDICTIONS
# ============================================================

lstm_scaled_predictions = (

    lstm_model.predict(
        X_lstm_test,
        verbose=0
    )
    .flatten()

)


lstm_log_predictions = (

    lstm_scaler
    .inverse_transform(
        lstm_scaled_predictions.reshape(
            -1,
            1
        )
    )
    .flatten()

)


lstm_predictions = np.expm1(
    lstm_log_predictions
)


lstm_predictions = np.maximum(
    lstm_predictions,
    0
)


lstm_actual = (
    monthly_values[
        monthly_split:
    ]
)


# ============================================================
# 21. LSTM METRICS
# ============================================================

lstm_metrics = calculate_metrics(

    lstm_actual,

    lstm_predictions

)


lstm_mae = lstm_metrics["MAE"]
lstm_mse = lstm_metrics["MSE"]
lstm_rmse = lstm_metrics["RMSE"]
lstm_r2 = lstm_metrics["R2"]
lstm_mape = lstm_metrics["MAPE"]
lstm_smape = lstm_metrics["SMAPE"]


print("\nLSTM RESULTS")


print(
    "MAE   :",
    round(lstm_mae, 4)
)


print(
    "MSE   :",
    round(lstm_mse, 4)
)


print(
    "RMSE  :",
    round(lstm_rmse, 4)
)


print(
    "R²    :",
    round(lstm_r2, 4)
)


print(
    "MAPE  :",
    round(lstm_mape, 4),
    "%"
)


print(
    "SMAPE :",
    round(lstm_smape, 4),
    "%"
)


# Save LSTM predictions

lstm_prediction_df = pd.DataFrame({

    "Date":
        pd.to_datetime(
            lstm_test_dates
        ),

    "Actual Sales":
        lstm_actual,

    "LSTM Predicted Sales":
        lstm_predictions,

    "Error":
        (
            lstm_actual
            -
            lstm_predictions
        ),

    "Absolute Error":
        np.abs(
            lstm_actual
            -
            lstm_predictions
        )

})


lstm_prediction_df.to_excel(

    os.path.join(
        LSTM_DIR,
        "LSTM_Predictions.xlsx"
    ),

    index=False

)


pd.DataFrame({

    "Metric": [

        "MAE",
        "MSE",
        "RMSE",
        "R2",
        "MAPE",
        "SMAPE"

    ],

    "Value": [

        lstm_mae,
        lstm_mse,
        lstm_rmse,
        lstm_r2,
        lstm_mape,
        lstm_smape

    ]

}).to_excel(

    os.path.join(
        LSTM_DIR,
        "LSTM_Metrics.xlsx"
    ),

    index=False

)


lstm_model.save(

    os.path.join(
        LSTM_DIR,
        "LSTM_Sales_Model.keras"
    )

)


pd.DataFrame(
    history.history
).to_excel(

    os.path.join(
        LSTM_DIR,
        "LSTM_Training_History.xlsx"
    ),

    index=False

)


# LSTM plot

plt.figure(
    figsize=(11, 6)
)


plt.plot(

    pd.to_datetime(
        lstm_test_dates
    ),

    lstm_actual,

    marker="o",

    label="Actual Sales"

)


plt.plot(

    pd.to_datetime(
        lstm_test_dates
    ),

    lstm_predictions,

    marker="o",

    label="LSTM Predicted Sales"

)


plt.xlabel(
    "Date"
)

plt.ylabel(
    "Monthly Sales"
)

plt.title(
    "LSTM Deep Learning Baseline"
)

plt.legend()

plt.xticks(
    rotation=30
)

plt.tight_layout()


plt.savefig(

    os.path.join(
        LSTM_DIR,
        "LSTM_Actual_vs_Predicted.png"
    ),

    dpi=300

)


plt.close()


# ============================================================
# 22. PROPOSED XGBOOST + LSTM FEATURE HYBRID
# ============================================================
#
# The LSTM is used to learn temporal monthly-sales behaviour.
#
# Its prediction becomes an additional feature for XGBoost.
#
# Therefore:
#
# Transaction Features
#          +
# LSTM Temporal Prediction
#          |
#          v
# Final XGBoost
#          |
#          v
# Hybrid Sales Prediction
#
# ============================================================

print("\n" + "=" * 80)
print("PROPOSED XGBOOST + LSTM FEATURE-LEVEL HYBRID")
print("=" * 80)


# ============================================================
# 23. GENERATE LSTM TEMPORAL FEATURES
# ============================================================

def generate_lstm_temporal_features(
    model,
    scaler,
    log_values,
    dates,
    lookback
):

    temporal_predictions = []
    temporal_dates = []

    scaled_values = (

        scaler
        .transform(
            log_values.reshape(
                -1,
                1
            )
        )
        .flatten()

    )


    for i in range(
        lookback,
        len(log_values)
    ):

        sequence = np.array(

            scaled_values[
                i - lookback:i
            ],

            dtype="float32"

        ).reshape(

            1,
            lookback,
            1

        )


        prediction_scaled = (

            model
            .predict(
                sequence,
                verbose=0
            )[0][0]

        )


        prediction_log = (

            scaler
            .inverse_transform(

                np.array(
                    [[
                        prediction_scaled
                    ]]
                )

            )[0][0]

        )


        prediction_sales = np.expm1(
            prediction_log
        )


        prediction_sales = max(
            0,
            prediction_sales
        )


        temporal_predictions.append(
            prediction_sales
        )


        temporal_dates.append(
            pd.to_datetime(
                dates[i]
            )
        )


    return pd.DataFrame({

        "Date":
            temporal_dates,

        "LSTM Temporal Prediction":
            temporal_predictions

    })


lstm_temporal_df = (

    generate_lstm_temporal_features(

        lstm_model,

        lstm_scaler,

        log_monthly_values,

        monthly_dates,

        LOOKBACK

    )

)


print(
    "\nGenerated LSTM temporal features:",
    len(lstm_temporal_df)
)


lstm_temporal_df.to_excel(

    os.path.join(
        HYBRID_DIR,
        "LSTM_Temporal_Features.xlsx"
    ),

    index=False

)


# ============================================================
# 24. CREATE TRANSACTION-LEVEL HYBRID DATA
# ============================================================

hybrid_model_df = model_df.copy()


hybrid_model_df["YearMonth"] = (

    hybrid_model_df[
        "Order Date"
    ]
    .dt
    .to_period("M")

)


lstm_temporal_lookup = (
    lstm_temporal_df.copy()
)


lstm_temporal_lookup["YearMonth"] = (

    lstm_temporal_lookup[
        "Date"
    ]
    .dt
    .to_period("M")

)


lstm_temporal_lookup = (

    lstm_temporal_lookup[
        [
            "YearMonth",
            "LSTM Temporal Prediction"
        ]
    ]

)


hybrid_model_df = hybrid_model_df.merge(

    lstm_temporal_lookup,

    on="YearMonth",

    how="left"

)


# ============================================================
# 25. HANDLE MISSING EARLY LSTM FEATURES
# ============================================================

early_lstm_fallback = (

    monthly_values[:LOOKBACK]
    .mean()

)


hybrid_model_df[
    "LSTM Temporal Prediction"
] = (

    hybrid_model_df[
        "LSTM Temporal Prediction"
    ]
    .fillna(
        early_lstm_fallback
    )

)


# ============================================================
# 26. HYBRID TRAIN-TEST SPLIT
# ============================================================

hybrid_train_df = hybrid_model_df[

    hybrid_model_df[
        "Order Date"
    ]
    <
    hybrid_test_start_date

].copy()


hybrid_test_df = hybrid_model_df[

    hybrid_model_df[
        "Order Date"
    ]
    >=
    hybrid_test_start_date

].copy()


print(
    "\nHybrid training rows:",
    len(hybrid_train_df)
)


print(
    "Hybrid testing rows :",
    len(hybrid_test_df)
)


# ============================================================
# 27. CREATE HYBRID FEATURES
# ============================================================

hybrid_features = (

    selected_features

    +

    [
        "LSTM Temporal Prediction"
    ]

)


X_hybrid_train = (

    hybrid_train_df[
        hybrid_features
    ]

)


X_hybrid_test = (

    hybrid_test_df[
        hybrid_features
    ]

)


y_hybrid_train = (

    hybrid_train_df[
        target
    ]

)


y_hybrid_test = (

    hybrid_test_df[
        target
    ]

)


# ============================================================
# 28. HYBRID PREPROCESSOR
# ============================================================

hybrid_numeric_features = (

    numeric_features

    +

    [
        "LSTM Temporal Prediction"
    ]

)


hybrid_categorical_features = (

    categorical_features

)


hybrid_preprocessor = ColumnTransformer(

    transformers=[

        (
            "numeric",

            StandardScaler(),

            hybrid_numeric_features

        ),

        (
            "categorical",

            OneHotEncoder(
                handle_unknown="ignore"
            ),

            hybrid_categorical_features

        )

    ]

)


# ============================================================
# 29. FINAL HYBRID XGBOOST MODEL
# ============================================================

hybrid_model = Pipeline(

    steps=[

        (
            "preprocessor",

            hybrid_preprocessor

        ),

        (
            "model",

            XGBRegressor(

                n_estimators=350,

                learning_rate=0.04,

                max_depth=6,

                min_child_weight=2,

                subsample=0.85,

                colsample_bytree=0.85,

                objective="reg:squarederror",

                eval_metric="rmse",

                random_state=42,

                n_jobs=-1

            )

        )

    ]

)


print(
    "\nTraining final XGBoost + LSTM hybrid..."
)


hybrid_model.fit(

    X_hybrid_train,

    y_hybrid_train

)


print(
    "Final hybrid training completed."
)


# ============================================================
# 30. HYBRID PREDICTIONS
# ============================================================

hybrid_predictions = (

    hybrid_model
    .predict(
        X_hybrid_test
    )

)


hybrid_predictions = np.maximum(

    hybrid_predictions,

    0

)


# ============================================================
# 31. HYBRID METRICS
# ============================================================

hybrid_metrics = calculate_metrics(

    y_hybrid_test,

    hybrid_predictions

)


hybrid_mae = hybrid_metrics["MAE"]
hybrid_mse = hybrid_metrics["MSE"]
hybrid_rmse = hybrid_metrics["RMSE"]
hybrid_r2 = hybrid_metrics["R2"]
hybrid_mape = hybrid_metrics["MAPE"]
hybrid_smape = hybrid_metrics["SMAPE"]


print("\n" + "-" * 80)

print(
    "PROPOSED XGBOOST + LSTM FEATURE HYBRID RESULTS"
)

print("-" * 80)


print(
    "MAE   :",
    round(
        hybrid_mae,
        4
    )
)


print(
    "MSE   :",
    round(
        hybrid_mse,
        4
    )
)


print(
    "RMSE  :",
    round(
        hybrid_rmse,
        4
    )
)


print(
    "R²    :",
    round(
        hybrid_r2,
        4
    )
)


print(
    "MAPE  :",
    round(
        hybrid_mape,
        4
    ),
    "%"
)


print(
    "SMAPE :",
    round(
        hybrid_smape,
        4
    ),
    "%"
)


# ============================================================
# 32. SAVE HYBRID PREDICTIONS
# ============================================================

hybrid_prediction_df = hybrid_test_df[

    [
        "Order Date",
        "Sales",
        "LSTM Temporal Prediction"
    ]

].copy()


hybrid_prediction_df[
    "XGBoost Prediction"
] = (

    xgb_model.predict(
        X_test
    )

)


hybrid_prediction_df[
    "Hybrid Prediction"
] = hybrid_predictions


hybrid_prediction_df[
    "Hybrid Error"
] = (

    hybrid_prediction_df[
        "Sales"
    ]

    -

    hybrid_prediction_df[
        "Hybrid Prediction"
    ]

)


hybrid_prediction_df[
    "Absolute Error"
] = np.abs(

    hybrid_prediction_df[
        "Hybrid Error"
    ]

)


hybrid_prediction_df.to_excel(

    os.path.join(
        HYBRID_DIR,
        "Hybrid_Predictions.xlsx"
    ),

    index=False

)


# ============================================================
# 33. SAVE HYBRID METRICS
# ============================================================

pd.DataFrame({

    "Metric": [

        "MAE",
        "MSE",
        "RMSE",
        "R2",
        "MAPE",
        "SMAPE"

    ],

    "Value": [

        hybrid_mae,
        hybrid_mse,
        hybrid_rmse,
        hybrid_r2,
        hybrid_mape,
        hybrid_smape

    ]

}).to_excel(

    os.path.join(
        HYBRID_DIR,
        "Hybrid_Metrics.xlsx"
    ),

    index=False

)


# ============================================================
# 34. HYBRID COMPONENT INFORMATION
# ============================================================

hybrid_component_info = pd.DataFrame({

    "Component": [

        "LSTM",
        "XGBoost",
        "Final Hybrid Model"

    ],

    "Role": [

        "Learn monthly temporal sales patterns",

        "Learn transaction-level sales relationships",

        "Combine transaction features with LSTM temporal feature"

    ],

    "Input": [

        "Historical monthly Sales",

        "Transaction-level features",

        "Transaction features + LSTM Temporal Prediction"

    ],

    "Output": [

        "LSTM Temporal Prediction",

        "Transaction-level Sales Prediction",

        "Final Hybrid Sales Prediction"

    ]

})


hybrid_component_info.to_excel(

    os.path.join(
        HYBRID_DIR,
        "Hybrid_Component_Architecture.xlsx"
    ),

    index=False

)


# ============================================================
# 35. HYBRID MODEL SUMMARY
# ============================================================

hybrid_model_summary = pd.DataFrame({

    "Item": [

        "Proposed Model",

        "Model Type",

        "Architecture",

        "Machine Learning Component",

        "Deep Learning Component",

        "Hybrid Feature",

        "Lookback Months",

        "Training Rows",

        "Testing Rows",

        "Testing Start Date",

        "Testing End Date",

        "MAE",

        "MSE",

        "RMSE",

        "R2",

        "MAPE",

        "SMAPE"

    ],

    "Value": [

        "XGBoost + LSTM Feature Hybrid",

        "Hybrid ML + DL",

        "LSTM Temporal Feature + XGBoost",

        "XGBoost",

        "LSTM",

        "LSTM Temporal Prediction",

        LOOKBACK,

        len(
            X_hybrid_train
        ),

        len(
            X_hybrid_test
        ),

        hybrid_test_df[
            "Order Date"
        ].min(),

        hybrid_test_df[
            "Order Date"
        ].max(),

        hybrid_mae,

        hybrid_mse,

        hybrid_rmse,

        hybrid_r2,

        hybrid_mape,

        hybrid_smape

    ]

})


hybrid_model_summary.to_excel(

    os.path.join(
        HYBRID_DIR,
        "Hybrid_Model_Summary.xlsx"
    ),

    index=False

)


# ============================================================
# 36. SAVE FINAL HYBRID MODEL
# ============================================================

joblib.dump(

    hybrid_model,

    os.path.join(
        HYBRID_DIR,
        "XGBoost_LSTM_Hybrid_Model.pkl"
    )

)


joblib.dump(

    {

        "model_name":
            "XGBoost + LSTM Feature Hybrid",

        "model_type":
            "Hybrid ML + DL",

        "architecture":
            "LSTM Temporal Feature + XGBoost",

        "lookback":
            LOOKBACK,

        "lstm_model_path":
            os.path.join(
                LSTM_DIR,
                "LSTM_Sales_Model.keras"
            ),

        "hybrid_model_path":
            os.path.join(
                HYBRID_DIR,
                "XGBoost_LSTM_Hybrid_Model.pkl"
            )

    },

    os.path.join(
        FINAL_DIR,
        "Final_Hybrid_Model_Metadata.pkl"
    )

)


# ============================================================
# 37. HYBRID MONTHLY COMPARISON
# ============================================================

hybrid_plot_df = hybrid_prediction_df.copy()


hybrid_plot_df[
    "YearMonth"
] = (

    pd.to_datetime(
        hybrid_plot_df[
            "Order Date"
        ]
    )
    .dt
    .to_period("M")

)


hybrid_monthly_plot = (

    hybrid_plot_df

    .groupby(
        "YearMonth"
    )

    .agg(

        Actual_Sales=(
            "Sales",
            "sum"
        ),

        XGBoost_Sales=(
            "XGBoost Prediction",
            "sum"
        ),

        Hybrid_Sales=(
            "Hybrid Prediction",
            "sum"
        ),

        LSTM_Temporal_Sales=(
            "LSTM Temporal Prediction",
            "mean"
        )

    )

    .reset_index()

)


hybrid_monthly_plot[
    "Date"
] = (

    hybrid_monthly_plot[
        "YearMonth"
    ]
    .dt
    .to_timestamp()

)


hybrid_monthly_plot.to_excel(

    os.path.join(
        HYBRID_DIR,
        "Hybrid_Monthly_Predictions.xlsx"
    ),

    index=False

)


# ============================================================
# 38. HYBRID PLOT
# ============================================================

plt.figure(
    figsize=(12, 6)
)


plt.plot(

    hybrid_monthly_plot[
        "Date"
    ],

    hybrid_monthly_plot[
        "Actual_Sales"
    ],

    marker="o",

    label="Actual Sales"

)


plt.plot(

    hybrid_monthly_plot[
        "Date"
    ],

    hybrid_monthly_plot[
        "XGBoost_Sales"
    ],

    marker="o",

    label="XGBoost"

)


plt.plot(

    hybrid_monthly_plot[
        "Date"
    ],

    hybrid_monthly_plot[
        "Hybrid_Sales"
    ],

    marker="o",

    label="XGBoost + LSTM Hybrid"

)


plt.xlabel(
    "Date"
)

plt.ylabel(
    "Monthly Sales"
)

plt.title(
    "Proposed XGBoost + LSTM Feature Hybrid"
)

plt.legend()

plt.xticks(
    rotation=30
)

plt.tight_layout()


plt.savefig(

    os.path.join(
        HYBRID_DIR,
        "Hybrid_Actual_vs_Predicted.png"
    ),

    dpi=300

)


plt.close()


# ============================================================
# 39. HYBRID VS XGBOOST IMPROVEMENT
# ============================================================

def percentage_reduction(
    baseline,
    proposed
):

    if baseline == 0:

        return np.nan

    return (

        (
            baseline
            -
            proposed
        )
        /
        baseline
        *
        100

    )


hybrid_improvement = pd.DataFrame({

    "Metric": [

        "MAE",
        "MSE",
        "RMSE",
        "MAPE",
        "SMAPE",
        "R2"

    ],

    "XGBoost Baseline": [

        xgb_mae,
        xgb_mse,
        xgb_rmse,
        xgb_mape,
        xgb_smape,
        xgb_r2

    ],

    "Proposed Hybrid": [

        hybrid_mae,
        hybrid_mse,
        hybrid_rmse,
        hybrid_mape,
        hybrid_smape,
        hybrid_r2

    ],

    "Improvement_Percent": [

        percentage_reduction(
            xgb_mae,
            hybrid_mae
        ),

        percentage_reduction(
            xgb_mse,
            hybrid_mse
        ),

        percentage_reduction(
            xgb_rmse,
            hybrid_rmse
        ),

        percentage_reduction(
            xgb_mape,
            hybrid_mape
        ),

        percentage_reduction(
            xgb_smape,
            hybrid_smape
        ),

        (
            hybrid_r2
            -
            xgb_r2
        )
        /
        max(
            abs(xgb_r2),
            1e-9
        )
        *
        100

    ]

})


hybrid_improvement.to_excel(

    os.path.join(
        HYBRID_DIR,
        "Hybrid_vs_XGBoost_Improvement.xlsx"
    ),

    index=False

)


# ============================================================
# 40. ADDITIONAL REVIEW-3 HYBRID EXPERIMENTS
# ============================================================
#
# Review-3 experiment matrix:
#   Temporal learner: LSTM / GRU
#   Tree learner    : XGBoost / Random Forest
#
# Existing experiment:
#   LSTM + XGBoost
#
# Added experiments:
#   LSTM + Random Forest
#   GRU + XGBoost
#   GRU + Random Forest
#
# The same chronological split and regression metrics are used.
# ============================================================

print("\n" + "=" * 80)
print("REVIEW-3: GRU BASELINE + MULTIPLE HYBRID EXPERIMENTS")
print("=" * 80)


# ------------------------------------------------------------
# 40A. GRU BASELINE
# ------------------------------------------------------------

print("\n[Review-3 1/4] Training GRU baseline...")

log_monthly_values_gru = np.log1p(monthly_values)

gru_scaler = StandardScaler()
gru_scaler.fit(
    log_monthly_values_gru[:monthly_split].reshape(-1, 1)
)

scaled_log_values_gru = (
    gru_scaler
    .transform(log_monthly_values_gru.reshape(-1, 1))
    .flatten()
)

X_gru_train, y_gru_train = create_sequences(
    scaled_log_values_gru[:monthly_split], LOOKBACK
)

gru_test_start = monthly_split - LOOKBACK
X_gru_test, y_gru_test = create_sequences(
    scaled_log_values_gru[gru_test_start:], LOOKBACK
)

X_gru_train = X_gru_train.reshape(X_gru_train.shape[0], X_gru_train.shape[1], 1)
X_gru_test = X_gru_test.reshape(X_gru_test.shape[0], X_gru_test.shape[1], 1)

gru_model = Sequential([
    Input(shape=(LOOKBACK, 1)),
    GRU(16),
    Dropout(0.15),
    Dense(8, activation="relu"),
    Dense(1)
])

gru_model.compile(
    optimizer="adam",
    loss=Huber()
)

gru_early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=15,
    restore_best_weights=True
)

gru_history = gru_model.fit(
    X_gru_train,
    y_gru_train,
    epochs=120,
    batch_size=4,
    validation_split=0.20,
    shuffle=False,
    callbacks=[gru_early_stopping],
    verbose=0
)

gru_scaled_predictions = gru_model.predict(X_gru_test, verbose=0).flatten()
gru_log_predictions = gru_scaler.inverse_transform(
    gru_scaled_predictions.reshape(-1, 1)
).flatten()
gru_predictions = np.maximum(np.expm1(gru_log_predictions), 0)
gru_actual = monthly_values[monthly_split:]
gru_metrics = calculate_metrics(gru_actual, gru_predictions)

gru_mae = gru_metrics["MAE"]
gru_mse = gru_metrics["MSE"]
gru_rmse = gru_metrics["RMSE"]
gru_r2 = gru_metrics["R2"]
gru_mape = gru_metrics["MAPE"]
gru_smape = gru_metrics["SMAPE"]

pd.DataFrame([gru_metrics]).to_excel(
    os.path.join(GRU_DIR, "GRU_Metrics.xlsx"), index=False
)

pd.DataFrame({
    "Date": pd.to_datetime(lstm_test_dates),
    "Actual Sales": gru_actual,
    "GRU Prediction": gru_predictions
}).to_excel(
    os.path.join(GRU_DIR, "GRU_Predictions.xlsx"), index=False
)

gru_model.save(os.path.join(GRU_DIR, "GRU_Model.keras"))
joblib.dump(gru_scaler, os.path.join(GRU_DIR, "GRU_Scaler.joblib"))

print("GRU baseline completed.")
print(gru_metrics)


# ------------------------------------------------------------
# 40B. GENERATE GRU TEMPORAL FEATURES
# ------------------------------------------------------------

def generate_gru_temporal_features(
    model,
    scaler,
    log_values,
    dates,
    lookback
):
    temporal_predictions = []
    temporal_dates = []

    scaled_values = scaler.transform(
        log_values.reshape(-1, 1)
    ).flatten()

    for i in range(lookback, len(log_values)):
        sequence = np.array(
            scaled_values[i - lookback:i],
            dtype="float32"
        ).reshape(1, lookback, 1)

        prediction_scaled = model.predict(
            sequence, verbose=0
        )[0][0]

        prediction_log = scaler.inverse_transform(
            np.array([[prediction_scaled]])
        )[0][0]

        prediction_sales = max(0, np.expm1(prediction_log))

        temporal_predictions.append(prediction_sales)
        temporal_dates.append(pd.to_datetime(dates[i]))

    return pd.DataFrame({
        "Date": temporal_dates,
        "GRU Temporal Prediction": temporal_predictions
    })


gru_temporal_df = generate_gru_temporal_features(
    gru_model,
    gru_scaler,
    log_monthly_values_gru,
    monthly_dates,
    LOOKBACK
)

gru_temporal_df.to_excel(
    os.path.join(GRU_DIR, "GRU_Temporal_Features.xlsx"),
    index=False
)


# ------------------------------------------------------------
# 40C. GENERIC TRANSACTION-LEVEL HYBRID RUNNER
# ------------------------------------------------------------

def run_tree_hybrid(
    hybrid_name,
    temporal_df,
    temporal_column,
    tree_model,
    output_dir,
    model_type_label
):
    print("\n" + "-" * 80)
    print(f"Training {hybrid_name}...")
    print("-" * 80)

    hybrid_df = model_df.copy()
    hybrid_df["YearMonth"] = hybrid_df["Order Date"].dt.to_period("M")

    lookup = temporal_df.copy()
    lookup["YearMonth"] = lookup["Date"].dt.to_period("M")
    lookup = lookup[["YearMonth", temporal_column]]

    hybrid_df = hybrid_df.merge(
        lookup,
        on="YearMonth",
        how="left"
    )

    fallback = monthly_values[:LOOKBACK].mean()
    hybrid_df[temporal_column] = hybrid_df[temporal_column].fillna(fallback)

    train_part = hybrid_df[
        hybrid_df["Order Date"] < hybrid_test_start_date
    ].copy()
    test_part = hybrid_df[
        hybrid_df["Order Date"] >= hybrid_test_start_date
    ].copy()

    hybrid_features_local = selected_features + [temporal_column]
    numeric_local = numeric_features + [temporal_column]

    local_preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_local),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", local_preprocessor),
            ("model", tree_model)
        ]
    )

    pipeline.fit(
        train_part[hybrid_features_local],
        train_part[target]
    )

    predictions = pipeline.predict(
        test_part[hybrid_features_local]
    )

    metrics = calculate_metrics(
        test_part[target], predictions
    )

    metrics_df = pd.DataFrame([metrics])
    metrics_df.insert(0, "Model", hybrid_name)
    metrics_df.insert(1, "Model Type", model_type_label)
    metrics_df.insert(2, "Training Rows", len(train_part))
    metrics_df.insert(3, "Testing Rows", len(test_part))
    metrics_df.insert(
        4,
        "Test Period",
        f"{test_part['Order Date'].min().date()} to {test_part['Order Date'].max().date()}"
    )

    metrics_df.to_excel(
        os.path.join(output_dir, f"{hybrid_name.replace(' + ', '_')}_Metrics.xlsx"),
        index=False
    )

    prediction_df = pd.DataFrame({
        "Order Date": test_part["Order Date"].values,
        "Actual Sales": test_part[target].values,
        "Predicted Sales": predictions
    })
    prediction_df.to_excel(
        os.path.join(output_dir, f"{hybrid_name.replace(' + ', '_')}_Predictions.xlsx"),
        index=False
    )

    joblib.dump(
        pipeline,
        os.path.join(output_dir, f"{hybrid_name.replace(' + ', '_')}_Pipeline.joblib")
    )

    print(f"{hybrid_name} completed.")
    print(metrics)

    return metrics


# ------------------------------------------------------------
# 40D. LSTM + RANDOM FOREST
# ------------------------------------------------------------

lstm_rf_metrics = run_tree_hybrid(
    "LSTM + Random Forest",
    lstm_temporal_df,
    "LSTM Temporal Prediction",
    RandomForestRegressor(
        n_estimators=150,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    ),
    HYBRID_LSTM_RF_DIR,
    "Hybrid ML + DL"
)


# ------------------------------------------------------------
# 40E. GRU + XGBOOST
# ------------------------------------------------------------

gru_xgb_metrics = run_tree_hybrid(
    "GRU + XGBoost",
    gru_temporal_df,
    "GRU Temporal Prediction",
    XGBRegressor(
        n_estimators=350,
        learning_rate=0.04,
        max_depth=6,
        min_child_weight=2,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=42,
        n_jobs=-1
    ),
    HYBRID_GRU_XGB_DIR,
    "Hybrid ML + DL"
)


# ------------------------------------------------------------
# 40F. GRU + RANDOM FOREST
# ------------------------------------------------------------

gru_rf_metrics = run_tree_hybrid(
    "GRU + Random Forest",
    gru_temporal_df,
    "GRU Temporal Prediction",
    RandomForestRegressor(
        n_estimators=150,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    ),
    HYBRID_GRU_RF_DIR,
    "Hybrid ML + DL"
)


# ============================================================
# 41. COMPLETE MODEL COMPARISON
# ============================================================

print("\n" + "=" * 80)
print("COMPLETE MODEL COMPARISON")
print("=" * 80)


comparison = pd.DataFrame({

    "Model": [

        "Linear Regression",
        "Decision Tree",
        "Random Forest",
        "XGBoost",
        "LSTM",
        "GRU",
        "XGBoost + LSTM Feature Hybrid",
        "LSTM + Random Forest",
        "GRU + XGBoost",
        "GRU + Random Forest"

    ],

    "Model Type": [

        "ML Baseline",
        "ML Baseline",
        "ML Baseline",
        "Advanced ML Baseline",
        "DL Baseline",
        "DL Baseline",
        "Hybrid ML + DL",
        "Hybrid ML + DL",
        "Hybrid ML + DL",
        "Hybrid ML + DL"

    ],

    "Evaluation Level": [

        "Transaction",
        "Transaction",
        "Transaction",
        "Transaction",
        "Monthly",
        "Monthly",
        "Transaction",
        "Transaction",
        "Transaction",
        "Transaction"

    ],

       "Test Period": [

        f"{test_df['Order Date'].min().date()} to "
        f"{test_df['Order Date'].max().date()}",

        f"{test_df['Order Date'].min().date()} to "
        f"{test_df['Order Date'].max().date()}",

        f"{test_df['Order Date'].min().date()} to "
        f"{test_df['Order Date'].max().date()}",

        f"{test_df['Order Date'].min().date()} to "
        f"{test_df['Order Date'].max().date()}",

        f"{pd.to_datetime(lstm_test_dates).min().date()} to "
        f"{pd.to_datetime(lstm_test_dates).max().date()}",

        f"{pd.to_datetime(lstm_test_dates).min().date()} to "
        f"{pd.to_datetime(lstm_test_dates).max().date()}",

        f"{test_df['Order Date'].min().date()} to "
        f"{test_df['Order Date'].max().date()}",

        f"{test_df['Order Date'].min().date()} to "
        f"{test_df['Order Date'].max().date()}",

        f"{test_df['Order Date'].min().date()} to "
        f"{test_df['Order Date'].max().date()}",

        f"{test_df['Order Date'].min().date()} to "
        f"{test_df['Order Date'].max().date()}"

    ],

    "MAE": [

        linear_metrics["MAE"],
        tree_metrics["MAE"],
        rf_metrics["MAE"],
        xgb_metrics["MAE"],
        lstm_mae,
        gru_mae,
        hybrid_mae,
        lstm_rf_metrics["MAE"],
        gru_xgb_metrics["MAE"],
        gru_rf_metrics["MAE"]

    ],

    "MSE": [

        linear_metrics["MSE"],
        tree_metrics["MSE"],
        rf_metrics["MSE"],
        xgb_metrics["MSE"],
        lstm_mse,
        gru_mse,
        hybrid_mse,
        lstm_rf_metrics["MSE"],
        gru_xgb_metrics["MSE"],
        gru_rf_metrics["MSE"]

    ],

    "RMSE": [

        linear_metrics["RMSE"],
        tree_metrics["RMSE"],
        rf_metrics["RMSE"],
        xgb_metrics["RMSE"],
        lstm_rmse,
        gru_rmse,
        hybrid_rmse,
        lstm_rf_metrics["RMSE"],
        gru_xgb_metrics["RMSE"],
        gru_rf_metrics["RMSE"]

    ],

    "R2": [

        linear_metrics["R2"],
        tree_metrics["R2"],
        rf_metrics["R2"],
        xgb_metrics["R2"],
        lstm_r2,
        gru_r2,
        hybrid_r2,
        lstm_rf_metrics["R2"],
        gru_xgb_metrics["R2"],
        gru_rf_metrics["R2"]

    ],

    "MAPE": [

        linear_metrics["MAPE"],
        tree_metrics["MAPE"],
        rf_metrics["MAPE"],
        xgb_metrics["MAPE"],
        lstm_mape,
        gru_mape,
        hybrid_mape,
        lstm_rf_metrics["MAPE"],
        gru_xgb_metrics["MAPE"],
        gru_rf_metrics["MAPE"]

    ],

    "SMAPE": [

        linear_metrics["SMAPE"],
        tree_metrics["SMAPE"],
        rf_metrics["SMAPE"],
        xgb_metrics["SMAPE"],
        lstm_smape,
        gru_smape,
        hybrid_smape,
        lstm_rf_metrics["SMAPE"],
        gru_xgb_metrics["SMAPE"],
        gru_rf_metrics["SMAPE"]

    ]

})


metric_columns = [

    "MAE",
    "MSE",
    "RMSE",
    "R2",
    "MAPE",
    "SMAPE"

]


for col in metric_columns:

    comparison[col] = (
        comparison[col]
        .round(4)
    )


print(
    comparison.to_string(
        index=False
    )
)


comparison.to_excel(

    os.path.join(
        COMPARISON_DIR,
        "Model_Comparison_All_Metrics.xlsx"
    ),

    index=False

)


comparison.to_excel(

    os.path.join(
        COMPARISON_DIR,
        "Model_Comparison.xlsx"
    ),

    index=False

)


# ============================================================
# 42. MODEL COMPARISON PLOTS
# ============================================================

# ------------------------------------------------------------
# R2
# ------------------------------------------------------------

plt.figure(
    figsize=(11, 6)
)


plt.bar(

    comparison["Model"],

    comparison["R2"]

)


plt.ylabel(
    "R² Score"
)

plt.xlabel(
    "Model"
)

plt.title(
    "Model Comparison - R² Score"
)

plt.xticks(
    rotation=25,
    ha="right"
)

plt.tight_layout()


plt.savefig(

    os.path.join(
        COMPARISON_DIR,
        "Model_Comparison_R2.png"
    ),

    dpi=300

)


plt.close()


# ------------------------------------------------------------
# MAE
# ------------------------------------------------------------

plt.figure(
    figsize=(11, 6)
)


plt.bar(

    comparison["Model"],

    comparison["MAE"]

)


plt.ylabel(
    "MAE"
)

plt.xlabel(
    "Model"
)

plt.title(
    "Model Comparison - MAE"
)

plt.xticks(
    rotation=25,
    ha="right"
)

plt.tight_layout()


plt.savefig(

    os.path.join(
        COMPARISON_DIR,
        "Model_Comparison_MAE.png"
    ),

    dpi=300

)


plt.close()


# ------------------------------------------------------------
# MSE
# ------------------------------------------------------------

plt.figure(
    figsize=(11, 6)
)


plt.bar(

    comparison["Model"],

    comparison["MSE"]

)


plt.ylabel(
    "MSE"
)

plt.xlabel(
    "Model"
)

plt.title(
    "Model Comparison - MSE"
)

plt.xticks(
    rotation=25,
    ha="right"
)

plt.tight_layout()


plt.savefig(

    os.path.join(
        COMPARISON_DIR,
        "Model_Comparison_MSE.png"
    ),

    dpi=300

)


plt.close()


# ------------------------------------------------------------
# RMSE
# ------------------------------------------------------------

plt.figure(
    figsize=(11, 6)
)


plt.bar(

    comparison["Model"],

    comparison["RMSE"]

)


plt.ylabel(
    "RMSE"
)

plt.xlabel(
    "Model"
)

plt.title(
    "Model Comparison - RMSE"
)

plt.xticks(
    rotation=25,
    ha="right"
)

plt.tight_layout()


plt.savefig(

    os.path.join(
        COMPARISON_DIR,
        "Model_Comparison_RMSE.png"
    ),

    dpi=300

)


plt.close()


# ------------------------------------------------------------
# MAPE
# ------------------------------------------------------------

plt.figure(
    figsize=(11, 6)
)


plt.bar(

    comparison["Model"],

    comparison["MAPE"]

)


plt.ylabel(
    "MAPE (%)"
)

plt.xlabel(
    "Model"
)

plt.title(
    "Model Comparison - MAPE"
)

plt.xticks(
    rotation=25,
    ha="right"
)

plt.tight_layout()


plt.savefig(

    os.path.join(
        COMPARISON_DIR,
        "Model_Comparison_MAPE.png"
    ),

    dpi=300

)


plt.close()


# ------------------------------------------------------------
# SMAPE
# ------------------------------------------------------------

plt.figure(
    figsize=(11, 6)
)


plt.bar(

    comparison["Model"],

    comparison["SMAPE"]

)


plt.ylabel(
    "SMAPE (%)"
)

plt.xlabel(
    "Model"
)

plt.title(
    "Model Comparison - SMAPE"
)

plt.xticks(
    rotation=25,
    ha="right"
)

plt.tight_layout()


plt.savefig(

    os.path.join(
        COMPARISON_DIR,
        "Model_Comparison_SMAPE.png"
    ),

    dpi=300

)


plt.close()


# ============================================================
# 42. XGBOOST MODEL SUMMARY
# ============================================================

xgb_model_info = pd.DataFrame({

    "Item": [

        "Model",
        "Model Type",
        "Target Variable",
        "Training Rows",
        "Testing Rows",
        "Test Start Date",
        "Test End Date",
        "MAE",
        "MSE",
        "RMSE",
        "R2",
        "MAPE",
        "SMAPE"

    ],

    "Value": [

        "XGBoost",

        "Advanced ML Baseline",

        "Sales",

        len(X_train),

        len(X_test),

        test_df["Order Date"].min(),

        test_df["Order Date"].max(),

        xgb_mae,

        xgb_mse,

        xgb_rmse,

        xgb_r2,

        xgb_mape,

        xgb_smape

    ]

})


xgb_model_info.to_excel(

    os.path.join(
        XGB_DIR,
        "XGBoost_Model_Summary.xlsx"
    ),

    index=False

)


# ============================================================
# 43. FINAL MODEL SUMMARY
# ============================================================

final_model_info = pd.DataFrame({

    "Item": [

        "Proposed Model",

        "Model Type",

        "Architecture",

        "Components",

        "Target Variable",

        "LSTM Lookback",

        "Hybrid Feature",

        "MAE",

        "MSE",

        "RMSE",

        "R2",

        "MAPE",

        "SMAPE"

    ],

    "Value": [

        "XGBoost + LSTM Feature Hybrid",

        "Hybrid ML + DL",

        "LSTM Temporal Feature + XGBoost",

        "XGBoost + LSTM",

        "Sales",

        LOOKBACK,

        "LSTM Temporal Prediction",

        hybrid_mae,

        hybrid_mse,

        hybrid_rmse,

        hybrid_r2,

        hybrid_mape,

        hybrid_smape

    ]

})


final_model_info.to_excel(

    os.path.join(
        FINAL_DIR,
        "Final_Model_Summary.xlsx"
    ),

    index=False

)


# ============================================================
# 44. K-MEANS CUSTOMER SEGMENTATION
# ============================================================

print("\n" + "=" * 80)
print("K-MEANS CUSTOMER SEGMENTATION")
print("=" * 80)


if (

    "Customer ID" in df.columns

    and

    "Customer Order Count" in df.columns

    and

    "Customer Total Sales" in df.columns

    and

    "Customer Total Profit" in df.columns

):


    customer_df = df[

        [
            "Customer ID",
            "Customer Order Count",
            "Customer Total Sales",
            "Customer Total Profit"
        ]

    ].drop_duplicates(

        subset=[
            "Customer ID"
        ]

    ).copy()


else:

    print(
        "Engineered customer features not found."
    )

    print(
        "Creating customer-level features..."
    )


    customer_df = df.groupby(

        "Customer ID"

    ).agg(

        Customer_Order_Count=(

            "Order ID",
            "count"

        ),

        Customer_Total_Sales=(

            "Sales",
            "sum"

        ),

        Customer_Total_Profit=(

            "Profit",
            "sum"

        )

    ).reset_index()


    customer_df.rename(

        columns={

            "Customer_Order_Count":
                "Customer Order Count",

            "Customer_Total_Sales":
                "Customer Total Sales",

            "Customer_Total_Profit":
                "Customer Total Profit"

        },

        inplace=True

    )


cluster_features = [

    "Customer Order Count",
    "Customer Total Sales",
    "Customer Total Profit"

]


cluster_data = customer_df[
    cluster_features
].copy()


cluster_data = cluster_data.fillna(
    cluster_data.median()
)


cluster_scaler = StandardScaler()


cluster_scaled = cluster_scaler.fit_transform(
    cluster_data
)


# ============================================================
# 45. FIND BEST NUMBER OF CLUSTERS
# ============================================================

silhouette_results = []


for k in range(
    2,
    7
):

    kmeans_temp = KMeans(

        n_clusters=k,

        random_state=42,

        n_init=10

    )


    labels_temp = (
        kmeans_temp.fit_predict(
            cluster_scaled
        )
    )


    score = silhouette_score(

        cluster_scaled,

        labels_temp

    )


    silhouette_results.append({

        "Number of Clusters":
            k,

        "Silhouette Score":
            score

    })


silhouette_df = pd.DataFrame(
    silhouette_results
)


print(
    "\nSilhouette Scores:"
)


print(
    silhouette_df.to_string(
        index=False
    )
)


silhouette_df.to_excel(

    os.path.join(
        CLUSTER_DIR,
        "Silhouette_Scores.xlsx"
    ),

    index=False

)


best_k = int(

    silhouette_df.loc[

        silhouette_df[
            "Silhouette Score"
        ].idxmax(),

        "Number of Clusters"

    ]

)


print(
    "\nSelected clusters:",
    best_k
)


# ============================================================
# 46. FINAL K-MEANS
# ============================================================

kmeans = KMeans(

    n_clusters=best_k,

    random_state=42,

    n_init=10

)


customer_df["Cluster"] = (

    kmeans.fit_predict(
        cluster_scaled
    )

)


cluster_summary = (

    customer_df

    .groupby(
        "Cluster"
    )

    .agg(

        Customers=(

            "Customer ID",
            "count"

        ),

        Average_Order_Count=(

            "Customer Order Count",
            "mean"

        ),

        Average_Sales=(

            "Customer Total Sales",
            "mean"

        ),

        Average_Profit=(

            "Customer Total Profit",
            "mean"

        )

    )

    .reset_index()

)


# ============================================================
# 47. CUSTOMER SEGMENT NAMES
# ============================================================

cluster_summary["Value Score"] = (

    cluster_summary[
        "Average_Sales"
    ].rank(
        method="first"
    )

    +

    cluster_summary[
        "Average_Profit"
    ].rank(
        method="first"
    )

)


cluster_summary = (

    cluster_summary

    .sort_values(
        "Value Score"
    )

    .reset_index(
        drop=True
    )

)


ordered_clusters = (

    cluster_summary[
        "Cluster"
    ]
    .tolist()

)


segment_names = {}


if best_k == 3:

    segment_names = {

        ordered_clusters[0]:
            "Low Value",

        ordered_clusters[1]:
            "Medium Value",

        ordered_clusters[2]:
            "High Value"

    }


else:

    for i, cluster in enumerate(
        ordered_clusters
    ):

        if i == 0:

            segment_names[
                cluster
            ] = "Low Value"

        elif i == len(
            ordered_clusters
        ) - 1:

            segment_names[
                cluster
            ] = "High Value"

        else:

            segment_names[
                cluster
            ] = (
                f"Medium Value {i}"
            )


customer_df[
    "Customer Segment"
] = (

    customer_df[
        "Cluster"
    ]
    .map(
        segment_names
    )

)


cluster_summary[
    "Segment"
] = (

    cluster_summary[
        "Cluster"
    ]
    .map(
        segment_names
    )

)


# ============================================================
# 48. SAVE CUSTOMER SEGMENTATION
# ============================================================

customer_df.to_excel(

    os.path.join(
        CLUSTER_DIR,
        "Customer_Segmentation.xlsx"
    ),

    index=False

)


cluster_summary.to_excel(

    os.path.join(
        CLUSTER_DIR,
        "Customer_Segment_Summary.xlsx"
    ),

    index=False

)


# ============================================================
# 49. CUSTOMER SEGMENT PLOT
# ============================================================

plt.figure(
    figsize=(9, 6)
)


sns.scatterplot(

    data=customer_df,

    x="Customer Total Sales",

    y="Customer Total Profit",

    hue="Customer Segment",

    s=70

)


plt.title(
    "Customer Segmentation using K-Means"
)

plt.xlabel(
    "Customer Total Sales"
)

plt.ylabel(
    "Customer Total Profit"
)

plt.tight_layout()


plt.savefig(

    os.path.join(
        CLUSTER_DIR,
        "Customer_Segmentation.png"
    ),

    dpi=300

)


plt.close()


# ============================================================
# 50. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("MACHINE LEARNING + DEEP LEARNING COMPLETED")
print("=" * 80)


print(
    "\nSALES PREDICTION RESULTS"
)


print(
    "----------------------------------------"
)


print(
    f"Linear Regression : R² = {linear_metrics['R2']:.4f}"
)


print(
    f"Decision Tree     : R² = {tree_metrics['R2']:.4f}"
)


print(
    f"Random Forest     : R² = {rf_metrics['R2']:.4f}"
)


print(
    f"XGBoost           : R² = {xgb_r2:.4f}"
)


print(
    f"LSTM              : R² = {lstm_r2:.4f}"
)


print(
    f"Hybrid            : R² = {hybrid_r2:.4f}"
)


print(
    "\nPROPOSED MODEL"
)


print(
    "XGBoost + LSTM Feature Hybrid"
)


print(
    "\nMODEL TYPE"
)


print(
    "Hybrid Machine Learning + Deep Learning"
)


print(
    "\nHYBRID ARCHITECTURE"
)


print(
    "Historical Monthly Sales"
)


print(
    "        ↓"
)


print(
    "LSTM Temporal Learning"
)


print(
    "        ↓"
)


print(
    "LSTM Temporal Prediction"
)


print(
    "        ↓"
)


print(
    "Transaction Features + LSTM Feature"
)


print(
    "        ↓"
)


print(
    "Final XGBoost Hybrid Model"
)


print(
    "        ↓"
)


print(
    "Final Hybrid Sales Prediction"
)


print(
    "\nHybrid Feature:"
)


print(
    "LSTM Temporal Prediction"
)


print(
    "\nEvaluation Metrics:"
)


print(
    "MAE, MSE, RMSE, R², MAPE, SMAPE"
)


print(
    "\nCustomer Segmentation:"
)


print(
    "Number of clusters:",
    best_k
)


print(
    "\nOutput folder:"
)


print(
    ML_DIR
)


print(
    "\nAll Machine Learning, Deep Learning,"
)


print(
    "Hybrid Model and Customer Segmentation outputs"
)


print(
    "have been saved successfully."
)


print("=" * 80)