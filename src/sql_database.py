# ============================================================
# RETAIL ANALYTICS - SQL DATABASE INTEGRATION
# ============================================================

import pandas as pd
from sqlalchemy import create_engine, text
from pathlib import Path
from getpass import getpass

# ------------------------------------------------------------
# 1. Project paths
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "outputs"
    / "01_Preprocessing"
    / "Preprocessed_Global_Superstore.xlsx"
)

print("Input file:")
print(INPUT_FILE)

# ------------------------------------------------------------
# 2. MySQL connection details
# ------------------------------------------------------------

MYSQL_USER = "root"
MYSQL_PASSWORD = getpass("Enter your MySQL password: ")
MYSQL_HOST = "localhost"
MYSQL_PORT = 3306
DATABASE = "retail_analytics"

# Create MySQL connection
connection_string = (
    f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}"
    f"@{MYSQL_HOST}:{MYSQL_PORT}/{DATABASE}"
)

engine = create_engine(connection_string)

# ------------------------------------------------------------
# 3. Test MySQL connection
# ------------------------------------------------------------

try:
    with engine.connect() as connection:
        result = connection.execute(text("SELECT DATABASE();"))
        database_name = result.scalar()

    print("\nMySQL connection successful!")
    print("Connected database:", database_name)

except Exception as e:
    print("\nMySQL connection failed.")
    print("Error:", e)
    raise SystemExit

# ------------------------------------------------------------
# 4. Load preprocessed Excel data
# ------------------------------------------------------------

print("\nLoading preprocessed dataset...")

df = pd.read_excel(INPUT_FILE)

print("Rows loaded:", len(df))
print("Columns loaded:", len(df.columns))

# ------------------------------------------------------------
# 5. Select columns required for SQL table
# ------------------------------------------------------------

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
    "Profit",
    "Shipping Cost"
]

# Check that all required columns exist
missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    print("\nMissing columns:")
    print(missing_columns)
    raise SystemExit

df = df[required_columns].copy()

# ------------------------------------------------------------
# 6. Rename columns to SQL-friendly names
# ------------------------------------------------------------

df.rename(columns={
    "Order ID": "order_id",
    "Order Date": "order_date",
    "Ship Date": "ship_date",
    "Ship Mode": "ship_mode",
    "Customer ID": "customer_id",
    "Customer Name": "customer_name",
    "Segment": "segment",
    "City": "city",
    "State": "state",
    "Country": "country",
    "Market": "market",
    "Region": "region",
    "Product ID": "product_id",
    "Category": "category",
    "Sub-Category": "sub_category",
    "Product Name": "product_name",
    "Order Priority": "order_priority",
    "Sales": "sales",
    "Quantity": "quantity",
    "Discount": "discount",
    "Profit": "profit",
    "Shipping Cost": "shipping_cost"
}, inplace=True)

# ------------------------------------------------------------
# 7. Convert dates
# ------------------------------------------------------------

df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
df["ship_date"] = pd.to_datetime(df["ship_date"], errors="coerce")

# Convert dates to Python date format
df["order_date"] = df["order_date"].dt.date
df["ship_date"] = df["ship_date"].dt.date

# ------------------------------------------------------------
# 8. Check data before uploading
# ------------------------------------------------------------

print("\nData prepared for MySQL.")
print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nMissing values:")
print(df.isnull().sum().sum())

# ------------------------------------------------------------
# 9. Clear existing records if script is run again
# ------------------------------------------------------------

with engine.begin() as connection:
    connection.execute(text("DELETE FROM sales_data"))

print("\nExisting records cleared from sales_data.")

# ------------------------------------------------------------
# 10. Upload data to MySQL
# ------------------------------------------------------------

print("\nUploading data to MySQL...")

df.to_sql(
    "sales_data",
    con=engine,
    if_exists="append",
    index=False,
    chunksize=1000,
    method="multi"
)

print("\nData uploaded successfully!")

# ------------------------------------------------------------
# 11. Verify number of records
# ------------------------------------------------------------

with engine.connect() as connection:
    result = connection.execute(
        text("SELECT COUNT(*) FROM sales_data")
    )

    row_count = result.scalar()

print("\nRecords in MySQL:", row_count)

# ------------------------------------------------------------
# 12. Display sample records
# ------------------------------------------------------------

sample = pd.read_sql(
    "SELECT * FROM sales_data LIMIT 5",
    engine
)

print("\nFirst 5 records:")
print(sample)

print("\n================================================")
print("SQL DATABASE INTEGRATION COMPLETED")
print("================================================")