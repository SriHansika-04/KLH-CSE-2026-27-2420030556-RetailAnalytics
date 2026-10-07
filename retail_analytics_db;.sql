CREATE DATABASE retail_analytics;
SHOW DATABASES;
USE retail_analytics;

SELECT DATABASE();

USE retail_analytics;

CREATE TABLE sales_data (
    order_id VARCHAR(50),
    order_date DATE,
    ship_date DATE,
    ship_mode VARCHAR(50),
    customer_id VARCHAR(50),
    customer_name VARCHAR(150),
    segment VARCHAR(50),
    city VARCHAR(100),
    state VARCHAR(100),
    country VARCHAR(100),
    market VARCHAR(50),
    region VARCHAR(100),
    product_id VARCHAR(50),
    category VARCHAR(50),
    sub_category VARCHAR(100),
    product_name VARCHAR(255),
    order_priority VARCHAR(50),
    sales DECIMAL(12,2),
    quantity INT,
    discount DECIMAL(5,2),
    profit DECIMAL(12,2),
    shipping_cost DECIMAL(12,2)
);

SHOW TABLES;

DESCRIBE sales_data;

USE retail_analytics;

SELECT
    SUM(sales) AS total_sales,
    SUM(profit) AS total_profit,
    SUM(quantity) AS total_quantity
FROM sales_data;

USE retail_analytics;

-- 1. Overall business summary
SELECT
    COUNT(*) AS total_records,
    ROUND(SUM(sales), 2) AS total_sales,
    ROUND(SUM(profit), 2) AS total_profit,
    SUM(quantity) AS total_quantity
FROM sales_data;


-- 2. Sales and profit by category
SELECT
    category,
    ROUND(SUM(sales), 2) AS total_sales,
    ROUND(SUM(profit), 2) AS total_profit,
    SUM(quantity) AS total_quantity
FROM sales_data
GROUP BY category
ORDER BY total_sales DESC;


-- 3. Top 10 products by sales
SELECT
    product_id,
    product_name,
    ROUND(SUM(sales), 2) AS total_sales,
    ROUND(SUM(profit), 2) AS total_profit
FROM sales_data
GROUP BY product_id, product_name
ORDER BY total_sales DESC
LIMIT 10;


-- 4. Top 10 customers by sales
SELECT
    customer_id,
    customer_name,
    COUNT(DISTINCT order_id) AS order_count,
    ROUND(SUM(sales), 2) AS total_sales,
    ROUND(SUM(profit), 2) AS total_profit
FROM sales_data
GROUP BY customer_id, customer_name
ORDER BY total_sales DESC
LIMIT 10;


-- 5. JOIN: Customer-level sales and profit
SELECT
    c.customer_id,
    c.customer_name,
    c.total_sales,
    p.total_profit
FROM
    (
        SELECT
            customer_id,
            customer_name,
            SUM(sales) AS total_sales
        FROM sales_data
        GROUP BY customer_id, customer_name
    ) c
JOIN
    (
        SELECT
            customer_id,
            SUM(profit) AS total_profit
        FROM sales_data
        GROUP BY customer_id
    ) p
ON c.customer_id = p.customer_id
ORDER BY c.total_sales DESC
LIMIT 10;