-- ============================================================================
-- Olist Marketplace: Delivery Risk & Revenue Analysis
-- Business question: Where is Olist's marketplace losing revenue and customer
-- satisfaction in the order-to-delivery pipeline, and which product categories
-- / states carry the most risk?
--
-- Run with DuckDB (works directly against the raw CSVs, no import step needed):
--   duckdb < queries.sql
-- Or in Python: duckdb.connect().execute(open('queries.sql').read())
-- ============================================================================

-- Register raw CSVs as queryable tables
CREATE VIEW orders AS SELECT * FROM read_csv_auto('data/olist_orders_dataset.csv');
CREATE VIEW order_items AS SELECT * FROM read_csv_auto('data/olist_order_items_dataset.csv');
CREATE VIEW payments AS SELECT * FROM read_csv_auto('data/olist_order_payments_dataset.csv');
CREATE VIEW reviews AS SELECT * FROM read_csv_auto('data/olist_order_reviews_dataset.csv');
CREATE VIEW customers AS SELECT * FROM read_csv_auto('data/olist_customers_dataset.csv');
CREATE VIEW products AS SELECT * FROM read_csv_auto('data/olist_products_dataset.csv');
CREATE VIEW sellers AS SELECT * FROM read_csv_auto('data/olist_sellers_dataset.csv');
CREATE VIEW category_translation AS SELECT * FROM read_csv_auto('data/product_category_name_translation.csv');


-- ----------------------------------------------------------------------------
-- 1. Headline KPIs
-- ----------------------------------------------------------------------------
SELECT
    COUNT(DISTINCT o.order_id) AS total_orders,
    ROUND(SUM(oi.price + oi.freight_value), 2) AS total_revenue,
    ROUND(AVG(oi.price + oi.freight_value), 2) AS avg_order_item_value,
    ROUND(AVG(r.review_score), 2) AS avg_review_score
FROM orders o
JOIN order_items oi ON o.order_id = oi.order_id
LEFT JOIN reviews r ON o.order_id = r.order_id
WHERE o.order_status = 'delivered';


-- ----------------------------------------------------------------------------
-- 2. THE KEY FINDING: does late delivery drive bad reviews?
-- This is the central business question of the whole analysis.
-- ----------------------------------------------------------------------------
SELECT
    CASE
        WHEN o.order_delivered_customer_date IS NULL THEN 'Not Delivered'
        WHEN CAST(o.order_delivered_customer_date AS TIMESTAMP) > CAST(o.order_estimated_delivery_date AS TIMESTAMP)
            THEN 'Late'
        ELSE 'On Time'
    END AS delivery_status,
    ROUND(AVG(r.review_score), 2) AS avg_review_score,
    COUNT(DISTINCT o.order_id) AS order_count
FROM orders o
LEFT JOIN reviews r ON o.order_id = r.order_id
WHERE o.order_status = 'delivered'
GROUP BY delivery_status
ORDER BY avg_review_score DESC;


-- ----------------------------------------------------------------------------
-- 3. Late-delivery rate trend by month
-- ----------------------------------------------------------------------------
SELECT
    strftime(CAST(o.order_purchase_timestamp AS TIMESTAMP), '%Y-%m') AS month,
    ROUND(100.0 * SUM(CASE
        WHEN o.order_delivered_customer_date IS NOT NULL
         AND CAST(o.order_delivered_customer_date AS TIMESTAMP) > CAST(o.order_estimated_delivery_date AS TIMESTAMP)
        THEN 1 ELSE 0 END) / COUNT(*), 2) AS late_pct
FROM orders o
WHERE o.order_status = 'delivered'
GROUP BY month
ORDER BY month;


-- ----------------------------------------------------------------------------
-- 4. Top product categories: revenue vs. late-delivery risk
-- ----------------------------------------------------------------------------
SELECT
    COALESCE(ct.product_category_name_english, p.product_category_name, 'unknown') AS category,
    ROUND(SUM(oi.price), 2) AS revenue,
    COUNT(DISTINCT o.order_id) AS order_count,
    ROUND(100.0 * SUM(CASE
        WHEN o.order_delivered_customer_date IS NOT NULL
         AND CAST(o.order_delivered_customer_date AS TIMESTAMP) > CAST(o.order_estimated_delivery_date AS TIMESTAMP)
        THEN 1 ELSE 0 END) / COUNT(*), 2) AS late_pct,
    ROUND(AVG(r.review_score), 2) AS avg_review_score
FROM orders o
JOIN order_items oi ON o.order_id = oi.order_id
JOIN products p ON oi.product_id = p.product_id
LEFT JOIN category_translation ct ON p.product_category_name = ct.product_category_name
LEFT JOIN reviews r ON o.order_id = r.order_id
WHERE o.order_status = 'delivered'
GROUP BY category
HAVING order_count >= 100
ORDER BY revenue DESC
LIMIT 15;


-- ----------------------------------------------------------------------------
-- 5. State-level performance (customer state)
-- ----------------------------------------------------------------------------
SELECT
    c.customer_state AS state,
    ROUND(SUM(oi.price + oi.freight_value), 2) AS revenue,
    COUNT(DISTINCT o.order_id) AS order_count,
    ROUND(AVG(DATE_DIFF('day',
        CAST(o.order_purchase_timestamp AS TIMESTAMP),
        CAST(o.order_delivered_customer_date AS TIMESTAMP))), 1) AS avg_delivery_days,
    ROUND(AVG(r.review_score), 2) AS avg_review_score
FROM orders o
JOIN order_items oi ON o.order_id = oi.order_id
JOIN customers c ON o.customer_id = c.customer_id
LEFT JOIN reviews r ON o.order_id = r.order_id
WHERE o.order_status = 'delivered' AND o.order_delivered_customer_date IS NOT NULL
GROUP BY state
HAVING order_count >= 50
ORDER BY revenue DESC;


-- ----------------------------------------------------------------------------
-- 6. Payment method breakdown
-- ----------------------------------------------------------------------------
SELECT
    payment_type,
    COUNT(*) AS count,
    ROUND(AVG(payment_installments), 1) AS avg_installments,
    ROUND(SUM(payment_value), 2) AS total_value
FROM payments
WHERE payment_type != 'not_defined'
GROUP BY payment_type
ORDER BY total_value DESC;
