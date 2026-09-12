"""
analysis.py

Business question this project answers:
"Where is Olist's marketplace losing revenue and customer satisfaction in
the order-to-delivery pipeline, and which product categories/states carry
the most risk?"

This mirrors a real ops/marketplace-analytics ask: leadership doesn't want
"explore the data" - they want specific, actionable findings. All queries
below are real SQL run with DuckDB directly against the raw CSVs (no
Python-side aggregation) - this is deliberately built to demonstrate SQL,
not to route around it.
"""

import duckdb
import json

con = duckdb.connect()

# Register the raw CSVs as SQL-queryable tables
con.execute("""
    CREATE VIEW orders AS SELECT * FROM read_csv_auto('data/olist_orders_dataset.csv');
    CREATE VIEW order_items AS SELECT * FROM read_csv_auto('data/olist_order_items_dataset.csv');
    CREATE VIEW payments AS SELECT * FROM read_csv_auto('data/olist_order_payments_dataset.csv');
    CREATE VIEW reviews AS SELECT * FROM read_csv_auto('data/olist_order_reviews_dataset.csv');
    CREATE VIEW customers AS SELECT * FROM read_csv_auto('data/olist_customers_dataset.csv');
    CREATE VIEW products AS SELECT * FROM read_csv_auto('data/olist_products_dataset.csv');
    CREATE VIEW sellers AS SELECT * FROM read_csv_auto('data/olist_sellers_dataset.csv');
    CREATE VIEW category_translation AS SELECT * FROM read_csv_auto('data/product_category_name_translation.csv');
""")

results = {}

# ---------------------------------------------------------------------
# 1. Headline KPIs
# ---------------------------------------------------------------------
kpis = con.execute("""
    SELECT
        COUNT(DISTINCT o.order_id) AS total_orders,
        ROUND(SUM(oi.price + oi.freight_value), 2) AS total_revenue,
        ROUND(AVG(oi.price + oi.freight_value), 2) AS avg_order_item_value,
        ROUND(AVG(r.review_score), 2) AS avg_review_score
    FROM orders o
    JOIN order_items oi ON o.order_id = oi.order_id
    LEFT JOIN reviews r ON o.order_id = r.order_id
    WHERE o.order_status = 'delivered'
""").fetchone()
results['kpis'] = {
    'total_orders': kpis[0], 'total_revenue': kpis[1],
    'avg_order_item_value': kpis[2], 'avg_review_score': kpis[3]
}
print("=== HEADLINE KPIs ===")
print(f"Delivered orders: {kpis[0]:,} | Revenue: ${kpis[1]:,.2f} | "
      f"Avg item value: ${kpis[2]} | Avg review score: {kpis[3]}/5")

# ---------------------------------------------------------------------
# 2. Revenue trend over time (monthly)
# ---------------------------------------------------------------------
monthly_revenue = con.execute("""
    SELECT
        strftime(CAST(o.order_purchase_timestamp AS TIMESTAMP), '%Y-%m') AS month,
        ROUND(SUM(oi.price + oi.freight_value), 2) AS revenue,
        COUNT(DISTINCT o.order_id) AS orders
    FROM orders o
    JOIN order_items oi ON o.order_id = oi.order_id
    WHERE o.order_status = 'delivered'
    GROUP BY month
    ORDER BY month
""").fetchall()
results['monthly_revenue'] = [{'month': m, 'revenue': r, 'orders': o} for m, r, o in monthly_revenue]
print(f"\n=== MONTHLY REVENUE ({len(monthly_revenue)} months) ===")
for m, r, o in monthly_revenue[-3:]:
    print(f"{m}: ${r:,.2f} revenue, {o} orders")

# ---------------------------------------------------------------------
# 3. THE KEY FINDING: does late delivery drive bad reviews?
# ---------------------------------------------------------------------
delivery_vs_reviews = con.execute("""
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
    ORDER BY avg_review_score DESC
""").fetchall()
results['delivery_vs_reviews'] = [
    {'status': s, 'avg_score': sc, 'count': c} for s, sc, c in delivery_vs_reviews
]
print("\n=== DELIVERY TIMELINESS vs REVIEW SCORE (key finding) ===")
for status, score, count in delivery_vs_reviews:
    print(f"{status}: {score}/5 avg review ({count:,} orders)")

# ---------------------------------------------------------------------
# 4. Late delivery rate over time (is it getting better or worse?)
# ---------------------------------------------------------------------
late_rate_trend = con.execute("""
    SELECT
        strftime(CAST(o.order_purchase_timestamp AS TIMESTAMP), '%Y-%m') AS month,
        ROUND(100.0 * SUM(CASE
            WHEN o.order_delivered_customer_date IS NOT NULL
             AND CAST(o.order_delivered_customer_date AS TIMESTAMP) > CAST(o.order_estimated_delivery_date AS TIMESTAMP)
            THEN 1 ELSE 0 END) / COUNT(*), 2) AS late_pct
    FROM orders o
    WHERE o.order_status = 'delivered'
    GROUP BY month
    ORDER BY month
""").fetchall()
results['late_rate_trend'] = [{'month': m, 'late_pct': p} for m, p in late_rate_trend]

# ---------------------------------------------------------------------
# 5. Top product categories by revenue AND by late-delivery risk
# ---------------------------------------------------------------------
category_risk = con.execute("""
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
    LIMIT 15
""").fetchall()
results['category_risk'] = [
    {'category': c, 'revenue': rev, 'orders': oc, 'late_pct': lp, 'avg_score': sc}
    for c, rev, oc, lp, sc in category_risk
]
print("\n=== TOP CATEGORIES: revenue vs late-delivery risk ===")
for c, rev, oc, lp, sc in category_risk[:8]:
    print(f"{c}: ${rev:,.0f} revenue, {oc} orders, {lp}% late, {sc}/5 reviews")

# ---------------------------------------------------------------------
# 6. State-level performance (customer state)
# ---------------------------------------------------------------------
state_performance = con.execute("""
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
    ORDER BY revenue DESC
""").fetchall()
results['state_performance'] = [
    {'state': s, 'revenue': rev, 'orders': oc, 'avg_delivery_days': d, 'avg_score': sc}
    for s, rev, oc, d, sc in state_performance
]
print(f"\n=== STATE PERFORMANCE (top 5 of {len(state_performance)}) ===")
for s, rev, oc, d, sc in state_performance[:5]:
    print(f"{s}: ${rev:,.0f}, {oc} orders, {d} days avg delivery, {sc}/5 reviews")

# ---------------------------------------------------------------------
# 7. Payment method breakdown
# ---------------------------------------------------------------------
payment_breakdown = con.execute("""
    SELECT
        payment_type,
        COUNT(*) AS count,
        ROUND(AVG(payment_installments), 1) AS avg_installments,
        ROUND(SUM(payment_value), 2) AS total_value
    FROM payments
    WHERE payment_type != 'not_defined'
    GROUP BY payment_type
    ORDER BY total_value DESC
""").fetchall()
results['payment_breakdown'] = [
    {'type': t, 'count': c, 'avg_installments': ai, 'total_value': tv}
    for t, c, ai, tv in payment_breakdown
]
print("\n=== PAYMENT METHODS ===")
for t, c, ai, tv in payment_breakdown:
    print(f"{t}: {c:,} payments, {ai} avg installments, ${tv:,.0f} total")

# Save everything for the dashboard to consume
with open('dashboard_data.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print("\n\nSaved dashboard_data.json for the dashboard build.")
