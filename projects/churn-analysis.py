"""
analysis.py — Telco Customer Churn: Prediction + Business Recommendation

Business question: Which customers are about to churn, and what's the
actual dollar-optimal way to spend a limited retention budget on them?

This deliberately goes beyond "train a classifier and report accuracy":
1. Segments customers (RFM-style: tenure, monthly spend, service engagement)
2. Trains and compares two model families on a real held-out test set
3. Converts model output into a per-customer cost-optimized retention
   decision, using each customer's own revenue at stake (not a flat cost
   matrix) — a more realistic version of the cost-threshold framework.
"""

import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix

RANDOM_STATE = 42
results = {}

# ---------------------------------------------------------------------
# 1. Load and clean
# ---------------------------------------------------------------------
df = pd.read_csv('data/telco_churn.csv')

# Known data quality issue: 11 rows have blank TotalCharges (all are
# customers with tenure=0, i.e. brand new — TotalCharges wasn't computed
# yet). Documenting rather than silently imputing.
blank_mask = df['TotalCharges'].str.strip() == ''
n_blank = blank_mask.sum()
df.loc[blank_mask, 'TotalCharges'] = '0'
df['TotalCharges'] = df['TotalCharges'].astype(float)
df['Churn_binary'] = (df['Churn'] == 'Yes').astype(int)

results['data_quality'] = {
    'total_customers': len(df),
    'blank_total_charges_rows': int(n_blank),
    'overall_churn_rate': round(df['Churn_binary'].mean() * 100, 2)
}
print(f"=== DATA QUALITY ===")
print(f"{len(df)} customers, {n_blank} had blank TotalCharges (all tenure=0, new customers) -> set to 0")
print(f"Overall churn rate: {results['data_quality']['overall_churn_rate']}%")

# ---------------------------------------------------------------------
# 2. Customer segmentation (RFM-style, adapted for a subscription business)
#    Recency/loyalty -> tenure | Frequency/engagement -> # services subscribed
#    | Monetary -> MonthlyCharges
# ---------------------------------------------------------------------
service_cols = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
df['num_services'] = df[service_cols].apply(lambda row: (row == 'Yes').sum(), axis=1)

seg_features = df[['tenure', 'num_services', 'MonthlyCharges']].copy()
scaler = StandardScaler()
seg_scaled = scaler.fit_transform(seg_features)

kmeans = KMeans(n_clusters=4, random_state=RANDOM_STATE, n_init=10)
df['segment'] = kmeans.fit_predict(seg_scaled)

segment_summary = df.groupby('segment').agg(
    customers=('customerID', 'count'),
    avg_tenure=('tenure', 'mean'),
    avg_services=('num_services', 'mean'),
    avg_monthly_charge=('MonthlyCharges', 'mean'),
    churn_rate=('Churn_binary', 'mean'),
).round(2).reset_index()

# Label segments by their actual characteristics rather than just cluster number
def label_segment(row):
    if row['churn_rate'] > 0.35:
        return 'New & At-Risk'
    elif row['avg_tenure'] > 40 and row['avg_monthly_charge'] > 70:
        return 'Loyal High-Value'
    elif row['avg_tenure'] > 40:
        return 'Loyal Low-Spend'
    else:
        return 'New Low-Engagement'

segment_summary['label'] = segment_summary.apply(label_segment, axis=1)
results['segments'] = segment_summary.to_dict('records')
print(f"\n=== CUSTOMER SEGMENTS ===")
print(segment_summary[['label', 'customers', 'avg_tenure', 'avg_services', 'avg_monthly_charge', 'churn_rate']].to_string(index=False))

# ---------------------------------------------------------------------
# 3. Model training — compare Logistic Regression vs Random Forest
# ---------------------------------------------------------------------
categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                     'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                     'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                     'PaperlessBilling', 'PaymentMethod']

df_model = df.copy()
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])
    encoders[col] = le

feature_cols = categorical_cols + ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 'num_services']
X = df_model[feature_cols]
y = df_model['Churn_binary']

X_train, X_test, y_train, y_test, df_train_full, df_test_full = train_test_split(
    X, y, df_model, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)

scaler2 = StandardScaler()
X_train_scaled = scaler2.fit_transform(X_train)
X_test_scaled = scaler2.transform(X_test)

log_reg = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
log_reg.fit(X_train_scaled, y_train)
log_reg_probs = log_reg.predict_proba(X_test_scaled)[:, 1]

rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=RANDOM_STATE)
rf.fit(X_train, y_train)
rf_probs = rf.predict_proba(X_test)[:, 1]

def eval_model(y_true, probs, name):
    preds = (probs >= 0.5).astype(int)
    auc = roc_auc_score(y_true, probs)
    prec = precision_score(y_true, preds)
    rec = recall_score(y_true, preds)
    f1 = f1_score(y_true, preds)
    print(f"{name}: AUC={auc:.3f} Precision={prec:.3f} Recall={rec:.3f} F1={f1:.3f}")
    return {'auc': round(auc, 3), 'precision': round(prec, 3), 'recall': round(rec, 3), 'f1': round(f1, 3)}

print(f"\n=== MODEL COMPARISON (held-out test set, {len(y_test)} customers) ===")
log_reg_metrics = eval_model(y_test, log_reg_probs, "Logistic Regression")
rf_metrics = eval_model(y_test, rf_probs, "Random Forest")

results['model_comparison'] = {'logistic_regression': log_reg_metrics, 'random_forest': rf_metrics}

# Use whichever model has higher AUC for the business recommendation
best_probs = rf_probs if rf_metrics['auc'] >= log_reg_metrics['auc'] else log_reg_probs
best_model_name = 'Random Forest' if rf_metrics['auc'] >= log_reg_metrics['auc'] else 'Logistic Regression'
print(f"\nUsing {best_model_name} for the business recommendation (higher AUC).")

# Feature importance (Random Forest)
importances = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False).head(10)
results['feature_importance'] = importances.to_dict('records')
print(f"\n=== TOP FEATURES (Random Forest) ===")
print(importances.to_string(index=False))

# ---------------------------------------------------------------------
# 4. Cost-optimized retention threshold
#    Unlike a flat cost matrix, each customer's false-negative cost is
#    THEIR OWN annual revenue at stake (MonthlyCharges * 12) — a more
#    realistic model of what a real retention team would optimize.
# ---------------------------------------------------------------------
RETENTION_OFFER_COST = 100  # cost of a retention incentive (e.g. one month free / discount)

test_monthly_charges = df_test_full['MonthlyCharges'].values
annual_value_at_stake = test_monthly_charges * 12

def total_cost_at_threshold(y_true, probs, threshold, annual_value):
    preds = (probs >= threshold).astype(int)
    cost = 0.0
    for actual, pred, value in zip(y_true, preds, annual_value):
        if pred == 1:
            cost += RETENTION_OFFER_COST  # spent on retention offer whether needed or not
        elif pred == 0 and actual == 1:
            cost += value  # missed churner: lost their annual revenue
    return cost

thresholds = np.linspace(0.05, 0.95, 50)
costs = [total_cost_at_threshold(y_test.values, best_probs, t, annual_value_at_stake) for t in thresholds]

opt_idx = np.argmin(costs)
opt_threshold = thresholds[opt_idx]
min_cost = costs[opt_idx]

default_cost = total_cost_at_threshold(y_test.values, best_probs, 0.5, annual_value_at_stake)
no_model_cost = sum(v for actual, v in zip(y_test.values, annual_value_at_stake) if actual == 1)  # do-nothing baseline

print(f"\n=== COST-OPTIMIZED RETENTION THRESHOLD ===")
print(f"Do-nothing baseline (no retention program): ${no_model_cost:,.2f}")
print(f"Standard threshold (0.50): ${default_cost:,.2f}")
print(f"Optimized threshold ({opt_threshold:.2f}): ${min_cost:,.2f}")
print(f"Savings vs do-nothing: ${no_model_cost - min_cost:,.2f}")
print(f"Savings vs standard threshold: ${default_cost - min_cost:,.2f}")

results['cost_optimization'] = {
    'retention_offer_cost': RETENTION_OFFER_COST,
    'test_set_size': len(y_test),
    'no_model_cost': round(no_model_cost, 2),
    'default_threshold_cost': round(default_cost, 2),
    'optimal_threshold': round(float(opt_threshold), 3),
    'optimal_cost': round(min_cost, 2),
    'savings_vs_no_model': round(no_model_cost - min_cost, 2),
    'savings_vs_default': round(default_cost - min_cost, 2),
    'threshold_curve': [{'threshold': round(float(t), 3), 'cost': round(c, 2)} for t, c in zip(thresholds, costs)]
}

# ---------------------------------------------------------------------
# 5. Churn rate by key business dimensions (contract type, tenure bucket)
# ---------------------------------------------------------------------
contract_churn = df.groupby('Contract')['Churn_binary'].agg(['mean', 'count']).round(3).reset_index()
contract_churn.columns = ['contract_type', 'churn_rate', 'customer_count']
results['churn_by_contract'] = contract_churn.to_dict('records')
print(f"\n=== CHURN RATE BY CONTRACT TYPE ===")
print(contract_churn.to_string(index=False))

df['tenure_bucket'] = pd.cut(df['tenure'], bins=[-1, 12, 24, 48, 100],
                               labels=['0-12 mo', '13-24 mo', '25-48 mo', '49+ mo'])
tenure_churn = df.groupby('tenure_bucket', observed=True)['Churn_binary'].agg(['mean', 'count']).round(3).reset_index()
tenure_churn.columns = ['tenure_bucket', 'churn_rate', 'customer_count']
results['churn_by_tenure'] = tenure_churn.to_dict('records')
print(f"\n=== CHURN RATE BY TENURE ===")
print(tenure_churn.to_string(index=False))

with open('dashboard_data.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print("\n\nSaved dashboard_data.json")
