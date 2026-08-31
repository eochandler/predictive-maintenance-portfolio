import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# --- 1. SIMULATE FLEET PREDICTIONS ---
# 1,000 machines over a 1-year period
n_samples = 1000

# Actual ground truth: 10% of machines experience critical failure (1 = Failure, 0 = Healthy)
y_true = np.random.choice([0, 1], size=n_samples, p=[0.90, 0.10])

# Simulated model predicted probabilities of failure
# Add noise to represent a model with ~0.85 ROC-AUC performance
y_prob = np.where(y_true == 1, 
                  np.random.beta(a=5, b=2, size=n_samples),  # High probabilities for actual failures
                  np.random.beta(a=1, b=5, size=n_samples)) # Low probabilities for healthy assets

# --- 2. COST PARAMETERS (Industry Benchmarks) ---
COST_UNPLANNED_FAILURE = 10000  # False Negative cost (Catastrophic downtime)
COST_PREVENTIVE_MAINT = 1500   # True Positive cost (Scheduled replacement)
COST_FALSE_ALARM       = 300    # False Positive cost (Unnecessary inspection)

# --- 3. THRESHOLD OPTIMIZATION LOOP ---
thresholds = np.linspace(0.01, 0.99, 99)
results = []

for t in thresholds:
    y_pred = (y_prob >= t).astype(int)
    
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    
    # Financial calculation
    total_cost = (fn * COST_UNPLANNED_FAILURE) + (tp * COST_PREVENTIVE_MAINT) + (fp * COST_FALSE_ALARM)
    savings_vs_reactive = (np.sum(y_true) * COST_UNPLANNED_FAILURE) - total_cost
    
    results.append({
        'threshold': t,
        'total_cost': total_cost,
        'savings_vs_reactive': savings_vs_reactive,
        'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn
    })

df_results = pd.DataFrame(results)

# Find optimal threshold (minimum cost)
best_row = df_results.loc[df_results['total_cost'].idxmin()]

print("=" * 50)
print(f"OPTIMAL THRESHOLD FOUND: {best_row['threshold']:.2f}")
print("=" * 50)
print(f"Total Operational Cost at Optimal Threshold: ${best_row['total_cost']:,.2f}")
print(f"Total Savings vs Run-to-Failure Policy:     ${best_row['savings_vs_reactive']:,.2f}")
print(f"Unplanned Failures Prevented (TP):           {int(best_row['TP'])} / {np.sum(y_true)}")
print(f"Missed Catastrophic Failures (FN):           {int(best_row['FN'])}")
print(f"False Alarm Inspections (FP):                {int(best_row['FP'])}")
print("=" * 50)