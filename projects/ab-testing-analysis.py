"""
analysis.py — E-Commerce A/B Test: Did the New Landing Page Actually Work?

Business question: An e-commerce company built a new landing page and ran
an A/B test to see if it improves conversion. Should they ship it, keep the
old page, or run the test longer?

This project deliberately includes the parts most A/B-test tutorials skip:
- Real data contamination (users assigned to the wrong page) that must be
  cleaned before any test is valid
- A sample-ratio-mismatch (SRM) check, which catches broken randomization
- A proper two-proportion z-test with a confidence interval, not just "it
  went up"
- A statistical power / minimum-detectable-effect discussion, since a null
  result can mean "no effect" OR "not enough data to tell"
"""

import pandas as pd
import numpy as np
import json
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest, proportion_confint
from statsmodels.stats.power import NormalIndPower

results = {}

# ---------------------------------------------------------------------
# 1. Load and inspect raw data quality
# ---------------------------------------------------------------------
df_raw = pd.read_csv('data/ab_data.csv')

mismatch = df_raw[
    ((df_raw['group'] == 'control') & (df_raw['landing_page'] == 'new_page')) |
    ((df_raw['group'] == 'treatment') & (df_raw['landing_page'] == 'old_page'))
]
n_mismatch = len(mismatch)
n_duplicates = df_raw['user_id'].duplicated().sum()

results['data_quality'] = {
    'raw_rows': len(df_raw),
    'mismatched_assignment_rows': int(n_mismatch),
    'duplicate_user_id_rows': int(n_duplicates),
}
print("=== DATA QUALITY CHECK ===")
print(f"Raw rows: {len(df_raw):,}")
print(f"Mismatched group/page assignment: {n_mismatch:,} rows "
      f"({n_mismatch/len(df_raw)*100:.2f}%) — these users saw the WRONG page "
      f"for their assigned group and must be excluded, since we can't know "
      f"which page actually influenced their behavior")
print(f"Duplicate user_id rows: {n_duplicates:,}")

# ---------------------------------------------------------------------
# 2. Clean: remove mismatches, then remove duplicate users
# ---------------------------------------------------------------------
df_clean = df_raw[
    ((df_raw['group'] == 'control') & (df_raw['landing_page'] == 'old_page')) |
    ((df_raw['group'] == 'treatment') & (df_raw['landing_page'] == 'new_page'))
].copy()

# For any remaining duplicate user_ids, keep only the first occurrence
n_dupes_after_mismatch_removal = df_clean['user_id'].duplicated().sum()
df_clean = df_clean.drop_duplicates(subset='user_id', keep='first')

results['data_quality']['rows_after_cleaning'] = len(df_clean)
results['data_quality']['rows_removed'] = len(df_raw) - len(df_clean)
print(f"\nAfter removing mismatches and duplicates: {len(df_clean):,} rows "
      f"({len(df_raw) - len(df_clean):,} removed, "
      f"{(len(df_raw) - len(df_clean))/len(df_raw)*100:.2f}% of raw data)")

# ---------------------------------------------------------------------
# 3. Sample Ratio Mismatch (SRM) check
#    Before trusting ANY result, verify the split is actually ~50/50 as
#    intended. A significant SRM means something broke in randomization
#    or logging, and the whole test should be distrusted regardless of
#    the conversion result.
# ---------------------------------------------------------------------
group_counts = df_clean['group'].value_counts()
n_control, n_treatment = group_counts['control'], group_counts['treatment']
srm_chi2, srm_p = stats.chisquare([n_control, n_treatment])

results['srm_check'] = {
    'n_control': int(n_control), 'n_treatment': int(n_treatment),
    'chi2': round(srm_chi2, 4), 'p_value': round(srm_p, 4),
    'passed': bool(srm_p > 0.01)  # common SRM threshold is stricter than 0.05
}
print(f"\n=== SAMPLE RATIO MISMATCH (SRM) CHECK ===")
print(f"Control: {n_control:,} | Treatment: {n_treatment:,}")
print(f"Chi-square p-value: {srm_p:.4f} "
      f"({'PASSED - randomization looks healthy' if srm_p > 0.01 else 'FAILED - investigate before trusting results'})")

# ---------------------------------------------------------------------
# 4. THE MAIN TEST: two-proportion z-test on conversion rate
# ---------------------------------------------------------------------
control_conversions = df_clean[df_clean['group'] == 'control']['converted'].sum()
treatment_conversions = df_clean[df_clean['group'] == 'treatment']['converted'].sum()

control_rate = control_conversions / n_control
treatment_rate = treatment_conversions / n_treatment

count = np.array([treatment_conversions, control_conversions])
nobs = np.array([n_treatment, n_control])
z_stat, p_value = proportions_ztest(count, nobs, alternative='larger')  # H1: new > old

# 95% confidence interval for the DIFFERENCE in conversion rates
diff = treatment_rate - control_rate
se_diff = np.sqrt(control_rate * (1 - control_rate) / n_control +
                   treatment_rate * (1 - treatment_rate) / n_treatment)
ci_low, ci_high = diff - 1.96 * se_diff, diff + 1.96 * se_diff

results['main_test'] = {
    'control_conversions': int(control_conversions), 'control_n': int(n_control),
    'control_rate': round(control_rate, 5),
    'treatment_conversions': int(treatment_conversions), 'treatment_n': int(n_treatment),
    'treatment_rate': round(treatment_rate, 5),
    'absolute_diff': round(diff, 5),
    'relative_diff_pct': round(diff / control_rate * 100, 2),
    'z_statistic': round(z_stat, 4),
    'p_value': round(p_value, 4),
    'ci_95_low': round(ci_low, 5),
    'ci_95_high': round(ci_high, 5),
    'significant_at_05': bool(p_value < 0.05),
}
print(f"\n=== MAIN TEST: New Page vs. Old Page Conversion ===")
print(f"Control (old page):   {control_conversions:,} / {n_control:,} = {control_rate*100:.3f}%")
print(f"Treatment (new page): {treatment_conversions:,} / {n_treatment:,} = {treatment_rate*100:.3f}%")
print(f"Absolute difference: {diff*100:.3f} percentage points")
print(f"95% CI for difference: [{ci_low*100:.3f}%, {ci_high*100:.3f}%]")
print(f"One-sided z-test (H1: new > old): z={z_stat:.4f}, p={p_value:.4f}")
print(f"{'SIGNIFICANT' if p_value < 0.05 else 'NOT SIGNIFICANT'} at alpha=0.05")

# ---------------------------------------------------------------------
# 5. Statistical power / minimum detectable effect
#    A null result can mean "truly no effect" OR "not enough data to
#    detect a real but small effect." This distinguishes the two.
# ---------------------------------------------------------------------
power_analysis = NormalIndPower()
# What effect size (Cohen's h) COULD we have reliably detected with this
# sample size, at 80% power?
achieved_power = power_analysis.power(
    effect_size=2 * (np.arcsin(np.sqrt(treatment_rate)) - np.arcsin(np.sqrt(control_rate))),
    nobs1=n_control, alpha=0.05, ratio=n_treatment/n_control
)

# Minimum detectable absolute lift at 80% power, given actual sample size
def cohens_h(p1, p2):
    return 2 * np.arcsin(np.sqrt(p2)) - 2 * np.arcsin(np.sqrt(p1))

# Solve for minimum detectable p2 numerically via search
target_power = 0.80
test_lifts = np.linspace(0.0001, 0.02, 200)
mde = None
for lift in test_lifts:
    h = cohens_h(control_rate, control_rate + lift)
    pw = power_analysis.power(effect_size=h, nobs1=n_control, alpha=0.05, ratio=1)
    if pw >= target_power:
        mde = lift
        break

results['power_analysis'] = {
    'achieved_power_for_observed_effect': round(float(achieved_power), 4),
    'minimum_detectable_lift_at_80pct_power': round(float(mde), 5) if mde else None,
}
print(f"\n=== STATISTICAL POWER ===")
print(f"Achieved power for the observed effect size: {achieved_power:.4f}")
print(f"Minimum detectable absolute lift at 80% power with this sample size: "
      f"{mde*100:.3f} percentage points" if mde else "N/A")

# ---------------------------------------------------------------------
# 6. Time trend check (novelty effect / decay)
# ---------------------------------------------------------------------
df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'])
df_clean['date'] = df_clean['timestamp'].dt.date

daily = df_clean.groupby(['date', 'group'])['converted'].agg(['mean', 'count']).reset_index()
daily_pivot = daily.pivot(index='date', columns='group', values='mean').reset_index()
daily_pivot.columns = ['date', 'control_rate', 'treatment_rate']
daily_pivot['date'] = daily_pivot['date'].astype(str)

results['daily_trend'] = daily_pivot.to_dict('records')
print(f"\n=== DAILY TREND ({len(daily_pivot)} days) ===")
print(daily_pivot.to_string(index=False))

with open('dashboard_data.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print("\n\nSaved dashboard_data.json")
