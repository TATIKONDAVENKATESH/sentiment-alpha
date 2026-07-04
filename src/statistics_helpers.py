"""
Phase 15 & 16: Statistical Testing & Correlation helpers.
Every test prints H0, H1, statistic, p-value, and a plain-English
interpretation using an explicit alpha = 0.05 significance threshold.
"""
import numpy as np
from scipy import stats

ALPHA = 0.05


def kruskal_test(groups: list, group_names: list, variable_name: str, context: str):
    print(f"\n--- Kruskal-Wallis test: {variable_name} across {context} ---")
    print(f"H0: The distribution of {variable_name} is the same across all groups ({', '.join(group_names)}).")
    print(f"H1: At least one group's distribution of {variable_name} differs.")
    groups = [g for g in groups if len(g) > 0]
    if len(groups) < 2:
        print("Insufficient groups with data - test skipped.")
        return None
    stat, p = stats.kruskal(*groups)
    print(f"Test: Kruskal-Wallis H (non-parametric, appropriate since financial "
          f"PnL/size data is heavy-tailed and non-normal)")
    print(f"H-statistic = {stat:.4f}, p-value = {p:.6g}")
    sig = "statistically significant" if p < ALPHA else "not statistically significant"
    print(f"Interpretation: at alpha={ALPHA}, the difference is {sig}.")
    return {"stat": stat, "p": p, "significant": p < ALPHA}


def mannwhitney_test(a, b, label_a, label_b, variable_name):
    print(f"\n--- Mann-Whitney U test: {variable_name}, {label_a} vs {label_b} ---")
    print(f"H0: {variable_name} distributions of {label_a} and {label_b} are equal.")
    print(f"H1: {variable_name} distributions differ between {label_a} and {label_b}.")
    if len(a) == 0 or len(b) == 0:
        print("Insufficient data in one group - test skipped.")
        return None
    stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    print(f"U-statistic = {stat:.2f}, p-value = {p:.6g}")
    sig = "statistically significant" if p < ALPHA else "not statistically significant"
    print(f"Interpretation: at alpha={ALPHA}, the difference is {sig}.")
    med_a, med_b = np.median(a), np.median(b)
    print(f"Medians: {label_a}={med_a:.4f}, {label_b}={med_b:.4f}")
    return {"stat": stat, "p": p, "significant": p < ALPHA, "median_a": med_a, "median_b": med_b}


def chi_square_test(contingency_table, variable_a, variable_b):
    print(f"\n--- Chi-square test of independence: {variable_a} vs {variable_b} ---")
    print(f"H0: {variable_a} and {variable_b} are independent.")
    print(f"H1: {variable_a} and {variable_b} are associated.")
    stat, p, dof, expected = stats.chi2_contingency(contingency_table)
    print(f"Chi2 = {stat:.4f}, dof = {dof}, p-value = {p:.6g}")
    sig = "statistically significant" if p < ALPHA else "not statistically significant"
    print(f"Interpretation: at alpha={ALPHA}, the association is {sig}.")
    n = contingency_table.to_numpy().sum()
    min_dim = min(contingency_table.shape) - 1
    cramers_v = np.sqrt(stat / (n * min_dim)) if min_dim > 0 else np.nan
    print(f"Effect size (Cramer's V) = {cramers_v:.4f} "
          f"({'small' if cramers_v < 0.1 else 'moderate' if cramers_v < 0.3 else 'large'} effect)")
    return {"stat": stat, "p": p, "significant": p < ALPHA, "cramers_v": cramers_v}


def spearman_corr(x, y, label_x, label_y):
    print(f"\n--- Spearman correlation: {label_x} vs {label_y} ---")
    mask = (~np.isnan(x)) & (~np.isnan(y))
    x, y = np.asarray(x)[mask], np.asarray(y)[mask]
    if len(x) < 3:
        print("Insufficient paired data - test skipped.")
        return None
    rho, p = stats.spearmanr(x, y)
    print(f"n = {len(x)}, Spearman rho = {rho:.4f}, p-value = {p:.6g}")
    sig = "statistically significant" if p < ALPHA else "not statistically significant"
    strength = ("negligible" if abs(rho) < 0.1 else "weak" if abs(rho) < 0.3
                else "moderate" if abs(rho) < 0.5 else "strong")
    print(f"Interpretation: {strength} monotonic relationship, {sig} at alpha={ALPHA}. "
          f"Correlation does not imply causation.")
    return {"rho": rho, "p": p, "significant": p < ALPHA}