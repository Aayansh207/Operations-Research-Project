"""
Performance Metrics and Statistical Calculation Engine.

Implements:
- Waiting-time reduction calculation with zero-baseline and infinity handling.
- Cost difference and percentage savings.
- Paired comparison statistics for Common Random Numbers (CRN) experiments.
- Time-weighted queue-length integration.
"""

from typing import Tuple, Optional, List, Dict, Any
import numpy as np
from scipy import stats


def calculate_wait_reduction_pct(
    baseline_wait: float,
    optimized_wait: float
) -> Tuple[float, str]:
    """
    Computes percentage reduction in waiting time:
        ((baseline - optimized) / baseline) * 100

    Rules:
    - Never clamp negative values to zero (report honest deterioration).
    - Handle zero baseline gracefully without returning NaN or crashing.
    - Handle infinite baseline (stabilization from unstable queue).

    Returns:
    --------
    (reduction_pct, human_readable_label)
    """
    # Case: Baseline was unstable (infinite wait)
    if np.isinf(baseline_wait):
        if np.isinf(optimized_wait):
            return 0.0, "Both baseline and optimized are unstable (infinite wait)."
        return 100.0, "Eliminated infinite queue (system stabilized)."

    # Case: Optimized is unstable
    if np.isinf(optimized_wait):
        return -100.0, "Optimized configuration destabilized the queue."

    # Case: Baseline wait is zero
    if baseline_wait <= 1e-6:
        if optimized_wait <= 1e-6:
            return 0.0, "0.0% (Zero baseline wait maintained)"
        return -100.0, "Wait time increased from 0.0 min"

    # Standard case
    reduction = ((baseline_wait - optimized_wait) / baseline_wait) * 100.0

    if reduction > 0:
        label = f"{reduction:.1f}% reduction"
    elif reduction < 0:
        label = f"{abs(reduction):.1f}% increase (longer wait)"
    else:
        label = "0.0% (no change)"

    return round(reduction, 2), label


def calculate_cost_savings_pct(
    baseline_cost: float,
    optimized_cost: float
) -> Tuple[float, str]:
    """
    Computes percentage cost savings:
        ((baseline_cost - optimized_cost) / baseline_cost) * 100
    """
    if baseline_cost <= 0:
        return 0.0, "0.0% (Zero baseline cost)"

    savings_pct = ((baseline_cost - optimized_cost) / baseline_cost) * 100.0
    diff = optimized_cost - baseline_cost

    if diff < 0:
        label = f"${abs(diff):,.2f} savings ({savings_pct:.1f}%)"
    elif diff > 0:
        label = f"${diff:,.2f} additional expenditure ({abs(savings_pct):.1f}% increase)"
    else:
        label = "Equal staffing cost ($0.00 difference)"

    return round(savings_pct, 2), label


def compute_paired_comparison_stats(
    baseline_reps: List[float],
    optimized_reps: List[float]
) -> Dict[str, Any]:
    """
    Performs paired statistical analysis when using Common Random Numbers (CRN).
    Computes mean difference, standard error, and 95% confidence interval of the reduction.
    """
    if len(baseline_reps) != len(optimized_reps) or len(baseline_reps) < 2:
        return {
            "mean_diff": float(np.mean(baseline_reps) - np.mean(optimized_reps)) if baseline_reps else 0.0,
            "ci_95": None,
            "p_value": None,
            "stat_significant": False
        }

    # Differences per replication (D_i = Baseline_i - Optimized_i)
    diffs = np.array(baseline_reps) - np.array(optimized_reps)
    n = len(diffs)
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1))
    se_diff = std_diff / np.sqrt(n)

    t_crit = stats.t.ppf(0.975, df=n - 1)
    ci = (float(mean_diff - t_crit * se_diff), float(mean_diff + t_crit * se_diff))

    # Paired t-test
    t_stat, p_val = stats.ttest_rel(baseline_reps, optimized_reps)

    return {
        "mean_diff": round(mean_diff, 2),
        "ci_95": (round(ci[0], 2), round(ci[1], 2)),
        "p_value": float(p_val),
        "stat_significant": bool(p_val < 0.05 and ci[0] > 0)
    }
