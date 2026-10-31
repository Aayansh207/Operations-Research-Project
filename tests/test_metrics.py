"""
Unit tests for Performance Metrics and Statistical Reporting.
Verifies wait-time reduction formulas, edge cases (zero baseline, unstable baseline), and paired stats.
"""

import pytest
import numpy as np
from src.experiments.metrics import (
    calculate_wait_reduction_pct,
    calculate_cost_savings_pct,
    compute_paired_comparison_stats
)


def test_wait_reduction_standard_case():
    """Reduction = ((baseline - optimized) / baseline) * 100"""
    reduc, label = calculate_wait_reduction_pct(baseline_wait=20.0, optimized_wait=5.0)
    assert reduc == 75.0
    assert "75.0% reduction" in label


def test_wait_reduction_deterioration_not_clamped():
    """Negative reduction (longer wait time) must NOT be clamped to zero."""
    reduc, label = calculate_wait_reduction_pct(baseline_wait=10.0, optimized_wait=15.0)
    assert reduc == -50.0
    assert "increase (longer wait)" in label


def test_wait_reduction_zero_baseline():
    """Zero baseline wait time must not produce division by zero or NaN."""
    reduc, label = calculate_wait_reduction_pct(baseline_wait=0.0, optimized_wait=0.0)
    assert reduc == 0.0
    assert not np.isnan(reduc)


def test_wait_reduction_unstable_baseline():
    """When baseline was unstable (infinite wait) and optimized is stable, return 100% / stabilized."""
    reduc, label = calculate_wait_reduction_pct(baseline_wait=float("inf"), optimized_wait=8.5)
    assert reduc == 100.0
    assert "Eliminated infinite queue" in label


def test_cost_savings_calculation():
    """Cost savings = ((baseline - opt) / baseline) * 100"""
    savings, label = calculate_cost_savings_pct(baseline_cost=4000.0, optimized_cost=3000.0)
    assert savings == 25.0
    assert "$1,000.00 savings" in label

    # Extra expenditure case
    savings_neg, label_neg = calculate_cost_savings_pct(baseline_cost=3000.0, optimized_cost=3600.0)
    assert savings_neg == -20.0
    assert "$600.00 additional expenditure" in label_neg


def test_paired_comparison_stats():
    """Paired t-test statistics across replications."""
    base_reps = [25.0, 28.0, 30.0, 22.0]
    opt_reps = [10.0, 12.0, 11.0, 9.0]
    stats = compute_paired_comparison_stats(base_reps, opt_reps)

    assert stats["mean_diff"] > 0
    assert stats["ci_95"] is not None
    ci_low, ci_high = stats["ci_95"]
    assert ci_low <= stats["mean_diff"] <= ci_high
    assert stats["stat_significant"] is True
