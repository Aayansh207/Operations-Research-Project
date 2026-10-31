"""
Unit tests for Technique 1: M/M/c Queueing Theory Model.
Verifies mathematical correctness against exact analytical formulas and edge cases.
"""

import pytest
import numpy as np
from src.models.queueing import (
    calculate_mmc,
    calculate_mmc_sensitivity_servers,
    calculate_mmc_sensitivity_arrivals
)


def test_mm1_special_case_exact_match():
    """
    For c = 1, M/M/1 queue metrics must match standard closed-form equations:
    Wq = lambda / (mu * (mu - lambda))
    W  = 1 / (mu - lambda)
    Lq = lambda^2 / (mu * (mu - lambda))
    L  = lambda / (mu - lambda)
    rho = lambda / mu
    """
    lam = 4.0
    mu = 5.0
    c = 1

    res = calculate_mmc(arrival_rate=lam, service_rate=mu, num_servers=c)

    expected_rho = lam / mu  # 0.8
    expected_Wq_hours = lam / (mu * (mu - lam))  # 4 / (5 * 1) = 0.8 hours
    expected_Wq_minutes = expected_Wq_hours * 60.0  # 48.0 min
    expected_W_hours = 1.0 / (mu - lam)  # 1.0 hour
    expected_W_minutes = 60.0  # 60.0 min
    expected_Lq = (lam ** 2) / (mu * (mu - lam))  # 16 / 5 = 3.2
    expected_L = lam / (mu - lam)  # 4 / 1 = 4.0

    assert res.is_stable is True
    assert pytest.approx(res.utilization, rel=1e-5) == expected_rho
    assert pytest.approx(res.Wq_minutes, rel=1e-5) == expected_Wq_minutes
    assert pytest.approx(res.W_minutes, rel=1e-5) == expected_W_minutes
    assert pytest.approx(res.Lq, rel=1e-5) == expected_Lq
    assert pytest.approx(res.L, rel=1e-5) == expected_L
    assert pytest.approx(res.p_wait, rel=1e-5) == expected_rho


def test_mmc_known_analytical_values():
    """
    Verifies M/M/2 queue against standard published benchmark:
    lambda = 8, mu = 5, c = 2:
    rho = 8 / 10 = 0.8
    P0 = 1 / 9 = 0.111111...
    Pw = (6.4 / 9) = 0.711111...
    Lq = (6.4 / 9) * 0.8 / 0.2 = 2.84444...
    Wq = 2.84444... / 8 = 0.355555... hr = 21.3333... min
    """
    res = calculate_mmc(arrival_rate=8.0, service_rate=5.0, num_servers=2)

    assert res.is_stable is True
    assert pytest.approx(res.utilization, rel=1e-4) == 0.8
    assert pytest.approx(res.p0, rel=1e-4) == (1.0 / 9.0)
    assert pytest.approx(res.p_wait, rel=1e-4) == (6.4 / 9.0)
    assert pytest.approx(res.Lq, rel=1e-4) == (25.6 / 9.0)
    assert pytest.approx(res.Wq_minutes, rel=1e-4) == 21.333333


def test_unstable_queue_handling():
    """
    When lambda >= c * mu, the queue is unstable.
    Must return is_stable=False, infinite wait times, and descriptive message.
    """
    # Overloaded: lambda = 12, capacity = 2 * 5 = 10 (rho = 1.2)
    res = calculate_mmc(arrival_rate=12.0, service_rate=5.0, num_servers=2)
    assert res.is_stable is False
    assert res.utilization == 1.2
    assert np.isinf(res.Wq_minutes)
    assert np.isinf(res.Lq)
    assert "UNSTABLE QUEUE" in res.status_message

    # Boundary: lambda = 10, capacity = 10 (rho = 1.0)
    res_bound = calculate_mmc(arrival_rate=10.0, service_rate=5.0, num_servers=2)
    assert res_bound.is_stable is False
    assert res_bound.utilization == 1.0


def test_zero_arrivals():
    """When arrival rate is zero, waiting time and queue length must be zero."""
    res = calculate_mmc(arrival_rate=0.0, service_rate=5.0, num_servers=2)
    assert res.is_stable is True
    assert res.Wq_minutes == 0.0
    assert res.Lq == 0.0
    assert res.utilization == 0.0
    assert res.p0 == 1.0
    assert res.p_wait == 0.0


def test_monotonicity_additional_doctors():
    """Adding servers to a stable queue must strictly decrease average waiting time."""
    lam = 15.0
    mu = 5.0
    w_3 = calculate_mmc(lam, mu, 4).Wq_minutes
    w_4 = calculate_mmc(lam, mu, 5).Wq_minutes
    w_5 = calculate_mmc(lam, mu, 6).Wq_minutes

    assert w_3 > w_4 > w_5 > 0.0


def test_monotonicity_increasing_arrivals():
    """Increasing arrival rate with fixed servers must strictly increase waiting time."""
    mu = 5.0
    c = 4
    w_low = calculate_mmc(10.0, mu, c).Wq_minutes
    w_med = calculate_mmc(14.0, mu, c).Wq_minutes
    w_high = calculate_mmc(17.0, mu, c).Wq_minutes

    assert w_low < w_med < w_high


def test_numerical_stability_large_c():
    """Model must compute without factorial overflow for larger server counts (e.g. c = 30)."""
    lam = 100.0
    mu = 4.0
    c = 30  # Capacity = 120, rho = 100/120 = 0.833

    res = calculate_mmc(arrival_rate=lam, service_rate=mu, num_servers=c)
    assert res.is_stable is True
    assert res.Wq_minutes > 0.0
    assert not np.isnan(res.Wq_minutes)
    assert not np.isinf(res.Wq_minutes)


def test_invalid_input_validation():
    """Model must validate input bounds and raise descriptive ValueErrors."""
    with pytest.raises(ValueError, match="strictly positive"):
        calculate_mmc(arrival_rate=5.0, service_rate=0.0, num_servers=2)

    with pytest.raises(ValueError, match="strictly positive"):
        calculate_mmc(arrival_rate=5.0, service_rate=-2.0, num_servers=2)

    with pytest.raises(ValueError, match="integer >= 1"):
        calculate_mmc(arrival_rate=5.0, service_rate=5.0, num_servers=0)

    with pytest.raises(ValueError, match="cannot be negative"):
        calculate_mmc(arrival_rate=-1.0, service_rate=5.0, num_servers=2)
