"""
Unit tests for Technique 2: PuLP Integer Linear Programming Staff Scheduling.
Verifies integer constraints, capacity safety margins, budget limits, and infeasibility handling.
"""

import pytest
from src.data.schemas import TimeSlot
from src.models.scheduling import (
    optimize_staff_ilp,
    optimize_staff_wait_time_constrained,
    evaluate_schedule,
    generate_cost_vs_waittime_pareto
)


@pytest.fixture
def sample_slots():
    return [
        TimeSlot("Morning", 9.0, 12.0, arrival_rate=12.0, service_rate=5.0, min_doctors=1, max_doctors=6, cost_per_doctor_hour=100.0, capacity_margin=1.0, baseline_doctors=2),
        TimeSlot("Afternoon", 12.0, 15.0, arrival_rate=18.0, service_rate=5.0, min_doctors=1, max_doctors=6, cost_per_doctor_hour=100.0, capacity_margin=1.0, baseline_doctors=2),
        TimeSlot("Evening", 15.0, 18.0, arrival_rate=14.0, service_rate=5.0, min_doctors=1, max_doctors=6, cost_per_doctor_hour=100.0, capacity_margin=1.0, baseline_doctors=2)
    ]


def test_ilp_optimal_solution_and_integer_values(sample_slots):
    """
    Capacity margin constraints:
    Morning: 12 + 1 = 13 -> ceil(13/5) = 3 docs
    Afternoon: 18 + 1 = 19 -> ceil(19/5) = 4 docs
    Evening: 14 + 1 = 15 -> ceil(15/5) = 3 docs
    Expected optimal allocation: Morning: 3, Afternoon: 4, Evening: 3
    Cost: (3*3*100) + (3*4*100) + (3*3*100) = 3000
    """
    res = optimize_staff_ilp(sample_slots)

    assert res.is_feasible is True
    assert res.status == "Optimal"
    assert res.allocation == {"Morning": 3, "Afternoon": 4, "Evening": 3}
    assert res.total_cost == 3000.0
    assert res.total_doctor_hours == 30.0

    # Ensure all decision variables are integers
    for count in res.allocation.values():
        assert isinstance(count, int)


def test_ilp_respects_workforce_limit(sample_slots):
    """If peak workforce is limited below required staffing, problem must be reported as Infeasible."""
    # Afternoon requires 4 doctors; restrict peak to 3
    res = optimize_staff_ilp(sample_slots, total_workforce_limit=3)

    assert res.is_feasible is False
    assert res.status == "Infeasible"
    assert "Optimization failed" in res.solver_message


def test_ilp_respects_budget_constraint(sample_slots):
    """Optimal solution costs 3000. If budget is 2500, must report Infeasible."""
    res = optimize_staff_ilp(sample_slots, budget_limit=2500.0)

    assert res.is_feasible is False
    assert res.status == "Infeasible"


def test_wait_time_constrained_optimization(sample_slots):
    """Constraining wait time to <= 10 min should allocate extra doctors where needed."""
    res = optimize_staff_wait_time_constrained(sample_slots, max_wait_minutes=10.0)

    assert res.is_feasible is True
    assert res.average_wait_minutes <= 10.0
    # Every slot must achieve wait <= 10 min
    for slot_name, q_metric in res.slot_queueing_metrics.items():
        assert q_metric.is_stable is True
        assert q_metric.Wq_minutes <= 10.0


def test_pareto_frontier_generation(sample_slots):
    """Pareto analysis should produce monotonically non-increasing wait times as cost increases."""
    df_pareto = generate_cost_vs_waittime_pareto(sample_slots)

    assert not df_pareto.empty
    pareto_pts = df_pareto[df_pareto["is_pareto_optimal"]].sort_values(by="total_cost")
    assert len(pareto_pts) >= 2

    # Higher cost must yield lower or equal wait time
    costs = pareto_pts["total_cost"].tolist()
    waits = pareto_pts["avg_wait_minutes"].tolist()
    for i in range(len(waits) - 1):
        assert costs[i] < costs[i + 1]
        assert waits[i] > waits[i + 1]
