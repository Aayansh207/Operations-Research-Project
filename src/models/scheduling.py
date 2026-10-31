"""
Technique 2: Optimal Staff Scheduling using Integer Linear Programming (PuLP).

Mathematical Formulation:
--------------------------
Let T be the set of operational time slots (e.g., Morning, Afternoon, Evening).
For each slot t in T:
- d_t: Slot duration in hours (e.g. 3.0)
- lambda_t: Expected patient arrival rate (patients / hour)
- mu_t: Service rate per doctor (patients / hour / doctor)
- c_min_t, c_max_t: Lower and upper bounds on doctor allocation
- cost_t: Staffing cost per doctor-hour ($)
- delta_t: Required capacity safety margin (patients / hour)

Decision Variables:
- x_t in {c_min_t, ..., c_max_t}, integer: number of doctors assigned to slot t.

Objective Function:
Minimize Total Staffing Cost:
    Z = sum_{t in T} (d_t * cost_t * x_t)

Subject to:
1. Capacity Safety Margin (guarantees rho < 1 with headroom):
    mu_t * x_t >= lambda_t + delta_t   for all t in T
    (Linear integer inequality)

2. Staffing Bounds:
    c_min_t <= x_t <= c_max_t          for all t in T

3. Peak Simultaneous Workforce (optional):
    x_t <= TotalWorkforce              for all t in T

4. Total Staffing Budget (optional):
    sum_{t in T} (d_t * cost_t * x_t) <= Budget

Post-Optimization Evaluation:
The resulting integer schedule {x_t} is subsequently evaluated using the exact
M/M/c queueing theory formulation to quantify expected queue lengths and waiting times.
"""

from typing import List, Dict, Optional, Tuple, Any
import itertools
import pulp
import numpy as np
import pandas as pd

from src.data.schemas import TimeSlot, OptimizationResult, QueueingResult
from src.models.queueing import calculate_mmc


def optimize_staff_ilp(
    slots: List[TimeSlot],
    total_workforce_limit: Optional[int] = None,
    budget_limit: Optional[float] = None,
    solver_timeout: int = 10
) -> OptimizationResult:
    """
    Solves the doctor allocation problem using PuLP with the CBC solver.

    Parameters:
    -----------
    slots : List[TimeSlot]
        List of configured hospital time slots.
    total_workforce_limit : Optional[int]
        Maximum number of doctors available at any single time slot.
    budget_limit : Optional[float]
        Maximum allowed total staffing cost across all slots.
    solver_timeout : int
        Solver timeout in seconds.

    Returns:
    --------
    OptimizationResult:
        Contains optimal allocation, solver status, total cost, and M/M/c metrics.
    """
    if not slots:
        return OptimizationResult(
            status="Infeasible",
            is_feasible=False,
            total_cost=0.0,
            allocation={},
            slot_queueing_metrics={},
            total_doctor_hours=0.0,
            solver_message="Error: No time slots provided for optimization."
        )

    prob = pulp.LpProblem("Hospital_Staff_Allocation", pulp.LpMinimize)

    # Decision variables
    x = {}
    for slot in slots:
        min_bound = max(1, slot.min_doctors)
        max_bound = max(min_bound, slot.max_doctors)
        if total_workforce_limit is not None:
            max_bound = min(max_bound, total_workforce_limit)

        x[slot.name] = pulp.LpVariable(
            f"doctors_{slot.name}",
            lowBound=min_bound,
            upBound=max_bound,
            cat=pulp.LpInteger
        )

    # Objective: Minimize total staffing cost
    cost_terms = [
        slot.duration_hours * slot.cost_per_doctor_hour * x[slot.name]
        for slot in slots
    ]
    prob += pulp.lpSum(cost_terms), "Total_Staffing_Cost"

    # Capacity safety margin constraints: x_t * mu_t >= lambda_t + delta_t
    for slot in slots:
        required_capacity = slot.arrival_rate + slot.capacity_margin
        prob += (
            slot.service_rate * x[slot.name] >= required_capacity,
            f"Capacity_Margin_{slot.name}"
        )

    # Budget constraint if specified
    if budget_limit is not None and budget_limit > 0:
        prob += (
            pulp.lpSum(cost_terms) <= budget_limit,
            "Total_Budget_Limit"
        )

    # Solve using CBC solver
    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=solver_timeout)
    status_code = prob.solve(solver)
    status_str = pulp.LpStatus.get(status_code, "Unknown")

    if status_str != "Optimal":
        # Formulate informative diagnostic message
        infeasibility_reasons = []
        for slot in slots:
            req_min_docs = int(np.ceil((slot.arrival_rate + slot.capacity_margin) / slot.service_rate))
            max_allowed = slot.max_doctors
            if total_workforce_limit is not None:
                max_allowed = min(max_allowed, total_workforce_limit)
            if req_min_docs > max_allowed:
                infeasibility_reasons.append(
                    f"Slot '{slot.name}' requires at least {req_min_docs} doctors to satisfy "
                    f"arrival rate ({slot.arrival_rate}/hr) + margin ({slot.capacity_margin}/hr), "
                    f"but maximum permitted is {max_allowed}."
                )

        diag_msg = (
            f"Optimization failed: Solver returned '{status_str}'. "
            + (" ".join(infeasibility_reasons) if infeasibility_reasons else
               "Check if total budget limit is too low for the required staffing.")
        )

        return OptimizationResult(
            status=status_str,
            is_feasible=False,
            total_cost=0.0,
            allocation={s.name: 0 for s in slots},
            slot_queueing_metrics={},
            total_doctor_hours=0.0,
            solver_message=diag_msg
        )

    # Optimal solution extracted
    allocation = {slot.name: int(round(pulp.value(x[slot.name]))) for slot in slots}
    return evaluate_schedule(slots, allocation, status="Optimal", solver_message="Integer Linear Program solved to optimality.")


def optimize_staff_wait_time_constrained(
    slots: List[TimeSlot],
    max_wait_minutes: float = 15.0,
    total_workforce_limit: Optional[int] = None,
    budget_limit: Optional[float] = None
) -> OptimizationResult:
    """
    Optimizes doctor allocation such that the steady-state expected waiting time
    W_q in every slot does not exceed max_wait_minutes.

    Approach:
    1. For each slot, calculate the minimum number of doctors required so that:
       a) Queue is stable (rho < 1)
       b) Safety margin is met
       c) W_q <= max_wait_minutes
    2. Enforce these strengthened minimum staffing bounds in the ILP model.
    3. Solve for minimum cost subject to workforce and budget limits.
    """
    modified_slots = []
    diagnostic_notes = []

    for slot in slots:
        # Search for minimum c where Wq <= max_wait_minutes
        min_c_margin = int(np.ceil((slot.arrival_rate + slot.capacity_margin) / slot.service_rate))
        c_search = max(1, slot.min_doctors, min_c_margin)
        max_search = slot.max_doctors
        if total_workforce_limit is not None:
            max_search = min(max_search, total_workforce_limit)

        chosen_c = None
        for c in range(c_search, max_search + 1):
            q_res = calculate_mmc(slot.arrival_rate, slot.service_rate, c)
            if q_res.is_stable and q_res.Wq_minutes <= max_wait_minutes:
                chosen_c = c
                break

        if chosen_c is None:
            # Even with max allowed doctors, wait time exceeds target
            max_q = calculate_mmc(slot.arrival_rate, slot.service_rate, max_search)
            wait_achieved = f"{max_q.Wq_minutes:.1f} min" if max_q.is_stable else "Unstable"
            diagnostic_notes.append(
                f"For slot '{slot.name}', target wait <= {max_wait_minutes} min is unreachable "
                f"within doctor limit of {max_search} (best achieved: {wait_achieved})."
            )
            chosen_c = max_search

        # Create updated slot with tightened lower bound
        new_slot = TimeSlot(
            name=slot.name,
            start_hour=slot.start_hour,
            end_hour=slot.end_hour,
            arrival_rate=slot.arrival_rate,
            service_rate=slot.service_rate,
            min_doctors=chosen_c,
            max_doctors=slot.max_doctors,
            cost_per_doctor_hour=slot.cost_per_doctor_hour,
            capacity_margin=slot.capacity_margin,
            baseline_doctors=slot.baseline_doctors
        )
        modified_slots.append(new_slot)

    # Solve ILP with wait-time bounds
    res = optimize_staff_ilp(modified_slots, total_workforce_limit, budget_limit)
    if diagnostic_notes:
        res.solver_message += " Note: " + " ".join(diagnostic_notes)
    return res


def evaluate_schedule(
    slots: List[TimeSlot],
    allocation: Dict[str, int],
    status: str = "Evaluated",
    solver_message: str = "Schedule evaluated."
) -> OptimizationResult:
    """
    Evaluates any given doctor allocation using M/M/c queueing theory and cost calculations.
    Computes arrival-weighted average waiting time.
    """
    slot_queueing = {}
    total_cost = 0.0
    total_doctor_hours = 0.0
    weighted_wait_sum = 0.0
    total_arrivals = 0.0

    all_stable = True
    for slot in slots:
        docs = allocation.get(slot.name, slot.baseline_doctors)
        q_res = calculate_mmc(slot.arrival_rate, slot.service_rate, docs)
        slot_queueing[slot.name] = q_res

        hours = slot.duration_hours * docs
        total_doctor_hours += hours
        total_cost += hours * slot.cost_per_doctor_hour

        slot_arrivals = slot.arrival_rate * slot.duration_hours
        total_arrivals += slot_arrivals

        if q_res.is_stable:
            weighted_wait_sum += q_res.Wq_minutes * slot_arrivals
        else:
            all_stable = False

    avg_wait = (weighted_wait_sum / total_arrivals) if (all_stable and total_arrivals > 0) else float("inf")

    return OptimizationResult(
        status=status,
        is_feasible=all_stable,
        total_cost=total_cost,
        allocation=allocation,
        slot_queueing_metrics=slot_queueing,
        total_doctor_hours=total_doctor_hours,
        solver_message=solver_message,
        average_wait_minutes=avg_wait
    )


def generate_cost_vs_waittime_pareto(
    slots: List[TimeSlot],
    total_workforce_limit: Optional[int] = None
) -> pd.DataFrame:
    """
    Enumerates feasible integer allocations across time slots to map the exact
    trade-off (Pareto frontier) between staffing expenditure and patient wait time.
    """
    slot_ranges = []
    for slot in slots:
        min_c = max(1, slot.min_doctors, int(np.ceil((slot.arrival_rate + 0.1) / slot.service_rate)))
        max_c = slot.max_doctors
        if total_workforce_limit is not None:
            max_c = min(max_c, total_workforce_limit)
        slot_ranges.append([(slot.name, c) for c in range(min_c, max_c + 1)])

    records = []
    for combo in itertools.product(*slot_ranges):
        alloc = dict(combo)
        eval_res = evaluate_schedule(slots, alloc)
        if eval_res.is_feasible and eval_res.average_wait_minutes < 180.0:
            records.append({
                "allocation_label": ", ".join(f"{k}:{v}" for k, v in alloc.items()),
                "total_cost": eval_res.total_cost,
                "total_doctor_hours": eval_res.total_doctor_hours,
                "avg_wait_minutes": eval_res.average_wait_minutes,
                **{f"docs_{k}": v for k, v in alloc.items()}
            })

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values(by=["total_cost", "avg_wait_minutes"]).reset_index(drop=True)
        # Mark Pareto efficient points
        pareto_flags = []
        min_wait_seen = float("inf")
        for wait in df["avg_wait_minutes"]:
            if wait < min_wait_seen:
                pareto_flags.append(True)
                min_wait_seen = wait
            else:
                pareto_flags.append(False)
        df["is_pareto_optimal"] = pareto_flags
    return df
