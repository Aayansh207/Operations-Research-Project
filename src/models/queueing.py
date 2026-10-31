"""
Technique 1: Queueing Theory (M/M/c Analytical Model).

Assumptions of M/M/c:
1. Patient arrivals follow a stationary Poisson process with rate lambda.
2. Service times follow an exponential distribution with mean 1/mu per doctor.
3. c identical, parallel doctors (servers).
4. First-Come, First-Served (FCFS) queue discipline.
5. Infinite queue capacity and infinite calling population.
6. Stable steady-state operation requires rho = lambda / (c * mu) < 1.
7. No patient abandonment or reneging in this basic analytical formulation.

Note: In hospitals, actual arrival patterns may deviate from pure Poisson processes,
and service times can be log-normal or Erlang distributed. We evaluate these
extensions using discrete-event simulation in Technique 3.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np
from src.data.schemas import QueueingResult


def calculate_mmc(
    arrival_rate: float,
    service_rate: float,
    num_servers: int
) -> QueueingResult:
    """
    Computes exact steady-state performance metrics for an M/M/c queue.

    Parameters:
    -----------
    arrival_rate : float
        Mean patient arrival rate (lambda), in patients per hour. Must be >= 0.
    service_rate : float
        Mean service rate per server (mu), in patients per hour. Must be > 0.
    num_servers : int
        Number of identical servers/doctors (c). Must be an integer >= 1.

    Returns:
    --------
    QueueingResult:
        Structured object containing all analytical metrics (Wq, W, Lq, L, rho, P0, Pw)
        and stability status.
    """
    if service_rate <= 0:
        raise ValueError(f"Service rate (mu) must be strictly positive. Got: {service_rate}")
    if num_servers < 1:
        raise ValueError(f"Number of servers (c) must be an integer >= 1. Got: {num_servers}")
    if arrival_rate < 0:
        raise ValueError(f"Arrival rate (lambda) cannot be negative. Got: {arrival_rate}")

    c = int(num_servers)
    lam = float(arrival_rate)
    mu = float(service_rate)

    # Edge case: zero arrivals
    if lam == 0.0:
        return QueueingResult(
            arrival_rate=0.0,
            service_rate=mu,
            num_servers=c,
            traffic_intensity=0.0,
            utilization=0.0,
            is_stable=True,
            p0=1.0,
            p_wait=0.0,
            Lq=0.0,
            L=0.0,
            Wq_hours=0.0,
            Wq_minutes=0.0,
            W_hours=0.0,
            W_minutes=0.0,
            status_message="Zero arrival rate: Queue is empty and waiting time is 0."
        )

    # Traffic intensity and utilization
    a = lam / mu  # Offered load / traffic intensity
    rho = lam / (c * mu)  # Server utilization

    # Stability check: rho must be strictly less than 1
    if rho >= 1.0:
        return QueueingResult(
            arrival_rate=lam,
            service_rate=mu,
            num_servers=c,
            traffic_intensity=a,
            utilization=rho,
            is_stable=False,
            p0=0.0,
            p_wait=1.0,
            Lq=float("inf"),
            L=float("inf"),
            Wq_hours=float("inf"),
            Wq_minutes=float("inf"),
            W_hours=float("inf"),
            W_minutes=float("inf"),
            status_message=(
                f"UNSTABLE QUEUE: Server utilization is {rho * 100:.1f}% (rho >= 1.0). "
                f"Arrival rate ({lam:.1f}/hr) exceeds or equals total service capacity "
                f"({c * mu:.1f}/hr). Steady-state waiting time is infinite. "
                f"Allocate at least {int(np.ceil(lam / mu + 1e-6))} doctors."
            )
        )

    # Numerically stable computation of P0 and Erlang C (P_wait)
    # T_n = a^n / n! computed iteratively to prevent factorial overflow
    sum_terms = 0.0
    current_term = 1.0  # n = 0: a^0 / 0! = 1.0
    sum_terms += current_term

    for n in range(1, c):
        current_term *= (a / n)
        sum_terms += current_term

    # term for n = c: a^c / c!
    term_c = current_term * (a / c)
    tail_term = term_c / (1.0 - rho)

    # Empty system probability P0
    p0 = 1.0 / (sum_terms + tail_term)

    # Erlang C formula: probability an arriving patient must wait
    p_wait = tail_term * p0

    # Average queue length Lq (number of patients in waiting room)
    Lq = (p_wait * rho) / (1.0 - rho)

    # Little's Law: Wq = Lq / lambda
    Wq_hours = Lq / lam
    Wq_minutes = Wq_hours * 60.0

    # Total time in system: W = Wq + 1 / mu
    W_hours = Wq_hours + (1.0 / mu)
    W_minutes = W_hours * 60.0

    # Average number of patients in system: L = lambda * W = Lq + a
    L = lam * W_hours

    return QueueingResult(
        arrival_rate=lam,
        service_rate=mu,
        num_servers=c,
        traffic_intensity=a,
        utilization=rho,
        is_stable=True,
        p0=p0,
        p_wait=p_wait,
        Lq=Lq,
        L=L,
        Wq_hours=Wq_hours,
        Wq_minutes=Wq_minutes,
        W_hours=W_hours,
        W_minutes=W_minutes,
        status_message="Stable steady-state M/M/c operation."
    )


def calculate_mmc_sensitivity_servers(
    arrival_rate: float,
    service_rate: float,
    min_servers: int = 1,
    max_servers: int = 10
) -> pd.DataFrame:
    """
    Evaluates queueing performance across a range of doctor counts.
    Returns a DataFrame suitable for plotting waiting time vs number of doctors.
    """
    rows = []
    for c in range(min_servers, max_servers + 1):
        res = calculate_mmc(arrival_rate, service_rate, c)
        rows.append({
            "servers": c,
            "utilization": res.utilization,
            "is_stable": res.is_stable,
            "waiting_time_min": res.Wq_minutes if res.is_stable else np.nan,
            "system_time_min": res.W_minutes if res.is_stable else np.nan,
            "queue_length": res.Lq if res.is_stable else np.nan,
            "prob_wait": res.p_wait if res.is_stable else 1.0,
            "capacity": c * service_rate,
            "status": "Stable" if res.is_stable else "Unstable (rho >= 1.0)"
        })
    return pd.DataFrame(rows)


def calculate_mmc_sensitivity_arrivals(
    min_lambda: float,
    max_lambda: float,
    service_rate: float,
    num_servers: int,
    num_points: int = 25
) -> pd.DataFrame:
    """
    Evaluates queueing performance as patient arrival rate increases towards capacity.
    """
    capacity = num_servers * service_rate
    lambdas = np.linspace(min_lambda, max_lambda, num_points)
    rows = []
    for lam in lambdas:
        res = calculate_mmc(lam, service_rate, num_servers)
        rows.append({
            "arrival_rate": lam,
            "capacity": capacity,
            "utilization": res.utilization,
            "is_stable": res.is_stable,
            "waiting_time_min": res.Wq_minutes if res.is_stable else np.nan,
            "queue_length": res.Lq if res.is_stable else np.nan,
            "prob_wait": res.p_wait if res.is_stable else 1.0,
            "status": "Stable" if res.is_stable else "Congested / Unstable"
        })
    return pd.DataFrame(rows)
