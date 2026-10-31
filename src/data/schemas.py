"""
Data schemas and type definitions for SmartCare OR.
All time units are explicitly documented.
Internally, arrival and service rates are in patients/hour.
Waiting times are calculated in hours and converted to minutes for reporting.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import numpy as np


@dataclass
class TimeSlot:
    """Represents a discrete operational hospital shift/slot."""
    name: str
    start_hour: float  # e.g. 9.0 for 09:00
    end_hour: float    # e.g. 12.0 for 12:00
    arrival_rate: float  # lambda: patients per hour
    service_rate: float  # mu: patients per hour per doctor
    min_doctors: int = 1
    max_doctors: int = 10
    cost_per_doctor_hour: float = 120.0  # Synthetic hourly rate ($)
    capacity_margin: float = 1.0  # delta: required surplus capacity (patients/hr)
    baseline_doctors: int = 2

    @property
    def duration_hours(self) -> float:
        return self.end_hour - self.start_hour


@dataclass
class QueueingResult:
    """Exact analytical results from M/M/c queueing theory."""
    arrival_rate: float  # lambda (patients/hour)
    service_rate: float  # mu (patients/hour/server)
    num_servers: int     # c
    traffic_intensity: float  # a = lambda / mu
    utilization: float        # rho = lambda / (c * mu)
    is_stable: bool
    p0: float            # Empty-system probability
    p_wait: float        # Erlang C probability of waiting
    Lq: float            # Average queue length (patients)
    L: float             # Average number of patients in system
    Wq_hours: float      # Average waiting time in queue (hours)
    Wq_minutes: float    # Average waiting time in queue (minutes)
    W_hours: float       # Average time in system (hours)
    W_minutes: float     # Average time in system (minutes)
    status_message: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "arrival_rate": self.arrival_rate,
            "service_rate": self.service_rate,
            "num_servers": self.num_servers,
            "traffic_intensity": self.traffic_intensity,
            "utilization": self.utilization,
            "is_stable": self.is_stable,
            "p0": self.p0,
            "p_wait": self.p_wait,
            "Lq": self.Lq,
            "L": self.L,
            "Wq_hours": self.Wq_hours,
            "Wq_minutes": self.Wq_minutes,
            "W_hours": self.W_hours,
            "W_minutes": self.W_minutes,
            "status_message": self.status_message,
        }


@dataclass
class OptimizationResult:
    """Result from Integer Linear Programming (PuLP) staff allocation."""
    status: str
    is_feasible: bool
    total_cost: float
    allocation: Dict[str, int]
    slot_queueing_metrics: Dict[str, QueueingResult]
    total_doctor_hours: float
    solver_message: str
    baseline_cost: Optional[float] = None
    cost_difference: Optional[float] = None
    average_wait_minutes: Optional[float] = None


@dataclass
class PatientRecord:
    """Detailed event record for an individual simulated patient."""
    patient_id: int
    slot_name: str
    arrival_time: float      # Simulation time in hours
    service_start_time: Optional[float] = None
    service_end_time: Optional[float] = None
    waiting_time_hours: Optional[float] = None
    waiting_time_minutes: Optional[float] = None
    service_duration_hours: Optional[float] = None
    service_duration_minutes: Optional[float] = None
    total_system_time_hours: Optional[float] = None
    total_system_time_minutes: Optional[float] = None
    completed: bool = False
    in_warm_up: bool = False


@dataclass
class SimulationMetrics:
    """Comprehensive performance metrics from discrete-event simulation."""
    mean_wait_minutes: float
    median_wait_minutes: float
    p90_wait_minutes: float
    mean_system_time_minutes: float
    time_weighted_queue_length: float
    server_utilization: float
    patients_arrived: int
    patients_served: int
    patients_waiting_at_end: int
    patients_in_service_at_end: int
    duration_hours: float
    warm_up_hours: float
    num_replications: int
    replications_mean_wait: List[float] = field(default_factory=list)
    confidence_interval_95: Optional[tuple] = None  # (lower, upper)
