"""
Predefined hospital operational scenarios and scenario builders.
Supports reproducible experiments across Low, Moderate, and Peak demand levels.
All scenarios use synthetic parameters designed for academic evaluation.
"""

from typing import Dict, List
from src.data.schemas import TimeSlot


def get_predefined_scenarios() -> Dict[str, Dict]:
    """
    Returns standard benchmark scenarios configured for hospital outpatient departments.
    """
    return {
        "Scenario A: Low Demand": {
            "description": "Low patient demand with modest congestion. Highlights staffing efficiency.",
            "slots": [
                TimeSlot(
                    name="Morning",
                    start_hour=9.0,
                    end_hour=12.0,
                    arrival_rate=7.0,
                    service_rate=5.0,
                    min_doctors=1,
                    max_doctors=5,
                    cost_per_doctor_hour=120.0,
                    capacity_margin=1.0,
                    baseline_doctors=2
                ),
                TimeSlot(
                    name="Afternoon",
                    start_hour=12.0,
                    end_hour=15.0,
                    arrival_rate=9.0,
                    service_rate=5.0,
                    min_doctors=1,
                    max_doctors=5,
                    cost_per_doctor_hour=120.0,
                    capacity_margin=1.0,
                    baseline_doctors=2
                ),
                TimeSlot(
                    name="Evening",
                    start_hour=15.0,
                    end_hour=18.0,
                    arrival_rate=8.0,
                    service_rate=5.0,
                    min_doctors=1,
                    max_doctors=5,
                    cost_per_doctor_hour=120.0,
                    capacity_margin=1.0,
                    baseline_doctors=2
                ),
            ],
            "total_workforce_limit": 6,
            "budget_limit": 2500.0,
        },
        "Scenario B: Moderate Demand (Standard)": {
            "description": "Standard hospital workflow with afternoon surge. Demonstrates queueing instability in baseline.",
            "slots": [
                TimeSlot(
                    name="Morning",
                    start_hour=9.0,
                    end_hour=12.0,
                    arrival_rate=14.0,
                    service_rate=5.0,
                    min_doctors=1,
                    max_doctors=7,
                    cost_per_doctor_hour=120.0,
                    capacity_margin=1.0,
                    baseline_doctors=3
                ),
                TimeSlot(
                    name="Afternoon",
                    start_hour=12.0,
                    end_hour=15.0,
                    arrival_rate=18.0,
                    service_rate=5.0,
                    min_doctors=1,
                    max_doctors=7,
                    cost_per_doctor_hour=120.0,
                    capacity_margin=1.0,
                    baseline_doctors=3  # Note: 3 * 5 = 15 < 18 (Unstable in baseline!)
                ),
                TimeSlot(
                    name="Evening",
                    start_hour=15.0,
                    end_hour=18.0,
                    arrival_rate=15.0,
                    service_rate=5.0,
                    min_doctors=1,
                    max_doctors=7,
                    cost_per_doctor_hour=120.0,
                    capacity_margin=1.0,
                    baseline_doctors=3
                ),
            ],
            "total_workforce_limit": 8,
            "budget_limit": 5000.0,
        },
        "Scenario C: Peak / Congested Demand": {
            "description": "Heavy emergency/outpatient influx testing capacity limits and budget trade-offs.",
            "slots": [
                TimeSlot(
                    name="Morning",
                    start_hour=9.0,
                    end_hour=12.0,
                    arrival_rate=20.0,
                    service_rate=5.0,
                    min_doctors=2,
                    max_doctors=9,
                    cost_per_doctor_hour=120.0,
                    capacity_margin=1.5,
                    baseline_doctors=4
                ),
                TimeSlot(
                    name="Afternoon",
                    start_hour=12.0,
                    end_hour=15.0,
                    arrival_rate=26.0,
                    service_rate=5.0,
                    min_doctors=2,
                    max_doctors=9,
                    cost_per_doctor_hour=120.0,
                    capacity_margin=1.5,
                    baseline_doctors=4  # Note: 4 * 5 = 20 < 26 (Severely unstable in baseline!)
                ),
                TimeSlot(
                    name="Evening",
                    start_hour=15.0,
                    end_hour=18.0,
                    arrival_rate=22.0,
                    service_rate=5.0,
                    min_doctors=2,
                    max_doctors=9,
                    cost_per_doctor_hour=120.0,
                    capacity_margin=1.5,
                    baseline_doctors=4
                ),
            ],
            "total_workforce_limit": 10,
            "budget_limit": 7500.0,
        }
    }


def create_custom_scenario(
    name: str,
    slots: List[TimeSlot],
    total_workforce_limit: int = 10,
    budget_limit: float = 5000.0
) -> Dict:
    """Helper to bundle custom user-configured slots into a scenario dict."""
    return {
        "description": f"Custom user scenario: {name}",
        "slots": slots,
        "total_workforce_limit": total_workforce_limit,
        "budget_limit": budget_limit
    }
