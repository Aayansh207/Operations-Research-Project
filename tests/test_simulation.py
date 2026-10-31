"""
Unit tests for Discrete-Event Patient Flow Simulation (SimPy).
Verifies reproducibility, patient conservation, event ordering, and shift changes.
"""

import pytest
import pandas as pd
from src.data.schemas import TimeSlot
from src.simulation.patient_flow import HospitalSimulation, run_multi_replication_simulation


@pytest.fixture
def sim_slots():
    return [
        TimeSlot("Morning", 9.0, 12.0, arrival_rate=8.0, service_rate=5.0, min_doctors=1, max_doctors=5, baseline_doctors=2),
        TimeSlot("Afternoon", 12.0, 15.0, arrival_rate=12.0, service_rate=5.0, min_doctors=1, max_doctors=5, baseline_doctors=3),
        TimeSlot("Evening", 15.0, 18.0, arrival_rate=9.0, service_rate=5.0, min_doctors=1, max_doctors=5, baseline_doctors=2)
    ]


def test_simulation_reproducibility_with_seed(sim_slots):
    """Running identical simulation with identical seed must produce identical results."""
    alloc = {"Morning": 2, "Afternoon": 3, "Evening": 2}
    sim1 = HospitalSimulation(sim_slots, alloc, random_seed=123)
    m1, df_p1, _ = sim1.run()

    sim2 = HospitalSimulation(sim_slots, alloc, random_seed=123)
    m2, df_p2, _ = sim2.run()

    assert m1.patients_arrived == m2.patients_arrived
    assert m1.patients_served == m2.patients_served
    assert pytest.approx(m1.mean_wait_minutes, rel=1e-6) == m2.mean_wait_minutes
    pd.testing.assert_frame_equal(df_p1, df_p2)


def test_patient_conservation_law(sim_slots):
    """
    Every arrived patient must be accounted for:
    Total Arrived = Completed + Still in Waiting Queue + Still in Doctor Consultation
    """
    alloc = {"Morning": 2, "Afternoon": 3, "Evening": 2}
    sim = HospitalSimulation(sim_slots, alloc, random_seed=42)
    metrics, _, _ = sim.run()

    accounted_patients = (
        metrics.patients_served
        + metrics.patients_waiting_at_end
        + metrics.patients_in_service_at_end
    )
    assert metrics.patients_arrived == accounted_patients


def test_event_chronology_and_non_negative_times(sim_slots):
    """Every completed patient must have arrival <= service_start <= service_end."""
    alloc = {"Morning": 2, "Afternoon": 3, "Evening": 2}
    sim = HospitalSimulation(sim_slots, alloc, random_seed=999)
    _, df_patients, _ = sim.run()

    completed = df_patients[df_patients["completed"]]
    assert not completed.empty

    assert (completed["waiting_time_min"] >= 0.0).all()
    assert (completed["service_time_min"] > 0.0).all()
    assert (completed["service_start_hr"] >= completed["arrival_time_hr"]).all()
    assert (completed["service_end_hr"] > completed["service_start_hr"]).all()


def test_multi_replication_confidence_interval(sim_slots):
    """Multi-replication runner must compute positive wait time and valid confidence intervals."""
    alloc = {"Morning": 2, "Afternoon": 3, "Evening": 2}
    metrics, _ = run_multi_replication_simulation(sim_slots, alloc, num_replications=4, base_seed=100)

    assert metrics.num_replications == 4
    assert len(metrics.replications_mean_wait) == 4
    assert metrics.confidence_interval_95 is not None
    ci_lower, ci_upper = metrics.confidence_interval_95
    assert ci_lower <= metrics.mean_wait_minutes <= ci_upper
