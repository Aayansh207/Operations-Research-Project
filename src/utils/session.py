"""
Session state manager for SmartCare OR.
Ensures scenario settings, baseline allocations, and experiment results
persist smoothly across multi-page navigation.
"""

from typing import Dict, Any, Optional
import streamlit as st
from src.simulation.scenarios import get_predefined_scenarios


def init_session_state():
    """Initializes standard session state keys if not already present."""
    predefined = get_predefined_scenarios()

    if "active_scenario_name" not in st.session_state:
        st.session_state["active_scenario_name"] = "Scenario B: Moderate Demand (Standard)"

    if "scenarios" not in st.session_state:
        st.session_state["scenarios"] = predefined

    if "current_slots" not in st.session_state:
        st.session_state["current_slots"] = predefined["Scenario B: Moderate Demand (Standard)"]["slots"]

    if "total_workforce_limit" not in st.session_state:
        st.session_state["total_workforce_limit"] = predefined["Scenario B: Moderate Demand (Standard)"]["total_workforce_limit"]

    if "budget_limit" not in st.session_state:
        st.session_state["budget_limit"] = predefined["Scenario B: Moderate Demand (Standard)"]["budget_limit"]

    if "latest_experiment" not in st.session_state:
        st.session_state["latest_experiment"] = None

    if "simulation_replications" not in st.session_state:
        st.session_state["simulation_replications"] = 3

    if "random_seed" not in st.session_state:
        st.session_state["random_seed"] = 42

    if "warm_up_hours" not in st.session_state:
        st.session_state["warm_up_hours"] = 0.5


def get_active_slots():
    """Retrieves current slots from session state."""
    init_session_state()
    return st.session_state["current_slots"]


def set_active_scenario(scenario_name: str):
    """Switches the active scenario and updates current slots."""
    init_session_state()
    scenarios = st.session_state["scenarios"]
    if scenario_name in scenarios:
        st.session_state["active_scenario_name"] = scenario_name
        st.session_state["current_slots"] = scenarios[scenario_name]["slots"]
        st.session_state["total_workforce_limit"] = scenarios[scenario_name]["total_workforce_limit"]
        st.session_state["budget_limit"] = scenarios[scenario_name]["budget_limit"]
