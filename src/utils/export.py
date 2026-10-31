"""
Export and formatting utilities for SmartCare OR.
Handles CSV and JSON exports of experiment data, scenario configs, and metric summaries.
"""

from typing import Dict, Any, List
import io
import json
import pandas as pd


def export_experiment_to_json(experiment_results: Dict[str, Any]) -> str:
    """Serializes experiment output to structured JSON."""
    cleaned = {
        "timestamp": experiment_results.get("timestamp"),
        "scenario_name": experiment_results.get("scenario_name"),
        "baseline_allocation": experiment_results.get("baseline_allocation"),
        "optimal_allocation": (
            experiment_results.get("opt_ilp_result").allocation
            if experiment_results.get("opt_ilp_result") else {}
        ),
        "analytical_comparison": {
            "baseline_wait_min": (
                experiment_results.get("baseline_analytical").average_wait_minutes
                if experiment_results.get("baseline_analytical") else None
            ),
            "optimized_wait_min": (
                experiment_results.get("opt_ilp_result").average_wait_minutes
                if experiment_results.get("opt_ilp_result") else None
            ),
            "waiting_time_reduction_pct": experiment_results.get("ana_wait_reduction_pct"),
            "cost_savings_pct": experiment_results.get("ana_cost_savings_pct")
        },
        "simulation_comparison": {
            "baseline_wait_min": (
                experiment_results.get("baseline_sim_metrics").mean_wait_minutes
                if experiment_results.get("baseline_sim_metrics") else None
            ),
            "optimized_wait_min": (
                experiment_results.get("opt_sim_metrics").mean_wait_minutes
                if experiment_results.get("opt_sim_metrics") else None
            ),
            "sim_waiting_reduction_pct": experiment_results.get("sim_wait_reduction_pct"),
            "paired_stats": experiment_results.get("paired_stats")
        }
    }
    return json.dumps(cleaned, indent=2)


def export_dataframe_to_csv(df: pd.DataFrame) -> bytes:
    """Encodes a DataFrame into a downloadable CSV byte stream."""
    return df.to_csv(index=False).encode("utf-8")
