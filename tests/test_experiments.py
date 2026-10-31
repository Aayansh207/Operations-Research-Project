"""
Unit tests for End-to-End Experiment Pipeline and Data Export.
Verifies complete experiment workflow, comparison dataframe structures, and JSON/CSV serialization.
"""

import pytest
import json
from src.simulation.scenarios import get_predefined_scenarios
from src.experiments.runner import ExperimentRunner
from src.utils.export import export_experiment_to_json, export_dataframe_to_csv


def test_full_experiment_pipeline_execution():
    """Runs full pipeline on Scenario A: Low Demand."""
    scenarios = get_predefined_scenarios()
    sc = scenarios["Scenario A: Low Demand"]

    runner = ExperimentRunner(
        scenario_name="Scenario A: Low Demand",
        slots=sc["slots"],
        total_workforce_limit=sc["total_workforce_limit"],
        budget_limit=sc["budget_limit"],
        num_replications=2,
        random_seed=42
    )

    res = runner.run_full_experiment()

    assert "df_comparison" in res
    df = res["df_comparison"]
    assert not df.empty
    assert "Policy / Method" in df.columns
    assert "Staffing Cost ($)" in df.columns
    assert "Avg Wait Time (min)" in df.columns
    assert len(df) >= 4  # Baseline Ana, Opt ILP, Wait-Constrained, Baseline Sim, Opt Sim

    # Check exports
    json_str = export_experiment_to_json(res)
    parsed_json = json.loads(json_str)
    assert parsed_json["scenario_name"] == "Scenario A: Low Demand"
    assert "analytical_comparison" in parsed_json

    csv_bytes = export_dataframe_to_csv(df)
    assert len(csv_bytes) > 50
