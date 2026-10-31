"""
Experiment Runner: End-to-End Evaluation Pipeline.

Executes a complete comparative experiment evaluating:
1. Baseline Configuration (Analytical M/M/c & Discrete-Event Simulation)
2. Technique 2: Integer Linear Programming Staff Allocation (Analytical M/M/c & Discrete-Event Simulation)
3. Technique 2 (Variant): Wait-Time Constrained Optimization
4. Quantitative Comparison & Performance Metric Calculations
"""

import math
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd
import numpy as np

from src.data.schemas import TimeSlot, OptimizationResult, SimulationMetrics
from src.models.queueing import calculate_mmc


def is_finite_number(x: Any) -> bool:
    """Safely checks if x is a finite number, compatible with Python 3.14 and NumPy 2.x."""
    try:
        return x is not None and not math.isinf(float(x)) and not math.isnan(float(x))
    except (ValueError, TypeError, OverflowError):
        return False

from src.models.scheduling import (
    optimize_staff_ilp,
    optimize_staff_wait_time_constrained,
    evaluate_schedule
)
from src.simulation.patient_flow import HospitalSimulation, run_multi_replication_simulation
from src.experiments.metrics import (
    calculate_wait_reduction_pct,
    calculate_cost_savings_pct,
    compute_paired_comparison_stats
)


class ExperimentRunner:
    """Coordinates reproducible end-to-end Operations Research experiments."""

    def __init__(
        self,
        scenario_name: str,
        slots: List[TimeSlot],
        total_workforce_limit: Optional[int] = None,
        budget_limit: Optional[float] = None,
        num_replications: int = 5,
        random_seed: int = 42,
        warm_up_hours: float = 0.0,
        max_wait_target_min: float = 15.0
    ):
        self.scenario_name = scenario_name
        self.slots = slots
        self.total_workforce_limit = total_workforce_limit
        self.budget_limit = budget_limit
        self.num_replications = num_replications
        self.random_seed = random_seed
        self.warm_up_hours = warm_up_hours
        self.max_wait_target_min = max_wait_target_min

    def run_full_experiment(self) -> Dict[str, Any]:
        """
        Executes analytical models, ILP optimization, and discrete-event simulation.
        Returns a comprehensive experiment result package.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # -------------------------------------------------------------
        # 1. Baseline Evaluation
        # -------------------------------------------------------------
        baseline_allocation = {s.name: s.baseline_doctors for s in self.slots}
        baseline_analytical = evaluate_schedule(
            self.slots,
            baseline_allocation,
            status="Baseline (Analytical)",
            solver_message="Baseline manual staffing policy."
        )

        # Baseline Simulation (with Common Random Numbers)
        baseline_sim_metrics, baseline_patient_dfs = run_multi_replication_simulation(
            slots=self.slots,
            allocation=baseline_allocation,
            num_replications=self.num_replications,
            base_seed=self.random_seed,
            warm_up_hours=self.warm_up_hours
        )

        # -------------------------------------------------------------
        # 2. Technique 2: ILP Cost-Minimizing Optimization
        # -------------------------------------------------------------
        opt_ilp_result = optimize_staff_ilp(
            slots=self.slots,
            total_workforce_limit=self.total_workforce_limit,
            budget_limit=self.budget_limit
        )

        # If ILP is feasible, run simulation using the EXACT SAME seeds (CRN)
        if opt_ilp_result.is_feasible:
            opt_sim_metrics, opt_patient_dfs = run_multi_replication_simulation(
                slots=self.slots,
                allocation=opt_ilp_result.allocation,
                num_replications=self.num_replications,
                base_seed=self.random_seed,  # Common Random Numbers
                warm_up_hours=self.warm_up_hours
            )
        else:
            opt_sim_metrics = None
            opt_patient_dfs = []

        # -------------------------------------------------------------
        # 3. Wait-Time Constrained Optimization (<= target minutes)
        # -------------------------------------------------------------
        opt_wt_result = optimize_staff_wait_time_constrained(
            slots=self.slots,
            max_wait_minutes=self.max_wait_target_min,
            total_workforce_limit=self.total_workforce_limit,
            budget_limit=self.budget_limit
        )

        # -------------------------------------------------------------
        # 4. Comparative Metrics
        # -------------------------------------------------------------
        # Analytical comparison
        base_wait_ana = baseline_analytical.average_wait_minutes
        opt_wait_ana = opt_ilp_result.average_wait_minutes if opt_ilp_result.is_feasible else float("inf")
        ana_wait_reduc, ana_wait_reduc_label = calculate_wait_reduction_pct(base_wait_ana, opt_wait_ana)
        ana_cost_sav, ana_cost_sav_label = calculate_cost_savings_pct(
            baseline_analytical.total_cost,
            opt_ilp_result.total_cost if opt_ilp_result.is_feasible else 0.0
        )

        # Simulation comparison (using paired statistics)
        if opt_sim_metrics:
            base_wait_sim = baseline_sim_metrics.mean_wait_minutes
            opt_wait_sim = opt_sim_metrics.mean_wait_minutes
            sim_wait_reduc, sim_wait_reduc_label = calculate_wait_reduction_pct(base_wait_sim, opt_wait_sim)
            paired_stats = compute_paired_comparison_stats(
                baseline_sim_metrics.replications_mean_wait,
                opt_sim_metrics.replications_mean_wait
            )
        else:
            base_wait_sim = baseline_sim_metrics.mean_wait_minutes
            opt_wait_sim = float("nan")
            sim_wait_reduc, sim_wait_reduc_label = 0.0, "Infeasible optimization"
            paired_stats = {"mean_diff": 0.0, "ci_95": None, "stat_significant": False}

        # -------------------------------------------------------------
        # 5. Build Unified Comparison Table
        # -------------------------------------------------------------
        comparison_rows = [
            {
                "Policy / Method": "Baseline Staffing",
                "Data Source": "Analytical M/M/c",
                "Total Doctors": sum(baseline_allocation.values()),
                "Staffing Cost ($)": baseline_analytical.total_cost,
                "Avg Wait Time (min)": f"{round(base_wait_ana, 2):.2f}" if is_finite_number(base_wait_ana) else "Unstable",
                "Doctor Utilization": f"{np.mean([q.utilization for q in baseline_analytical.slot_queueing_metrics.values()]):.1%}",
                "Stability Status": "Stable" if baseline_analytical.is_feasible else "Unstable"
            },
            {
                "Policy / Method": "Optimized Staffing (ILP)",
                "Data Source": "Integer Linear Program + M/M/c",
                "Total Doctors": sum(opt_ilp_result.allocation.values()) if opt_ilp_result.is_feasible else 0,
                "Staffing Cost ($)": opt_ilp_result.total_cost,
                "Avg Wait Time (min)": f"{round(opt_wait_ana, 2):.2f}" if is_finite_number(opt_wait_ana) else "Unstable",
                "Doctor Utilization": f"{np.mean([q.utilization for q in opt_ilp_result.slot_queueing_metrics.values()]):.1%}" if opt_ilp_result.is_feasible else "N/A",
                "Stability Status": opt_ilp_result.status
            },
            {
                "Policy / Method": "Wait-Time Constrained",
                "Data Source": "Wait-Bounded ILP + M/M/c",
                "Total Doctors": sum(opt_wt_result.allocation.values()) if opt_wt_result.is_feasible else 0,
                "Staffing Cost ($)": opt_wt_result.total_cost,
                "Avg Wait Time (min)": f"{round(opt_wt_result.average_wait_minutes, 2):.2f}" if is_finite_number(opt_wt_result.average_wait_minutes) else "Unstable",
                "Doctor Utilization": f"{np.mean([q.utilization for q in opt_wt_result.slot_queueing_metrics.values()]):.1%}" if opt_wt_result.is_feasible else "N/A",
                "Stability Status": opt_wt_result.status
            },
            {
                "Policy / Method": "Baseline Simulation",
                "Data Source": f"SimPy ({self.num_replications} reps, seed {self.random_seed})",
                "Total Doctors": sum(baseline_allocation.values()),
                "Staffing Cost ($)": baseline_analytical.total_cost,
                "Avg Wait Time (min)": f"{round(base_wait_sim, 2):.2f}",
                "Doctor Utilization": f"{baseline_sim_metrics.server_utilization:.1%}",
                "Stability Status": "Simulated"
            },
        ]

        if opt_sim_metrics:
            comparison_rows.append({
                "Policy / Method": "Optimized Simulation",
                "Data Source": f"SimPy ({self.num_replications} reps, seed {self.random_seed})",
                "Total Doctors": sum(opt_ilp_result.allocation.values()),
                "Staffing Cost ($)": opt_ilp_result.total_cost,
                "Avg Wait Time (min)": f"{round(opt_wait_sim, 2):.2f}",
                "Doctor Utilization": f"{opt_sim_metrics.server_utilization:.1%}",
                "Stability Status": "Simulated"
            })

        df_comparison = pd.DataFrame(comparison_rows)

        return {
            "timestamp": timestamp,
            "scenario_name": self.scenario_name,
            "slots": self.slots,
            "baseline_allocation": baseline_allocation,
            "baseline_analytical": baseline_analytical,
            "baseline_sim_metrics": baseline_sim_metrics,
            "opt_ilp_result": opt_ilp_result,
            "opt_wt_result": opt_wt_result,
            "opt_sim_metrics": opt_sim_metrics,
            "ana_wait_reduction_pct": ana_wait_reduc,
            "ana_wait_reduc_label": ana_wait_reduc_label,
            "ana_cost_savings_pct": ana_cost_sav,
            "ana_cost_savings_label": ana_cost_sav_label,
            "sim_wait_reduction_pct": sim_wait_reduc,
            "sim_wait_reduc_label": sim_wait_reduc_label,
            "paired_stats": paired_stats,
            "df_comparison": df_comparison,
            "baseline_patient_dfs": baseline_patient_dfs,
            "opt_patient_dfs": opt_patient_dfs
        }
