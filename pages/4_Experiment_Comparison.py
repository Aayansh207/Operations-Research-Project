"""
SmartCare OR: Experiment Comparison & Trade-off Analysis.
Performs cross-scenario benchmarking, paired statistical evaluations,
and quantitative cost-benefit trade-off calculations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import importlib

from src.simulation.scenarios import get_predefined_scenarios
from src.experiments.runner import ExperimentRunner
from src.utils.export import export_dataframe_to_csv, export_experiment_to_json
from src.utils.session import init_session_state
import src.utils.theme as theme

importlib.reload(theme)
init_session_state()

st.title("Experiment Comparison & Statistical Reporting")
st.markdown("""
This module enables **multi-scenario benchmarking** and rigorous academic comparison of Operations Research interventions.

Key quantitative evaluations:
- **Baseline vs. Optimized Wait Times:** Analytical M/M/c vs. SimPy Discrete-Event Simulation.
- **Statistical Rigor:** Paired difference $t$-tests and 95% Confidence Intervals using Common Random Numbers (CRN).
- **Cost-Benefit Ratio:** Financial expenditure per minute of waiting-time reduction.
""")

scenarios = get_predefined_scenarios()

# Multi-Scenario Benchmark Runner
st.markdown("### 🔬 Multi-Scenario Benchmark Suite")
st.markdown("Run automated evaluation across all predefined demand levels (Low, Moderate, Peak).")

if st.button("🚀 Run Multi-Scenario Benchmark", type="primary"):
    benchmark_results = []
    with st.spinner("Executing analytical models, ILP optimization, and SimPy simulations across all scenarios..."):
        for sc_name, sc_data in scenarios.items():
            runner = ExperimentRunner(
                scenario_name=sc_name,
                slots=sc_data["slots"],
                total_workforce_limit=sc_data["total_workforce_limit"],
                budget_limit=sc_data["budget_limit"],
                num_replications=st.session_state["simulation_replications"],
                random_seed=st.session_state["random_seed"],
                warm_up_hours=st.session_state["warm_up_hours"]
            )
            res = runner.run_full_experiment()
            res_df = res["df_comparison"].copy()
            res_df["Scenario"] = sc_name
            benchmark_results.append((res, res_df))

        st.session_state["benchmark_results"] = benchmark_results
        st.success("Multi-scenario benchmark execution complete!")

bench_data = st.session_state.get("benchmark_results", None)

if bench_data:
    st.markdown("### 📊 Benchmark Summary Results")

    all_dfs = pd.concat([b[1] for b in bench_data], ignore_index=True)
    # Reorder columns
    cols_order = ["Scenario", "Policy / Method", "Data Source", "Total Doctors", "Staffing Cost ($)", "Avg Wait Time (min)", "Doctor Utilization", "Stability Status"]
    all_dfs = all_dfs[cols_order]

    st.dataframe(all_dfs, use_container_width=True, hide_index=True)

    # Cross-Scenario Key Metrics Table
    st.markdown("#### 🎯 Quantitative Wait-Time Reductions Across Scenarios")
    summary_cards = []
    sc_names = []
    base_waits = []
    opt_waits = []
    ci_lows = []
    ci_highs = []
    reduc_vals = []

    for exp_res, _ in bench_data:
        base_w = exp_res["baseline_sim_metrics"].mean_wait_minutes
        opt_w = exp_res["opt_sim_metrics"].mean_wait_minutes if exp_res["opt_sim_metrics"] else float("nan")
        reduc = exp_res["sim_wait_reduction_pct"]
        cost_diff = (exp_res["opt_ilp_result"].total_cost if exp_res["opt_ilp_result"].is_feasible else 0) - exp_res["baseline_analytical"].total_cost

        sc_name_short = exp_res["scenario_name"].split(":")[0]
        sc_names.append(sc_name_short)
        base_waits.append(base_w)
        opt_waits.append(opt_w)
        reduc_vals.append(reduc)

        ci = exp_res["paired_stats"]["ci_95"]
        if ci:
            ci_lows.append(ci[0])
            ci_highs.append(ci[1])
        else:
            ci_lows.append(opt_w)
            ci_highs.append(opt_w)

        summary_cards.append({
            "Scenario": sc_name_short,
            "Baseline Wait (min)": f"{base_w:.2f}",
            "Optimized Wait (min)": f"{opt_w:.2f}",
            "Wait Reduction (%)": f"{reduc:.1f}%",
            "Cost Change ($)": f"{cost_diff:+,.0f}",
            "Cost/Min Saved": f"${(cost_diff / (base_w - opt_w)):.2f}" if (base_w - opt_w) > 0 and cost_diff > 0 else "N/A (Cost Saving or Negative)",
            "Paired 95% CI (min)": f"[{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "N/A"
        })

    st.dataframe(pd.DataFrame(summary_cards), use_container_width=True, hide_index=True)

    # Visualizations
    st.markdown("#### 📈 Operations Research Comparative Visualizations")
    tab_chart1, tab_chart2 = st.tabs([
        "1. Waiting-Time Reduction by Scenario (Grouped Bars + 95% CI)",
        "2. Cost-Benefit Intervention Trajectory (Cost vs. Wait Time)"
    ])

    with tab_chart1:
        st.markdown("""
        **Validation Insight:** Compares simulated patient waiting times under the Baseline Policy versus the ILP-Optimized Policy across demand tiers.
        Error bars represent empirical 95% confidence intervals from repeated SimPy replications.
        """)
        fig_wait_comp = theme.plot_cross_scenario_wait_comparison(
            scenario_names=sc_names,
            baseline_waits=base_waits,
            opt_waits=opt_waits,
            ci_lowers=ci_lows,
            ci_uppers=ci_highs,
            reductions=reduc_vals
        )
        st.plotly_chart(fig_wait_comp, use_container_width=True)

    with tab_chart2:
        st.markdown("""
        **Intervention Trajectory Insight:** Illustrates the vector of change from the **Baseline Policy (red squares)** to the **Optimized Policy (green diamonds)**.
        Demonstrates how modest targeted budget allocations resolve peak bottlenecks and drastically compress waiting times.
        """)
        fig_tradeoff = theme.plot_cross_scenario_cost_tradeoff(bench_data)
        st.plotly_chart(fig_tradeoff, use_container_width=True)

    # Download Buttons
    st.markdown("#### 💾 Export Benchmark Findings")
    d1, d2 = st.columns(2)
    with d1:
        csv_bench = export_dataframe_to_csv(all_dfs)
        st.download_button(
            "📥 Download Complete Benchmark Dataset (CSV)",
            data=csv_bench,
            file_name="smartcare_benchmark_comparison.csv",
            mime="text/csv",
            use_container_width=True
        )
    with d2:
        json_bench = export_experiment_to_json(bench_data[1][0])  # Scenario B
        st.download_button(
            "📥 Download Scenario B Experiment Log (JSON)",
            data=json_bench,
            file_name="smartcare_scenario_b_experiment.json",
            mime="application/json",
            use_container_width=True
        )
else:
    st.info("Click **'Run Multi-Scenario Benchmark'** above to run an automated comparative study across all demand tiers.")
