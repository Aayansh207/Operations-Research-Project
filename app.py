"""
SmartCare OR: Intelligent Hospital Patient Wait-Time Optimization System.
Overview Dashboard & Academic Evaluation Hub.
"""

import streamlit as st
import pandas as pd
import numpy as np

from src.utils.session import init_session_state, set_active_scenario
from src.simulation.scenarios import get_predefined_scenarios
from src.experiments.runner import ExperimentRunner, is_finite_number
from src.utils.export import export_experiment_to_json, export_dataframe_to_csv
import importlib
import src.utils.theme as theme
importlib.reload(theme)

# Page configuration
st.set_page_config(
    page_title="SmartCare OR — Hospital Wait-Time Optimization",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
init_session_state()

# Custom CSS for polished, professional healthcare analytics UI
st.markdown("""
<style>
    /* Metric Cards */
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 12px;
    }
    .metric-label {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #0F172A;
        line-height: 1.2;
    }
    .metric-delta-pos {
        font-size: 0.85rem;
        font-weight: 600;
        color: #10B981;
        margin-top: 4px;
    }
    .metric-delta-neg {
        font-size: 0.85rem;
        font-weight: 600;
        color: #EF4444;
        margin-top: 4px;
    }
    .badge-tag {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-blue { background-color: #EFF6FF; color: #1D4ED8; border: 1px solid #BFDBFE; }
    .badge-green { background-color: #ECFDF5; color: #047857; border: 1px solid #A7F3D0; }
    .badge-purple { background-color: #F5F3FF; color: #6D28D9; border: 1px solid #DDD6FE; }
    .badge-amber { background-color: #FFFBEB; color: #B45309; border: 1px solid #FDE68A; }
    .empty-state-box {
        border: 2px dashed #CBD5E1;
        border-radius: 12px;
        padding: 40px;
        text-align: center;
        background-color: #F8FAFC;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1519494026892-80bbd2d6fd0d?w=300&auto=format&fit=crop&q=60", use_container_width=True)
    st.markdown("### 🏥 System Configuration")

    predefined = get_predefined_scenarios()
    scenario_choice = st.selectbox(
        "Select Hospital Demand Scenario:",
        options=list(predefined.keys()),
        index=list(predefined.keys()).index(st.session_state["active_scenario_name"])
    )

    if scenario_choice != st.session_state["active_scenario_name"]:
        set_active_scenario(scenario_choice)
        st.session_state["latest_experiment"] = None
        st.rerun()

    st.markdown("---")
    st.markdown("#### ⚙️ Simulation Controls")
    reps = st.slider("Simulation Replications:", min_value=1, max_value=10, value=st.session_state["simulation_replications"], step=1)
    st.session_state["simulation_replications"] = reps

    seed = st.number_input("Controlled Random Seed:", min_value=1, max_value=99999, value=st.session_state["random_seed"])
    st.session_state["random_seed"] = seed

    warmup = st.slider("Warm-up Horizon (hours):", min_value=0.0, max_value=2.0, value=float(st.session_state["warm_up_hours"]), step=0.25)
    st.session_state["warm_up_hours"] = warmup

    st.markdown("---")
    st.markdown("""
    **Academic Notice:**
    This application is an educational decision-support experimentation platform for Operations Research.
    All data is synthetic. No medical decisions should be made based on this software.
    """)

# Main Page Header
col_header_left, col_header_right = st.columns([3, 1])

with col_header_left:
    st.title("SmartCare OR")
    st.markdown("##### *Intelligent Hospital Patient Wait-Time Optimization System*")
    st.markdown("""
    <span class="badge-tag badge-blue">Technique 1: M/M/c Queueing Theory</span>
    <span class="badge-tag badge-purple">Technique 2: PuLP Integer Programming</span>
    <span class="badge-tag badge-green">Evaluation: SimPy Discrete-Event Simulation</span>
    """, unsafe_allow_html=True)
    st.markdown("""
    SmartCare OR studies how Operations Research techniques resolve emergency and outpatient congestion
    by optimizing doctor allocations across time-varying demand shifts while balancing staffing expenditure.
    """)

with col_header_right:
    st.markdown("<br>", unsafe_allow_html=True)
    demo_btn = st.button("⚡ Live Demo Mode (3-Min Presets)", type="primary", use_container_width=True)
    if demo_btn:
        with st.spinner("Running full Operations Research evaluation on Scenario B..."):
            set_active_scenario("Scenario B: Moderate Demand (Standard)")
            sc = get_predefined_scenarios()["Scenario B: Moderate Demand (Standard)"]
            runner = ExperimentRunner(
                scenario_name="Scenario B: Moderate Demand (Standard)",
                slots=sc["slots"],
                total_workforce_limit=sc["total_workforce_limit"],
                budget_limit=sc["budget_limit"],
                num_replications=st.session_state["simulation_replications"],
                random_seed=st.session_state["random_seed"],
                warm_up_hours=st.session_state["warm_up_hours"]
            )
            st.session_state["latest_experiment"] = runner.run_full_experiment()
            st.success("Live demonstration run complete!")
            st.rerun()

st.markdown("---")

# Active Scenario Banner
active_sc_name = st.session_state["active_scenario_name"]
active_sc_desc = predefined[active_sc_name]["description"]
current_slots = st.session_state["current_slots"]

st.markdown(f"### 📋 Active Scenario: **{active_sc_name}**")
st.info(f"**Description:** {active_sc_desc}")

# Display Shift Schedule Preview
shift_cols = st.columns(len(current_slots))
for idx, slot in enumerate(current_slots):
    with shift_cols[idx]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{slot.name} Shift ({slot.start_hour:02.0f}:00 - {slot.end_hour:02.0f}:00)</div>
            <div style="font-size: 1.1rem; font-weight: 600; color: #1E293B; margin: 4px 0;">
                Demand: <b>{slot.arrival_rate:.1f}</b> pts/hr
            </div>
            <div style="font-size: 0.85rem; color: #64748B;">
                Doctor Service Rate: {slot.service_rate:.1f} pts/hr<br>
                Baseline Staff: <b>{slot.baseline_doctors}</b> doctors<br>
                Baseline Capacity: <b>{slot.baseline_doctors * slot.service_rate:.1f}</b> pts/hr<br>
                Cost: ${slot.cost_per_doctor_hour:.0f}/hr
            </div>
        </div>
        """, unsafe_allow_html=True)

# Run Full Experiment Action Button
col_act1, col_act2 = st.columns([1, 3])
with col_act1:
    run_exp_btn = st.button("🚀 Run Full OR Experiment", type="primary", use_container_width=True)
    if run_exp_btn:
        with st.spinner("Solving Integer Program & Simulating Patient Flows..."):
            runner = ExperimentRunner(
                scenario_name=active_sc_name,
                slots=current_slots,
                total_workforce_limit=st.session_state["total_workforce_limit"],
                budget_limit=st.session_state["budget_limit"],
                num_replications=st.session_state["simulation_replications"],
                random_seed=st.session_state["random_seed"],
                warm_up_hours=st.session_state["warm_up_hours"]
            )
            st.session_state["latest_experiment"] = runner.run_full_experiment()
            st.rerun()

st.markdown("<br>", unsafe_allow_html=True)

# Results Section
exp = st.session_state["latest_experiment"]

if exp is None:
    st.markdown("""
    <div class="empty-state-box">
        <h3 style="color: #475569;">No Experiment Executed Yet</h3>
        <p style="color: #64748B; max-width: 600px; margin: 0 auto 20px auto;">
            Click <b>'Run Full OR Experiment'</b> above or select <b>'⚡ Live Demo Mode'</b> to run the integer optimization solver, evaluate queueing equations, and simulate patient flows.
        </p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("### 📊 Baseline vs. Optimized Operations Research Findings")
    st.caption(f"Experiment executed at: {exp['timestamp']} | Scenario: {exp['scenario_name']}")

    # KPI Top Row
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    # 1. Average Waiting Time
    base_wait_sim = exp["baseline_sim_metrics"].mean_wait_minutes
    opt_wait_sim = exp["opt_sim_metrics"].mean_wait_minutes if exp["opt_sim_metrics"] else float("nan")
    reduc_pct = exp["sim_wait_reduction_pct"]
    delta_class = "metric-delta-pos" if reduc_pct > 0 else "metric-delta-neg"

    with kpi1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Simulated Avg Wait Time</div>
            <div class="metric-value">{opt_wait_sim:.2f} <span style="font-size: 1rem; font-weight: 500;">min</span></div>
            <div class="{delta_class}">{exp['sim_wait_reduc_label']} (Baseline: {base_wait_sim:.2f}m)</div>
            <div style="font-size: 0.75rem; color: #94A3B8; margin-top: 6px;">Source: SimPy (Common Random Numbers)</div>
        </div>
        """, unsafe_allow_html=True)

    # 2. Staffing Cost
    base_cost = exp["baseline_analytical"].total_cost
    opt_cost = exp["opt_ilp_result"].total_cost if exp["opt_ilp_result"].is_feasible else 0.0
    cost_diff = opt_cost - base_cost
    cost_delta_class = "metric-delta-pos" if cost_diff <= 0 else "metric-delta-neg"

    with kpi2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Staffing Cost</div>
            <div class="metric-value">${opt_cost:,.0f}</div>
            <div class="{cost_delta_class}">{exp['ana_cost_savings_label']}</div>
            <div style="font-size: 0.75rem; color: #94A3B8; margin-top: 6px;">Source: Integer Linear Program</div>
        </div>
        """, unsafe_allow_html=True)

    # 3. Doctor Utilization
    opt_util = exp["opt_sim_metrics"].server_utilization if exp["opt_sim_metrics"] else 0.0
    base_util = exp["baseline_sim_metrics"].server_utilization

    with kpi3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Simulated Doctor Utilization</div>
            <div class="metric-value">{opt_util:.1%}</div>
            <div style="font-size: 0.85rem; color: #64748B; margin-top: 4px;">Baseline: {base_util:.1%}</div>
            <div style="font-size: 0.75rem; color: #94A3B8; margin-top: 6px;">Source: Active Service / Available Time</div>
        </div>
        """, unsafe_allow_html=True)

    # 4. Queue Length
    opt_lq = exp["opt_sim_metrics"].time_weighted_queue_length if exp["opt_sim_metrics"] else 0.0
    base_lq = exp["baseline_sim_metrics"].time_weighted_queue_length

    with kpi4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Time-Weighted Queue Length</div>
            <div class="metric-value">{opt_lq:.2f} <span style="font-size: 1rem; font-weight: 500;">pts</span></div>
            <div style="font-size: 0.85rem; color: #64748B; margin-top: 4px;">Baseline: {base_lq:.2f} patients</div>
            <div style="font-size: 0.75rem; color: #94A3B8; margin-top: 6px;">Source: SimPy Continuous Integration</div>
        </div>
        """, unsafe_allow_html=True)

    # Comparative Summary Table
    st.markdown("#### 📋 Comprehensive Cross-Methodology Comparison")
    st.dataframe(exp["df_comparison"], use_container_width=True, hide_index=True)

    # Key Charts Row
    st.markdown("#### 📈 Operational Visualizations")
    col_chart_left, col_chart_right = st.columns(2)

    with col_chart_left:
        fig_alloc = theme.plot_staffing_allocation_comparison(
            slots=current_slots,
            baseline_alloc=exp["baseline_allocation"],
            opt_alloc=exp["opt_ilp_result"].allocation if exp["opt_ilp_result"].is_feasible else exp["baseline_allocation"]
        )
        st.plotly_chart(fig_alloc, use_container_width=True)

    with col_chart_right:
        if exp["opt_patient_dfs"] and exp["baseline_patient_dfs"]:
            fig_dist = theme.plot_simulation_wait_distribution(
                df_baseline=exp["baseline_patient_dfs"][0],
                df_optimized=exp["opt_patient_dfs"][0]
            )
            st.plotly_chart(fig_dist, use_container_width=True)

    # Download Actions
    st.markdown("#### 💾 Export Scientific Findings")
    dcol1, dcol2 = st.columns(2)
    with dcol1:
        json_exp = export_experiment_to_json(exp)
        st.download_button(
            label="📥 Download Experiment Summary (JSON)",
            data=json_exp,
            file_name=f"smartcare_experiment_{active_sc_name.split(':')[0].replace(' ', '_').lower()}.json",
            mime="application/json",
            use_container_width=True
        )
    with dcol2:
        csv_exp = export_dataframe_to_csv(exp["df_comparison"])
        st.download_button(
            label="📥 Download Metrics Comparison Table (CSV)",
            data=csv_exp,
            file_name=f"smartcare_comparison_{active_sc_name.split(':')[0].replace(' ', '_').lower()}.csv",
            mime="text/csv",
            use_container_width=True
        )

# Navigation Footer Cards
st.markdown("---")
st.markdown("### 🧭 In-Depth Exploratory Modules")
m_col1, m_col2, m_col3, m_col4 = st.columns(4)

with m_col1:
    st.markdown("""
    **1. Queueing Theory Analyzer**
    - Analytical M/M/c equations
    - Sensitivity curves & Erlang C
    - Capacity hockey-stick effects
    """)

with m_col2:
    st.markdown("""
    **2. Staff Scheduling ILP**
    - PuLP Integer Linear Program
    - Safety margins & budget limits
    - Pareto cost vs wait-time curves
    """)

with m_col3:
    st.markdown("""
    **3. Discrete-Event Simulation**
    - Non-preemptive shift handovers
    - Queue timelines & distributions
    - Synthetic patient arrival traces
    """)

with m_col4:
    st.markdown("""
    **4. Methodology & Literature**
    - Formal mathematical derivations
    - Academic citations & assumptions
    - Research limitation notes
    """)
