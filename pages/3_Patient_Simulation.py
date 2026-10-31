"""
SmartCare OR: Discrete-Event Patient Flow Simulation (SimPy).
Simulates dynamic patient arrivals, continuous FIFO queues, non-preemptive shift handovers,
and computes empirical distributions and time-weighted metrics.
"""

import streamlit as st
import pandas as pd
import numpy as np

from src.data.schemas import TimeSlot
from src.simulation.patient_flow import HospitalSimulation, run_multi_replication_simulation
from src.data.generators import generate_synthetic_patient_stream, validate_imported_patient_csv
from src.models.scheduling import optimize_staff_ilp
from src.utils.export import export_dataframe_to_csv
import src.utils.theme as theme
from src.utils.session import init_session_state, get_active_slots

init_session_state()

st.title("Discrete-Event Patient Flow Simulation (SimPy)")
st.markdown("""
This module uses **discrete-event simulation** to evaluate patient-flow behavior under dynamic, time-varying staffing.

Unlike steady-state queueing theory, the simulation models:
- **Shift Handover Dynamics:** Doctors finish active consultations before retiring (non-preemptive staffing adjustments).
- **Queue Persistence:** Queues are continuous across shift boundaries (not artificially reset).
- **Finite-Horizon Congestion:** Captures peak arrival transients, queue build-ups, and empty-to-full development.
""")

current_slots = get_active_slots()

# Simulation Setup
st.markdown("### ⚙️ Simulation Execution Parameters")
c1, c2, c3, c4 = st.columns(4)

with c1:
    policy_choice = st.selectbox(
        "Staffing Policy to Simulate:",
        options=["Both (Paired Baseline vs. Optimized)", "Baseline Staffing Only", "ILP Optimized Staffing Only"],
        index=0
    )

with c2:
    num_reps = st.slider("Number of Replications:", min_value=1, max_value=10, value=int(st.session_state["simulation_replications"]))

with c3:
    sim_seed = st.number_input("Random Seed (CRN):", min_value=1, max_value=99999, value=int(st.session_state["random_seed"]))

with c4:
    warmup_val = st.slider("Warm-up Horizon (hours):", min_value=0.0, max_value=2.0, value=float(st.session_state["warm_up_hours"]), step=0.25)

# Allocations setup
baseline_alloc = {s.name: s.baseline_doctors for s in current_slots}
opt_ilp = optimize_staff_ilp(current_slots, st.session_state["total_workforce_limit"], st.session_state["budget_limit"])
opt_alloc = opt_ilp.allocation if opt_ilp.is_feasible else baseline_alloc

st.markdown("---")

# Run Simulation Button
if st.button("▶️ Run SimPy Discrete-Event Simulation", type="primary", use_container_width=True):
    with st.spinner("Executing discrete-event simulation across scheduled shifts..."):
        # Run Baseline
        base_sim = HospitalSimulation(current_slots, baseline_alloc, warm_up_hours=warmup_val, random_seed=sim_seed)
        m_base, df_p_base, df_q_base = base_sim.run()

        # Run Optimized
        opt_sim = HospitalSimulation(current_slots, opt_alloc, warm_up_hours=warmup_val, random_seed=sim_seed)
        m_opt, df_p_opt, df_q_opt = opt_sim.run()

        # Multi-replication if requested
        if num_reps > 1:
            m_base_multi, _ = run_multi_replication_simulation(current_slots, baseline_alloc, num_reps, sim_seed, warmup_val)
            m_opt_multi, _ = run_multi_replication_simulation(current_slots, opt_alloc, num_reps, sim_seed, warmup_val)
        else:
            m_base_multi = m_base
            m_opt_multi = m_opt

        st.session_state["sim_results"] = {
            "m_base": m_base_multi,
            "df_p_base": df_p_base,
            "df_q_base": df_q_base,
            "m_opt": m_opt_multi,
            "df_p_opt": df_p_opt,
            "df_q_opt": df_q_opt,
            "opt_alloc": opt_alloc,
            "baseline_alloc": baseline_alloc
        }

sim_res = st.session_state.get("sim_results", None)

if sim_res:
    m_base = sim_res["m_base"]
    m_opt = sim_res["m_opt"]
    df_p_base = sim_res["df_p_base"]
    df_p_opt = sim_res["df_p_opt"]
    df_q_base = sim_res["df_q_base"]
    df_q_opt = sim_res["df_q_opt"]

    st.markdown("### 📊 Simulation Performance Metrics")

    # Metric Cards Row
    k1, k2, k3, k4 = st.columns(4)

    reduc_val = ((m_base.mean_wait_minutes - m_opt.mean_wait_minutes) / m_base.mean_wait_minutes) * 100 if m_base.mean_wait_minutes > 0 else 0.0

    with k1:
        st.metric(
            "Mean Waiting Time",
            f"{m_opt.mean_wait_minutes:.2f} min",
            delta=f"{reduc_val:.1f}% reduction vs Baseline ({m_base.mean_wait_minutes:.2f}m)"
        )
        if m_opt.confidence_interval_95 and num_reps > 1:
            st.caption(f"95% CI: [{m_opt.confidence_interval_95[0]:.2f}, {m_opt.confidence_interval_95[1]:.2f}] min")

    with k2:
        st.metric(
            "90th Percentile Wait",
            f"{m_opt.p90_wait_minutes:.2f} min",
            delta=f"Baseline: {m_base.p90_wait_minutes:.2f} min",
            delta_color="inverse"
        )
        st.caption("Captures extreme congestion delays")

    with k3:
        st.metric(
            "Time-Weighted Queue (Lq)",
            f"{m_opt.time_weighted_queue_length:.2f} pts",
            delta=f"Baseline: {m_base.time_weighted_queue_length:.2f} pts",
            delta_color="inverse"
        )
        st.caption("Exact integral of Q(t) dt / T")

    with k4:
        st.metric(
            "Doctor Utilization",
            f"{m_opt.server_utilization:.1%}",
            delta=f"Baseline: {m_base.server_utilization:.1%}"
        )
        st.caption("Active consultation / Available time")

    # Patient Conservation Inventory
    st.markdown("#### 🏥 Patient Accounting & Inventory")
    inv_df = pd.DataFrame([
        {
            "Staffing Policy": "Baseline Policy",
            "Doctors Assigned": sum(sim_res["baseline_alloc"].values()),
            "Total Arrived": m_base.patients_arrived,
            "Completed Service": m_base.patients_served,
            "Waiting in Queue at End": m_base.patients_waiting_at_end,
            "In Consultation at End": m_base.patients_in_service_at_end,
            "Mean Wait (min)": f"{m_base.mean_wait_minutes:.2f}",
            "Median Wait (min)": f"{m_base.median_wait_minutes:.2f}"
        },
        {
            "Staffing Policy": "Optimized Policy (ILP)",
            "Doctors Assigned": sum(sim_res["opt_alloc"].values()),
            "Total Arrived": m_opt.patients_arrived,
            "Completed Service": m_opt.patients_served,
            "Waiting in Queue at End": m_opt.patients_waiting_at_end,
            "In Consultation at End": m_opt.patients_in_service_at_end,
            "Mean Wait (min)": f"{m_opt.mean_wait_minutes:.2f}",
            "Median Wait (min)": f"{m_opt.median_wait_minutes:.2f}"
        }
    ])
    st.dataframe(inv_df, use_container_width=True, hide_index=True)

    # Simulation Charts
    st.markdown("#### 📈 Simulation Trajectories & Distributions")
    ch_col1, ch_col2 = st.columns(2)

    with ch_col1:
        fig_q = theme.plot_queue_length_timeline(df_q_base, df_q_opt)
        st.plotly_chart(fig_q, use_container_width=True)

    with ch_col2:
        fig_dist = theme.plot_simulation_wait_distribution(df_p_base, df_p_opt)
        st.plotly_chart(fig_dist, use_container_width=True)

    # Detailed Patient Event Log
    st.markdown("#### 📜 Synthetic Individual Patient Event Log (Sample)")
    st.caption("Shows individual synthetic patient arrival timestamps, service start/end, and waiting times.")
    st.dataframe(
        df_p_opt[[
            "patient_id", "slot_name", "arrival_time_hr", "service_start_hr", "service_end_hr",
            "waiting_time_min", "service_time_min", "completed"
        ]].head(50),
        use_container_width=True,
        hide_index=True
    )

st.markdown("---")

# Synthetic Data Tools & CSV Import/Export
st.markdown("### 💾 Synthetic Patient Generator & Data Tools")
gen_col1, gen_col2 = st.columns(2)

with gen_col1:
    st.markdown("#### Generate & Download Synthetic Patient Stream")
    st.markdown("Generates a synthetic Poisson arrival stream with exponential service times across current shifts.")
    gen_seed = st.number_input("Generator Seed:", min_value=1, value=42, key="gen_seed")
    if st.button("Generate Synthetic Patient CSV", type="secondary"):
        df_synth = generate_synthetic_patient_stream(current_slots, random_seed=gen_seed)
        csv_synth = export_dataframe_to_csv(df_synth)
        st.download_button(
            label=f"📥 Download {len(df_synth)} Synthetic Patients (CSV)",
            data=csv_synth,
            file_name="smartcare_synthetic_patients.csv",
            mime="text/csv"
        )
        st.dataframe(df_synth.head(10), use_container_width=True, hide_index=True)

with gen_col2:
    st.markdown("#### Import & Validate External Patient CSV")
    st.markdown("Upload a CSV with `arrival_time_hour` (or `arrival_time_min`) and `service_duration_min`.")
    uploaded_file = st.file_uploader("Upload Patient Arrival CSV:", type=["csv"])
    if uploaded_file is not None:
        is_valid, df_valid, msg = validate_imported_patient_csv(uploaded_file.getvalue())
        if is_valid:
            st.success(f"✅ {msg}")
            st.dataframe(df_valid.head(10), use_container_width=True, hide_index=True)
        else:
            st.error(f"❌ Validation Error: {msg}")
