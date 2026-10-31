"""
SmartCare OR: Technique 2 — Staff Scheduling Optimization.
Solves Integer Linear Programming (PuLP) staff allocation models with CBC solver,
evaluates queueing metrics, and generates Pareto cost-wait frontiers.
"""

import streamlit as st
import pandas as pd
import numpy as np

from src.data.schemas import TimeSlot
from src.models.scheduling import (
    optimize_staff_ilp,
    optimize_staff_wait_time_constrained,
    evaluate_schedule,
    generate_cost_vs_waittime_pareto
)
import src.utils.theme as theme
from src.utils.session import init_session_state, get_active_slots

init_session_state()

st.title("Technique 2: Optimal Staff Scheduling (Integer Programming)")
st.markdown("""
This module implements **Integer Linear Programming (ILP)** via **PuLP** (solved with **COIN-OR CBC**)
to mathematically allocate doctors across daily operational shifts.

Staffing decisions are subject to **capacity safety margins**, **peak workforce limitations**,
and **total budget constraints**.
""")

current_slots = get_active_slots()

# Slot Configuration Editor
st.markdown("### ⚙️ Operational Shift Configuration")
with st.expander("Edit Shift Arrival Rates, Service Rates, and Safety Margins", expanded=True):
    slot_configs = []
    cols = st.columns(len(current_slots))

    for idx, slot in enumerate(current_slots):
        with cols[idx]:
            st.markdown(f"#### **{slot.name} Shift**")
            st.caption(f"Hours: {slot.start_hour:02.0f}:00 – {slot.end_hour:02.0f}:00 ({slot.duration_hours:.0f} hrs)")
            arr = st.number_input(f"Arrival Rate ({slot.name})", min_value=1.0, max_value=50.0, value=float(slot.arrival_rate), step=1.0, key=f"arr_{slot.name}")
            srv = st.number_input(f"Service Rate ({slot.name})", min_value=1.0, max_value=20.0, value=float(slot.service_rate), step=0.5, key=f"srv_{slot.name}")
            margin = st.number_input(f"Safety Margin ({slot.name})", min_value=0.0, max_value=10.0, value=float(slot.capacity_margin), step=0.5, key=f"margin_{slot.name}")
            base_doc = st.number_input(f"Baseline Staff ({slot.name})", min_value=1, max_value=10, value=int(slot.baseline_doctors), step=1, key=f"base_{slot.name}")
            cost = st.number_input(f"Cost/Doc-Hr ({slot.name})", min_value=50.0, max_value=500.0, value=float(slot.cost_per_doctor_hour), step=10.0, key=f"cost_{slot.name}")

            slot_configs.append(TimeSlot(
                name=slot.name,
                start_hour=slot.start_hour,
                end_hour=slot.end_hour,
                arrival_rate=arr,
                service_rate=srv,
                min_doctors=slot.min_doctors,
                max_doctors=slot.max_doctors,
                cost_per_doctor_hour=cost,
                capacity_margin=margin,
                baseline_doctors=base_doc
            ))

# Global Constraints
st.markdown("### 🛡️ Global Workforce & Financial Constraints")
gc1, gc2, gc3 = st.columns(3)

with gc1:
    wf_limit = st.slider(
        "Peak Simultaneous Workforce Limit:",
        min_value=2,
        max_value=15,
        value=int(st.session_state["total_workforce_limit"]),
        help="Maximum doctors permitted on duty at any single time slot."
    )

with gc2:
    budget_limit = st.number_input(
        "Total Daily Staffing Budget ($):",
        min_value=1000.0,
        max_value=20000.0,
        value=float(st.session_state["budget_limit"]),
        step=250.0,
        help="Hard ceiling on cumulative daily staffing costs."
    )

with gc3:
    obj_mode = st.radio(
        "Optimization Objective:",
        options=["Minimize Staffing Cost (Standard ILP)", "Wait-Time Constrained (Wq <= Target)"],
        index=0
    )

if obj_mode == "Wait-Time Constrained (Wq <= Target)":
    target_wait = st.slider("Maximum Permitted Average Wait (minutes):", min_value=2.0, max_value=30.0, value=15.0, step=1.0)
else:
    target_wait = 15.0

st.markdown("---")

# Run Optimization Button
if st.button("⚡ Solve Optimization Model (PuLP CBC)", type="primary", use_container_width=True):
    with st.spinner("Formulating Mixed-Integer Linear Program and Executing CBC Solver..."):
        if obj_mode == "Minimize Staffing Cost (Standard ILP)":
            opt_res = optimize_staff_ilp(
                slots=slot_configs,
                total_workforce_limit=wf_limit,
                budget_limit=budget_limit
            )
        else:
            opt_res = optimize_staff_wait_time_constrained(
                slots=slot_configs,
                max_wait_minutes=target_wait,
                total_workforce_limit=wf_limit,
                budget_limit=budget_limit
            )

        st.session_state["active_opt_res"] = opt_res
        st.session_state["active_slot_configs"] = slot_configs

opt_res = st.session_state.get("active_opt_res", None)
saved_slots = st.session_state.get("active_slot_configs", slot_configs)

if opt_res:
    st.markdown("### 📋 Optimization Solution Results")

    if not opt_res.is_feasible:
        st.error(f"❌ **{opt_res.solver_message}**")
        st.warning("""
        **Troubleshooting Infeasibility:**
        1. Increase the maximum allowed doctors per slot or peak workforce limit.
        2. Increase the total daily staffing budget.
        3. If using Wait-Time Constrained optimization, relax the target waiting-time threshold.
        """)
    else:
        st.success(f"✅ **Solver Status: {opt_res.status}** — {opt_res.solver_message}")

        # Summary Metrics
        baseline_alloc = {s.name: s.baseline_doctors for s in saved_slots}
        baseline_eval = evaluate_schedule(saved_slots, baseline_alloc)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total Optimal Doctors", sum(opt_res.allocation.values()), delta=f"{sum(opt_res.allocation.values()) - sum(baseline_alloc.values())} vs baseline")
        with c2:
            st.metric("Total Staffing Cost", f"${opt_res.total_cost:,.0f}", delta=f"${opt_res.total_cost - baseline_eval.total_cost:,.0f} difference", delta_color="inverse")
        with c3:
            wait_str = f"{opt_res.average_wait_minutes:.2f} min" if opt_res.average_wait_minutes < 180 else "Congested"
            base_wait_str = f"{baseline_eval.average_wait_minutes:.2f}m" if baseline_eval.is_feasible else "Unstable"
            st.metric("Expected Wait Time", wait_str, delta=f"Baseline: {base_wait_str}")
        with c4:
            st.metric("Total Doctor Hours", f"{opt_res.total_doctor_hours:.1f} hrs")

        # Allocation Table
        alloc_rows = []
        for s in saved_slots:
            doc_opt = opt_res.allocation[s.name]
            q_m = opt_res.slot_queueing_metrics[s.name]
            cap = doc_opt * s.service_rate
            margin_actual = cap - s.arrival_rate
            alloc_rows.append({
                "Shift": s.name,
                "Arrival Rate": f"{s.arrival_rate:.1f} pts/hr",
                "Baseline Staff": s.baseline_doctors,
                "Optimized Staff": doc_opt,
                "Service Capacity": f"{cap:.1f} pts/hr",
                "Safety Margin Achieved": f"{margin_actual:+.1f} pts/hr (Req: +{s.capacity_margin:.1f})",
                "Utilization (rho)": f"{q_m.utilization:.1%}",
                "Expected Wait": f"{q_m.Wq_minutes:.1f} min" if q_m.is_stable else "Unstable",
                "Shift Cost ($)": f"${s.duration_hours * s.cost_per_doctor_hour * doc_opt:,.0f}"
            })
        st.dataframe(pd.DataFrame(alloc_rows), use_container_width=True, hide_index=True)

        # Optimization Charts
        st.markdown("#### 📊 Comparative Operational Charts")
        col_ch1, col_ch2 = st.columns(2)

        with col_ch1:
            fig_alloc = theme.plot_staffing_allocation_comparison(saved_slots, baseline_alloc, opt_res.allocation)
            st.plotly_chart(fig_alloc, use_container_width=True)

        with col_ch2:
            fig_demand = theme.plot_demand_vs_capacity(saved_slots, opt_res.allocation, label="Optimized")
            st.plotly_chart(fig_demand, use_container_width=True)

        # Pareto Frontier Analysis
        st.markdown("---")
        st.markdown("### 💎 Operations Research Pareto Frontier Analysis")
        st.markdown("""
        Evaluates the trade-off space across feasible integer allocations to compute the **Pareto Optimal Frontier**
        (allocations where no lower wait time can be achieved without increasing staffing expenditure).
        """)

        with st.spinner("Enumerating feasible schedule configurations..."):
            df_pareto = generate_cost_vs_waittime_pareto(saved_slots, total_workforce_limit=wf_limit)

        if not df_pareto.empty:
            fig_pareto = theme.plot_pareto_frontier(df_pareto)
            st.plotly_chart(fig_pareto, use_container_width=True)
            st.dataframe(
                df_pareto[df_pareto["is_pareto_optimal"]][[
                    "allocation_label", "total_cost", "total_doctor_hours", "avg_wait_minutes"
                ]].rename(columns={
                    "allocation_label": "Doctor Allocation (Shift:Count)",
                    "total_cost": "Total Daily Cost ($)",
                    "total_doctor_hours": "Doctor Hours",
                    "avg_wait_minutes": "Expected Arrival-Weighted Wait (min)"
                }),
                use_container_width=True,
                hide_index=True
            )
