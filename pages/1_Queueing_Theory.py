"""
SmartCare OR: Technique 1 — Queueing Theory Analyzer.
Evaluates steady-state M/M/c queueing formulations and sensitivity curves.
"""

import streamlit as st
import numpy as np
import pandas as pd

from src.models.queueing import (
    calculate_mmc,
    calculate_mmc_sensitivity_servers,
    calculate_mmc_sensitivity_arrivals
)
import src.utils.theme as theme
from src.utils.session import init_session_state

init_session_state()

st.title("Technique 1: M/M/c Queueing Theory Analyzer")
st.markdown("""
This module implements the **M/M/c analytical queueing model**, providing exact steady-state calculations
for multi-server hospital outpatient and emergency service departments.
""")

# Parameter Controls
col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)

with col_ctrl1:
    arrival_rate = st.slider(
        "Patient Arrival Rate (lambda, pts/hr):",
        min_value=0.0,
        max_value=40.0,
        value=14.0,
        step=0.5,
        help="Mean number of patient arrivals per hour (Poisson process)."
    )

with col_ctrl2:
    service_rate = st.slider(
        "Doctor Service Rate (mu, pts/hr/doc):",
        min_value=1.0,
        max_value=15.0,
        value=5.0,
        step=0.5,
        help="Mean number of patients one doctor consults per hour (Exponential service time)."
    )

with col_ctrl3:
    num_servers = st.number_input(
        "Number of Doctors on Duty (c):",
        min_value=1,
        max_value=15,
        value=3,
        step=1,
        help="Number of identical parallel consulting doctors."
    )

# Compute Queueing Metrics
res = calculate_mmc(arrival_rate, service_rate, num_servers)

# Display Stability Banner
if not res.is_stable:
    st.error(f"⚠️ **{res.status_message}**")
else:
    st.success(f"✅ **System is Stable:** Server utilization is {res.utilization:.1%} (rho < 1.0). Steady-state equations hold.")

# Analytical Metrics Cards
st.markdown("### 📐 Steady-State Performance Metrics")
m1, m2, m3, m4, m5, m6 = st.columns(6)

with m1:
    st.metric(
        "Doctor Utilization (rho)",
        f"{res.utilization:.1%}",
        help="rho = lambda / (c * mu)"
    )

with m2:
    wait_str = f"{res.Wq_minutes:.2f} min" if res.is_stable else "Infinite"
    st.metric(
        "Avg Queue Wait (Wq)",
        wait_str,
        help="Expected waiting time before doctor consultation begins (Little's Law)."
    )

with m3:
    sys_str = f"{res.W_minutes:.2f} min" if res.is_stable else "Infinite"
    st.metric(
        "Total System Time (W)",
        sys_str,
        help="W = Wq + 1/mu (waiting time plus consulting duration)."
    )

with m4:
    lq_str = f"{res.Lq:.2f} pts" if res.is_stable else "Infinite"
    st.metric(
        "Avg Queue Length (Lq)",
        lq_str,
        help="Expected number of patients waiting in the waiting room."
    )

with m5:
    st.metric(
        "Prob of Waiting (Pw)",
        f"{res.p_wait:.1%}",
        help="Erlang C probability that an arriving patient finds all doctors busy."
    )

with m6:
    st.metric(
        "Empty System Prob (P0)",
        f"{res.p0:.1%}",
        help="Probability that zero patients are in the hospital."
    )

st.markdown("---")

# Sensitivity Analysis Visualizations
st.markdown("### 📈 Mathematical Sensitivity Curves")
tab1, tab2, tab3 = st.tabs([
    "1. Waiting Time vs. Doctor Staffing",
    "2. Demand Surge (Hockey-Stick Effect)",
    "3. Multi-Staffing Comparison Table"
])

with tab1:
    st.markdown("""
    Shows the law of **diminishing marginal returns** in healthcare capacity:
    Initial doctors stabilize the queue and drastically cut delays, while additional doctors yield diminishing wait-time reductions.
    """)
    df_sens_servers = calculate_mmc_sensitivity_servers(
        arrival_rate=arrival_rate,
        service_rate=service_rate,
        min_servers=1,
        max_servers=max(10, num_servers + 3)
    )
    fig_servers = theme.plot_wait_time_vs_doctors(df_sens_servers, arrival_rate, service_rate)
    st.plotly_chart(fig_servers, use_container_width=True)

with tab2:
    st.markdown("""
    Illustrates the non-linear **hockey-stick congestion surge**:
    As arrival rate $\\lambda$ approaches total service capacity $c\\mu$, queue length and waiting time explode asymptotically towards infinity.
    """)
    cap = num_servers * service_rate
    max_lam_plot = min(cap * 1.05, 50.0)
    df_sens_arr = calculate_mmc_sensitivity_arrivals(
        min_lambda=1.0,
        max_lambda=max_lam_plot,
        service_rate=service_rate,
        num_servers=num_servers,
        num_points=35
    )
    fig_arr = theme.plot_arrival_rate_vs_wait_time(df_sens_arr, capacity=cap)
    st.plotly_chart(fig_arr, use_container_width=True)

with tab3:
    st.markdown("#### Staffing Evaluation Matrix")
    st.dataframe(
        df_sens_servers.rename(columns={
            "servers": "Doctors (c)",
            "utilization": "Utilization (rho)",
            "waiting_time_min": "Avg Wait (min)",
            "system_time_min": "Avg Time in Hospital (min)",
            "queue_length": "Avg Waiting Patients (Lq)",
            "prob_wait": "Erlang C P(Wait)",
            "capacity": "Capacity (pts/hr)",
            "status": "Queue Stability"
        }),
        use_container_width=True,
        hide_index=True
    )

st.markdown("---")

# Academic Callout
with st.expander("📚 M/M/c Queueing Assumptions & Mathematical Derivations"):
    st.markdown("""
    **Analytical Assumptions:**
    1. **Arrival Distribution ($M$):** Interarrival times follow an independent Exponential distribution with mean $1/\\lambda$, yielding a Poisson arrival process.
    2. **Service Distribution ($M$):** Doctor consultation times follow an Exponential distribution with mean $1/\\mu$.
    3. **Parallel Servers ($c$):** $c$ identical, independent doctors sharing a single FIFO waiting queue.
    4. **Stationary Steady-State:** Long-run equilibrium requires server utilization $\\rho = \\frac{\\lambda}{c\\mu} < 1$.
    
    **Core Equations:**
    - Traffic Intensity: $a = \\frac{\\lambda}{\\mu}$
    - Server Utilization: $\\rho = \\frac{\\lambda}{c\\mu}$
    - Empty-System Probability:
      $$P_0 = \\left[ \\sum_{n=0}^{c-1} \\frac{a^n}{n!} + \\frac{a^c}{c!(1-\\rho)} \\right]^{-1}$$
    - Erlang C Formula (Probability of Waiting):
      $$P_w = \\frac{a^c}{c!(1-\\rho)} P_0$$
    - Average Queue Length:
      $$L_q = \\frac{P_w \\rho}{1 - \\rho}$$
    - Little's Law:
      $$W_q = \\frac{L_q}{\\lambda}, \\quad W = W_q + \\frac{1}{\\mu}, \\quad L = \\lambda W$$
    """)
