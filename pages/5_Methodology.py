"""
SmartCare OR: Academic Methodology, Mathematical Foundations, and Literature.
Provides rigorous documentation of Operations Research formulations and verifiable citations.
"""

import streamlit as st

st.set_page_config(page_title="SmartCare OR — Methodology", page_icon="📚", layout="wide")

st.title("📚 Academic Methodology & Mathematical Formulations")
st.markdown("##### *Design and Development of Operations Research Techniques to Reduce Patient Waiting Times in Hospitals*")

st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. Problem & Objectives",
    "2. Queueing Theory (M/M/c)",
    "3. Integer Linear Programming",
    "4. Discrete-Event Simulation",
    "5. Academic References"
])

with tab1:
    st.markdown("""
    ### 1. Research Context & Problem Statement

    #### The Healthcare Congestion Dilemma
    Hospital emergency departments and outpatient specialty clinics frequently suffer from severe patient waiting delays,
    waiting room overcrowding, and operational congestion. These delays stem from:
    1. **Time-Varying Patient Demand:** Patient arrivals fluctuate sharply across the day (e.g., morning surges, post-lunch peaks).
    2. **Mismatched Service Capacity:** Static or rigid doctor shift assignments leave some hours understaffed (causing queue blow-ups) and others overstaffed (wasting costly physician hours).
    3. **Severe Non-Linear Congestion:** In stochastic queueing systems, delays do not scale linearly with arrival rates; as utilization $\\rho \\to 1$, waiting times increase asymptotically towards infinity.

    Indiscriminately hiring more physicians is financially prohibitive, while understaffing threatens patient outcomes and causes extreme queue buildup.

    #### Research Objectives
    1. **Formulate an M/M/c analytical queueing model** to estimate steady-state patient waiting times and identify instability thresholds.
    2. **Formulate and solve an Integer Linear Program (ILP)** to determine optimal doctor shift allocations that satisfy capacity safety margins and budget limits.
    3. **Implement a Discrete-Event Simulation (DES)** in SimPy to empirically validate policy performance under non-preemptive shift transitions and finite operational horizons.
    4. **Evaluate performance metrics:** Quantify waiting-time reductions, doctor utilization balance, and the financial cost-benefit frontier.
    """)

with tab2:
    st.markdown("""
    ### 2. Technique 1: M/M/c Queueing Theory Formulation

    #### Model Assumptions (Kendall's Notation: M/M/c)
    - **Arrivals ($M$):** Stationary Poisson process with mean rate $\\lambda$ (patients per hour). Inter-arrival times are independent and exponentially distributed with parameter $\\lambda$.
    - **Service Times ($M$):** Exponentially distributed consultation times with mean $1/\\mu$ hours per doctor (service rate $\\mu$ patients/hour/doctor).
    - **Servers ($c$):** $c$ identical, independent parallel doctors drawing from a single central First-Come, First-Served (FCFS) waiting queue.
    - **Queue Capacity:** Infinite waiting room capacity ($c$ servers, $\\infty$ queue size, $\\infty$ calling population).
    - **Patient Behavior:** No balking (patients entering without queueing) and no reneging (patients abandoning the queue).

    #### Mathematical Equations

    **1. Traffic Intensity ($a$) & Server Utilization ($\\rho$):**
    $$a = \\frac{\\lambda}{\\mu}, \\qquad \\rho = \\frac{\\lambda}{c\\mu}$$

    **2. Steady-State Stability Criterion:**
    $$\\rho < 1.0 \\iff \\lambda < c \\mu$$
    If $\\lambda \\ge c\\mu$, the queue is unstable and queue length grows unbounded over an infinite horizon.

    **3. Probability of an Empty System ($P_0$):**
    $$P_0 = \\left[ \\sum_{n=0}^{c-1} \\frac{a^n}{n!} + \\frac{a^c}{c!(1 - \\rho)} \\right]^{-1}$$

    **4. Erlang C Formula — Probability of Waiting ($P_w$):**
    $$P_w = P(N \\ge c) = \\frac{a^c}{c!(1 - \\rho)} P_0$$

    **5. Expected Queue Length ($L_q$):**
    $$L_q = \\frac{P_w \\rho}{1 - \\rho} = \\frac{P_0 a^c \\rho}{c!(1 - \\rho)^2}$$

    **6. Little's Law — Expected Waiting Time in Queue ($W_q$):**
    $$W_q = \\frac{L_q}{\\lambda} \\quad (\\text{hours}) = \\frac{L_q}{\\lambda} \\times 60 \\quad (\\text{minutes})$$

    **7. Total Time in System ($W$) and Total Patients in System ($L$):**
    $$W = W_q + \\frac{1}{\\mu}, \\qquad L = \\lambda W = L_q + a$$

    #### Special Case: M/M/1 ($c = 1$)
    When $c = 1$:
    $$P_0 = 1 - \\rho, \\quad P_w = \\rho, \\quad W_q = \\frac{\\lambda}{\\mu(\\mu - \\lambda)}, \\quad W = \\frac{1}{\\mu - \\lambda}$$
    """)

with tab3:
    st.markdown("""
    ### 3. Technique 2: Optimal Staff Scheduling (PuLP Integer Linear Program)

    #### Problem Formulation
    Let $T = \\{1, 2, \\dots, |T|\\}$ denote the set of daily operational shifts (e.g., Morning, Afternoon, Evening).

    #### Parameters
    - $d_t$: Duration of shift $t$ in hours (e.g., $d_t = 3.0$).
    - $\\lambda_t$: Forecasted patient arrival rate during shift $t$ (patients/hour).
    - $\\mu_t$: Consultation service rate per doctor during shift $t$ (patients/hour/doctor).
    - $c_{\\min, t}, c_{\\max, t}$: Minimum and maximum doctor allocation limits for shift $t$.
    - $w_t$: Staffing cost per physician-hour during shift $t$ ($/doctor-hour).
    - $\\delta_t$: Capacity safety margin required above nominal arrival rate (patients/hour).
    - $W_{\\max}$: Maximum simultaneous physicians available (peak workforce limit).
    - $B$: Total daily hospital staffing expenditure budget ceiling ($).

    #### Decision Variables
    $$x_t \\in \\mathbb{Z}^+, \\quad \\forall t \\in T$$
    where $x_t$ is the integer number of doctors assigned to duty during shift $t$.

    #### Mathematical Optimization Model

    $$\\min_{x_t} \\quad Z = \\sum_{t \\in T} d_t w_t x_t$$

    **Subject to:**

    1. **Capacity Safety Margin Inequality (Linear):**
       $$\\mu_t x_t \\ge \\lambda_t + \\delta_t, \\quad \\forall t \\in T$$
       This guarantees that total service capacity strictly exceeds arrival rate:
       $$x_t \\ge \\left\\lceil \\frac{\\lambda_t + \\delta_t}{\\mu_t} \\right\\rceil$$

    2. **Staffing Bounds:**
       $$c_{\\min, t} \\le x_t \\le c_{\\max, t}, \\quad \\forall t \\in T$$

    3. **Peak Simultaneous Workforce Constraint:**
       $$x_t \\le W_{\\max}, \\quad \\forall t \\in T$$

    4. **Total Daily Staffing Budget Constraint:**
       $$\\sum_{t \\in T} d_t w_t x_t \\le B$$

    5. **Integrality Requirements:**
       $$x_t \\in \\{1, 2, 3, \\dots\\}, \\quad \\forall t \\in T$$

    #### Wait-Time Constrained Optimization Variant
    To enforce an explicit maximum waiting-time quality-of-service target ($W_q(t) \\le W_{\\text{target}}$):
    For each shift $t$, we compute the minimal $c_t^*$ satisfying $W_q(c_t^*, \\lambda_t, \\mu_t) \\le W_{\\text{target}}$ via the analytical M/M/c equation,
    and enforce $x_t \\ge c_t^*$ as a strengthened lower bound in the ILP formulation.
    """)

with tab4:
    st.markdown("""
    ### 4. Discrete-Event Patient Flow Simulation (SimPy)

    #### Simulation Purpose
    While steady-state queueing theory offers mathematical tractability, real hospital departments operate over **finite daily horizons**
    with **dynamic staffing transitions**. Discrete-event simulation evaluates operational policies without imposing steady-state equilibrium assumptions.

    #### Process Flow Architecture
    1. **Arrival Generator Process:**
       - Non-homogeneous Poisson Process (NHPP) with rate $\\lambda_t$ during shift $t$.
       - Patients arrive at discrete timestamp $t_{\\text{arr}}$, enter a single centralized `simpy.Store` FIFO queue, and update state $Q(t)$.
    2. **Doctor Worker Processes:**
       - Pull patients from FIFO queue upon becoming idle.
       - Service duration sampled from Exponential($\\mu_t$).
       - Active consultation timestamp recorded: $t_{\\text{start}}$ and $t_{\\text{end}}$.
    3. **Non-Preemptive Shift Downsizing:**
       - If shift $t+1$ requires fewer doctors than shift $t$, departing doctors complete their active consultation before clocking out.
       - No active patient consultation is preempted or interrupted.
    4. **Queue Continuity:**
       - Queues are never artificially purged or reset at shift boundaries; patients carry over smoothly across shifts.

    #### Statistical Metrics & Time-Weighted Integration
    - **Time-Weighted Average Queue Length:**
      $$\\bar{L}_q = \\frac{1}{T_{\\text{eval}}} \\int_0^{T_{\\text{eval}}} Q(t) \\, dt$$
      computed as the exact Riemann sum of step-function areas: $\\sum_i Q(t_i) (t_{i+1} - t_i)$.
    - **Doctor Utilization:**
      $$\\rho_{\\text{sim}} = \\frac{\\text{Total Doctor Active Consultation Hours}}{\\text{Total Scheduled Doctor Hours on Duty}}$$
    - **Common Random Numbers (CRN):**
      Baseline and optimized runs use synchronized random number seeds for arrival streams and service durations to eliminate random experimental noise and isolate the exact policy effect.
    """)

with tab5:
    st.markdown("""
    ### 5. Verifiable Academic Literature References

    1. **Hillier, F. S., & Lieberman, G. J. (2021).** *Introduction to Operations Research* (11th ed.). McGraw-Hill Education.
       - *Foundational chapters on Queueing Theory (M/M/c models, Erlang C, Little's Law) and Integer Linear Programming formulations.*

    2. **Kleinrock, L. (1975).** *Queueing Systems, Volume 1: Theory*. John Wiley & Sons.
       - *Rigorous mathematical derivations of Markovian queues, birth-death processes, and steady-state distributions.*

    3. **Winston, W. L. (2004).** *Operations Research: Applications and Algorithms* (4th ed.). Duxbury Press.
       - *Integer programming methodologies, workforce shift scheduling models, and branch-and-bound optimization.*

    4. **Hall, R. (Ed.). (2012).** *Handbook of Healthcare System Scheduling: Predictive Modeling to Optimize Operational Performance*. Springer.
       - *Operations research applications in hospital patient flow management, doctor shift allocation, and emergency department congestion.*

    5. **Law, A. M. (2015).** *Simulation Modeling and Analysis* (5th ed.). McGraw-Hill.
       - *Methodology for discrete-event simulation, warm-up period selection, output analysis, and Common Random Numbers (CRN) variance reduction.*
    """)
