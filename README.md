# SmartCare OR — Intelligent Hospital Patient Wait-Time Optimization System

[![Python Version](https://img.shields.io/badge/Python-3.11%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Optimization](https://img.shields.io/badge/Solver-COIN--OR%20CBC%20%7C%20PuLP-green.svg)](https://coin-or.github.io/pulp/)
[![Simulation](https://img.shields.io/badge/Simulation-SimPy%204-purple.svg)](https://simpy.readthedocs.io/)
[![Dashboard](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/Tests-24%20Passing-brightgreen.svg)](https://docs.pytest.org/)

**Academic Project Title:**  
*"Design and Development of Operations Research Techniques to Reduce Patient Waiting Times in Hospitals"*

---

## 1. Project Overview & Problem Statement

Emergency departments, outpatient clinics, and ambulatory surgery centers frequently encounter severe patient delays, waiting room overcrowding, and physician burnout. A primary operational root cause is the severe mismatch between **time-varying stochastic patient arrivals** and **inflexible doctor staffing schedules**. 

Because stochastic queues exhibit non-linear congestion surges as server utilization approaches capacity ($\rho \to 1.0$), minor understaffing during peak demand periods triggers catastrophic waiting-time spikes.

**SmartCare OR** is an interactive Operations Research (OR) software system and decision-support experimentation platform designed to model patient flow, optimize physician staffing across operational shifts, and evaluate performance trade-offs between patient waiting time and staffing expenditure.

> **Academic Notice:** SmartCare OR is designed for educational experimentation, simulation, and academic evaluation. All parameters are synthetic and illustrative; it is not a clinical diagnosis system and does not make medical decisions.

---

## 2. Research Objectives

1. **Analytical Queueing Modeling:** Formulate an exact $M/M/c$ queueing model to quantify steady-state waiting times, Erlang C probability of delay, and queue instability boundaries ($\rho \ge 1.0$).
2. **Integer Staff Scheduling:** Formulate an Integer Linear Program (ILP) using PuLP with the open-source COIN-OR CBC solver to determine optimal physician allocations across daily shifts subject to capacity safety margins, peak workforce limits, and budget ceilings.
3. **Discrete-Event Simulation Validation:** Develop a discrete-event simulation engine in SimPy to empirically validate proposed staffing policies under non-preemptive shift transitions and finite daily operational horizons.
4. **Trade-Off & Sensitivity Analysis:** Quantify the Pareto frontier between financial expenditure ($/day) and patient waiting-time reduction (minutes saved).
5. **Experimental Reproducibility:** Support reproducible evaluations with controlled random seeds, Common Random Numbers (CRN), and paired statistical confidence intervals.

---

## 3. Technology Stack

- **Language:** Python 3.11+ / Python 3.14 (Verified in Anaconda environment)
- **Interface & Multi-Page Dashboard:** Streamlit 1.58+
- **Mathematical Computation:** NumPy 2.4+, SciPy 1.18+, Pandas 3.0+
- **Integer Linear Programming:** PuLP 2.9+ with bundled COIN-OR CBC solver
- **Discrete-Event Simulation:** SimPy 4.1+
- **Data Visualization:** Plotly 6.7+
- **Automated Testing Suite:** pytest 9.0+

---

## 4. Software Architecture

```
OR_Project/
|-- app.py                          # Overview Dashboard & Live Demo preset
|
|-- pages/
|   |-- 1_Queueing_Theory.py        # Technique 1: M/M/c Analyzer & Sensitivity Curves
|   |-- 2_Staff_Optimization.py     # Technique 2: PuLP ILP Staff Scheduler & Pareto Analysis
|   |-- 3_Patient_Simulation.py     # Evaluation: SimPy Patient Flow Simulator & Event Logs
|   |-- 4_Experiment_Comparison.py  # Multi-Scenario Benchmark Suite & Statistical Reporting
|   |-- 5_Methodology.py            # Academic Foundations, Mathematical Derivations & Citations
|
|-- src/
|   |-- __init__.py
|   |-- models/
|   |   |-- queueing.py             # Exact M/M/c formulas, Erlang C, Little's Law, numerical stability
|   |   |-- scheduling.py           # PuLP ILP formulation, CBC solve, wait-time bounded optimizer
|   |-- simulation/
|   |   |-- patient_flow.py         # SimPy engine, non-preemptive shift handover, time-weighted Lq
|   |   |-- scenarios.py            # Predefined scenarios (Scenario A, B, C) and custom builders
|   |-- experiments/
|   |   |-- runner.py               # End-to-end experiment pipeline (Baseline vs ILP vs SimPy)
|   |   |-- metrics.py              # Wait reduction %, cost savings, paired t-test statistics
|   |-- data/
|   |   |-- schemas.py              # Typed dataclasses (TimeSlot, QueueingResult, PatientRecord)
|   |   |-- generators.py           # Synthetic arrival stream generator & CSV import/export
|   |-- utils/
|       |-- theme.py                # Publication-grade Plotly chart builders & healthcare theme
|       |-- session.py              # Multi-page Streamlit session state manager
|       |-- export.py               # JSON & CSV serialization utilities
|
|-- tests/
|   |-- test_queueing.py            # M/M/1 agreement, M/M/c known values, instability, zero arrival
|   |-- test_scheduling.py          # Integrality, safety margins, budget limits, infeasibility
|   |-- test_simulation.py          # Reproducibility, conservation of patients, chronological order
|   |-- test_metrics.py             # Reduction %, negative non-clamping, zero baseline handling
|   |-- test_experiments.py         # Full pipeline execution and serialization
|
|-- data/
|   |-- README.md                   # Data schemas, scenario parameters, and CSV format guidelines
|
|-- reports/
|   |-- methodology.md              # Mathematical derivations, proofs, and theoretical equations
|   |-- experimental_results.md     # Genuine, computed benchmark results and analytical findings
|
|-- requirements.txt                # Pinned dependency definitions
|-- pytest.ini                      # Test runner configuration
|-- .gitignore                      # Standard project gitignore
`-- README.md                       # Master documentation
```

---

## 5. Mathematical Formulations

### Technique 1: M/M/c Analytical Queueing Model
- **Offered Load:** $a = \frac{\lambda}{\mu}$
- **Utilization:** $\rho = \frac{\lambda}{c\mu}$
- **Stability Criterion:** $\rho < 1.0 \iff \lambda < c\mu$
- **Empty-System Probability ($P_0$):**
  $$P_0 = \left[ \sum_{n=0}^{c-1} \frac{a^n}{n!} + \frac{a^c}{c!(1 - \rho)} \right]^{-1}$$
- **Probability of Waiting (Erlang C):**
  $$P_w = \frac{a^c}{c!(1 - \rho)} P_0$$
- **Average Queue Length ($L_q$):**
  $$L_q = \frac{P_w \rho}{1 - \rho}$$
- **Little's Law (Waiting Time in Queue):**
  $$W_q = \frac{L_q}{\lambda} \times 60 \quad (\text{minutes})$$

### Technique 2: Optimal Staff Scheduling (PuLP ILP)
$$\min_{x_t} \quad Z = \sum_{t \in T} d_t w_t x_t$$
**Subject to:**
1. Capacity Safety Margin: $\mu_t x_t \ge \lambda_t + \delta_t \iff x_t \ge \left\lceil \frac{\lambda_t + \delta_t}{\mu_t} \right\rceil$
2. Physician Shift Bounds: $c_{\min, t} \le x_t \le c_{\max, t}$
3. Peak Workforce Limit: $x_t \le W_{\max}$
4. Budget Constraint: $\sum_{t \in T} d_t w_t x_t \le B$
5. Integrality: $x_t \in \mathbb{Z}^+$

---

## 6. Installation & Execution Guide

### Prerequisites
- Python 3.11+ installed locally.

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run Automated Test Suite
Run the 24 automated unit tests:
```bash
pytest -v tests/
```
*All 24 tests verify queueing mathematics, CBC solver execution, SimPy patient conservation, and statistical functions.*

### Step 3: Launch the Interactive Web Application
```bash
streamlit run app.py
```
Open your browser to the local URL (typically `http://localhost:8501`).

---

## 7. Live Demonstration Script (3 to 5 Minutes)

For academic evaluators, SmartCare OR provides a rapid demonstration workflow:
1. **Launch App:** Run `streamlit run app.py`.
2. **One-Click Live Demo:** Click the **`⚡ Live Demo Mode`** button in the header. The system loads Scenario B (Moderate Demand with peak afternoon bottleneck), executes the ILP solver, runs 3 SimPy replications with Common Random Numbers, and generates the baseline-vs-optimized comparison cards.
3. **Inspect Overview Dashboard:** Note how the baseline schedule has an unstable afternoon queue ($\rho = 104.4\%$), while the ILP allocation stabilizes the system ($86.1\%$ utilization) and slashes simulated wait time from **19.75 min to 6.60 min (66.6% reduction)**.
4. **Queueing Theory Module:** Open `1_Queueing_Theory` to view real-time Erlang C calculations and the non-linear capacity hockey-stick sensitivity curves.
5. **Staff Optimization Module:** Open `2_Staff_Optimization` to inspect the integer solver, adjust safety margins, and view the **Pareto Optimal Frontier** of Cost vs. Wait Time.
6. **Patient Flow Simulation:** Open `3_Patient_Simulation` to inspect the time-weighted queue timeline $Q(t)$, wait distribution histogram, and individual patient logs.
7. **Export Findings:** Download the complete benchmark dataset as CSV or JSON from the dashboard.

---

## 8. Summary of Experimental Results

| Scenario | Arrival Rate $\lambda$ | Baseline Doctors | Opt Doctors | Baseline Sim Wait | Opt Sim Wait | Wait Reduction | Paired 95% CI |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A: Low Demand** | $7, 9, 8$ pts/hr | 6 | 6 | 10.68 min | 10.68 min | **0.0%** | $[0.0, 0.0]$ min |
| **B: Moderate Demand** | $14, 18, 15$ pts/hr | 9 (Unstable) | 11 | 19.75 min | 6.60 min | **66.6%** | $[6.41, 19.89]$ min |
| **C: Peak Congestion** | $20, 26, 22$ pts/hr | 12 (Unstable) | 16 | 32.68 min | 4.42 min | **86.5%** | $[20.40, 36.12]$ min |

*Full analysis available in [reports/experimental_results.md](file:///Users/aayanshniranjan/Projects/OR_Project/reports/experimental_results.md).*

---

## 9. Assumptions and Limitations

1. **Analytical M/M/c Assumptions:** Assumes Poisson arrivals and Exponential service durations. Real clinical consultation times often exhibit log-normal distributions or Erlang stages (examined via simulation).
2. **Steady-State vs. Finite Horizon:** The analytical model assumes an infinite-horizon steady state; the SimPy simulation models finite daily operating hours with non-preemptive shift handovers.
3. **No Patient Abandonment:** The basic analytical model assumes patients do not renege or leave without consultation.
4. **Homogeneous Physicians:** All doctors within a shift are assumed to consult at the same average rate $\mu$.

---

## 10. Verifiable Academic References

1. **Hillier, F. S., & Lieberman, G. J. (2021).** *Introduction to Operations Research* (11th ed.). McGraw-Hill Education.
2. **Kleinrock, L. (1975).** *Queueing Systems, Volume 1: Theory*. John Wiley & Sons.
3. **Winston, W. L. (2004).** *Operations Research: Applications and Algorithms* (4th ed.). Duxbury Press.
4. **Hall, R. (Ed.). (2012).** *Handbook of Healthcare System Scheduling: Predictive Modeling to Optimize Operational Performance*. Springer.
5. **Law, A. M. (2015).** *Simulation Modeling and Analysis* (5th ed.). McGraw-Hill.
