# SmartCare OR — Hospital Patient Wait-Time Optimization

[![Python Version](https://img.shields.io/badge/Python-3.11%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Optimization](https://img.shields.io/badge/Solver-COIN--OR%20CBC%20%7C%20PuLP-green.svg)](https://coin-or.github.io/pulp/)
[![Dashboard](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/Tests-13%20Passing-brightgreen.svg)](https://docs.pytest.org/)

**Academic Problem Statement:**  
*"Design and Development of Operations Research Techniques to Reduce Patient Waiting Times in Hospitals"*

---

## 1. Project Overview & The Problem

Hospitals and outpatient clinics frequently suffer from **long patient wait times, overcrowded waiting rooms, and physician burnout**. 

A primary operational root cause is the **temporal mismatch between time-varying patient arrivals and rigid, uniform doctor staffing schedules**:
- Patient arrivals fluctuate throughout the day (e.g., quiet mornings, massive early afternoon surges).
- Hospitals often assign a flat, uniform number of doctors (e.g., 2 doctors across all shifts).
- In queueing systems, delays do **not** increase linearly. When patient arrivals exceed doctor capacity ($\rho = \frac{\lambda}{c\mu} \ge 1.0$), queue lengths and wait times **explode towards infinity**.

### The Solution: Two Operations Research (OR) Techniques

This project solves this problem using the two most foundational and effective Operations Research techniques:

1. **Technique 1: Queueing Theory ($M/M/c$ Model)**
   - Calculates the exact steady-state patient waiting time ($W_q$), queue length ($L_q$), and doctor utilization ($\rho$).
   - Flags when a shift is mathematically **unstable** ($\rho \ge 1.0$) and shows why adding doctors exponentially drops wait times.

2. **Technique 2: Linear Programming / Integer Programming (Staff Scheduling via PuLP)**
   - Takes time-varying patient arrival rates across daily shifts (Morning, Afternoon, Evening).
   - Formulates and solves an integer linear program to find the **minimum number of doctors** needed in each shift to guarantee that doctor capacity covers patient demand with a safety buffer.
   - Slashes patient wait times from hours/unstable down to under 5–10 minutes!

---

## 2. Mathematical Formulations

### Technique 1: M/M/c Queueing Theory Model

- **Traffic Intensity (Offered Load):** $a = \frac{\lambda}{\mu}$
- **Server Utilization:** $\rho = \frac{\lambda}{c \cdot \mu}$
- **Stability Criterion:** $\rho < 1.0 \iff \lambda < c \cdot \mu$  
  *(If $\lambda \ge c\mu$, the queue is unstable and wait time grows infinitely).*
- **Empty-System Probability ($P_0$):**
  $$P_0 = \left[ \sum_{n=0}^{c-1} \frac{a^n}{n!} + \frac{a^c}{c!(1 - \rho)} \right]^{-1}$$
- **Probability of Delay (Erlang C, $P_w$):**
  $$P_w = \frac{a^c}{c!(1 - \rho)} P_0$$
- **Average Queue Length ($L_q$):**
  $$L_q = \frac{P_w \cdot \rho}{1 - \rho}$$
- **Little's Law (Waiting Time in Queue, $W_q$):**
  $$W_q = \frac{L_q}{\lambda} \times 60 \quad (\text{minutes})$$

---

### Technique 2: Staff Scheduling (Integer Linear Programming)

$$\min_{x_t} \quad \sum_{t \in T} x_t \quad (\text{Minimize Total Doctors Scheduled})$$

**Subject to:**
1. **Capacity Safety Constraint:**
   $$\mu_t \cdot x_t \ge \lambda_t + \delta_t \quad \forall t \in T$$
   *(Guarantees doctor capacity covers arrival rate with a safety buffer $\delta_t$, keeping $\rho_t < 1.0$)*
2. **Integer Decision Variables:**
   $$x_t \in \{1, 2, 3, \dots\} \quad \forall t \in T$$

---

## 3. Installation & Quick Start

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run Unit Tests
Verify mathematical and optimization calculations:
```bash
pytest -v tests/
```
*(All 13 tests pass in < 1 second).*

### Step 3: Run the Web Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 4. How the App Works (3 Simple Tabs)

The user interface in `app.py` is structured into three clean, focused tabs:

- **Tab 1: Technique 1 — Queueing Theory (M/M/c)**
  - Enter Patient Arrival Rate ($\lambda$), Doctor Service Rate ($\mu$), and Number of Doctors ($c$).
  - Instantly see Utilization ($\rho$), Average Wait Time in minutes ($W_q$), Queue Length ($L_q$), and Erlang C probability.
  - Interactive sensitivity chart shows the dramatic drop in wait times as you add doctors.
- **Tab 2: Technique 2 — Staff Scheduling (Linear Programming via PuLP)**
  - Enter shift arrival rates (Morning, Afternoon peak, Evening).
  - Compare the **Baseline Policy** (flat 2 doctors per shift $\to$ afternoon bottleneck!) against the **LP Optimized Policy**.
  - Click **"🚀 Solve with Integer Linear Programming"** to calculate optimal doctors and see wait times drop by over 70%.
- **Tab 3: How to Explain This Project (Viva & Demo Guide)**
  - Complete presentation script, 30-second elevator pitch, and answers to common examiner questions.

---

## 5. File Structure

```
OR_Project/
|-- app.py                     # Clean, single-page Streamlit web application
|-- requirements.txt           # Project dependencies
|-- pytest.ini                 # Pytest configuration
|-- README.md                  # This file
|
|-- src/
|   |-- models/
|   |   |-- queueing.py        # Technique 1: M/M/c queueing formulas & Little's Law
|   |   `-- scheduling.py      # Technique 2: PuLP Integer Linear Program solver
|   |-- data/
|   |   `-- schemas.py         # TimeSlot, QueueingResult, OptimizationResult
|   `-- utils/                 # Helper utilities
|
|-- tests/
|   |-- test_queueing.py       # M/M/1 match, M/M/c benchmarks, stability tests
|   `-- test_scheduling.py     # Integer solver tests, capacity margin checks
|
`-- reports/
    |-- simple_project_guide.md # Comprehensive Viva & Explanation Handbook
    `-- methodology.md         # Mathematical proofs and theoretical background
```

---

## 6. How to Explain This in an Exam / Presentation (The 30-Second Pitch)

> *"In hospital clinics, patient arrivals surge during peak hours (like afternoon), but hospitals often keep a flat, uniform number of doctors across all shifts. This causes massive waiting bottlenecks.
> 
> We solved this using two classic Operations Research techniques:
> 1. **Queueing Theory ($M/M/c$):** To calculate patient wait times and identify when a shift becomes unstable ($\rho \ge 1.0$).
> 2. **Linear Programming (PuLP):** To calculate the exact minimum number of doctors needed in each shift so capacity safely exceeds demand.
> 
> As a result, wait times during peak hours are slashed from over an hour (or unstable queues) down to under 5 minutes without over-hiring!"*
