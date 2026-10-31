# SmartCare OR — Experimental Results & Benchmark Report

**Project Title:** SmartCare OR — Intelligent Hospital Patient Wait-Time Optimization System  
**Evaluation Standard:** Operations Research Academic Evaluation  
**Execution Environment:** macOS, Python 3.14.6, PuLP 2.9.0 (CBC Solver), SimPy 4.1.2  
**Experimental Design:** 5 Replications per scenario, Controlled Common Random Numbers (Seed: 42), Warm-up: 0.5 hours  

---

## 1. Executive Summary of Experimental Findings

Across three standardized hospital operational demand tiers (Low, Moderate, and Peak), the application of Operations Research techniques demonstrated:
1. **Identification of Baseline Instabilities:** In Scenarios B and C, static baseline doctor schedules resulted in mathematically unstable queues ($\rho \ge 1.0$), causing queue lengths to explode over time.
2. **Integer Optimization Effectiveness:** PuLP ILP optimization with capacity safety margins resolved all instability by dynamically allocating doctors to afternoon and evening demand surges.
3. **Discrete-Event Validation:** Simulation validated waiting-time reductions of **66.6%** in standard moderate demand and **86.5%** in peak demand.
4. **Law of Diminishing Returns:** Adding doctors beyond the optimal capacity margin yielded progressively smaller wait reductions relative to staffing expenditure.

---

## 2. Quantitative Benchmark Comparison Table

All values in the table below were generated directly by running the automated benchmark suite.

| Scenario | Policy / Method | Data Source | Doctors (Total) | Daily Cost ($) | Avg Wait (min) | Server Utilization | Stability Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Scenario A** | Baseline Staffing | Analytical M/M/c | 6 | $2,160.00 | 29.66 | 80.0% | Stable |
| *(Low Demand)* | ILP Optimized | PuLP ILP + M/M/c | 6 | $2,160.00 | 29.66 | 80.0% | Optimal |
| | Baseline Sim | SimPy (5 reps, seed 42) | 6 | $2,160.00 | 10.68 | 70.4% | Simulated |
| | Optimized Sim | SimPy (5 reps, seed 42) | 6 | $2,160.00 | 10.68 | 70.4% | Simulated |
| **Scenario B** | Baseline Staffing | Analytical M/M/c | 9 | $3,240.00 | **Unstable** | **104.4%** | **Unstable ($\rho \ge 1$)** |
| *(Moderate / Std)* | ILP Optimized | PuLP ILP + M/M/c | 11 | $3,960.00 | 26.67 | 86.1% | Optimal |
| | Wait-Bounded ILP | Wait-Target ILP | 13 | $4,680.00 | 4.58 | 72.3% | Optimal |
| | Baseline Sim | SimPy (5 reps, seed 42) | 9 | $3,240.00 | 19.75 | 88.8% | Simulated |
| | Optimized Sim | SimPy (5 reps, seed 42) | 11 | $3,960.00 | **6.60** | 78.0% | **-66.6% Wait** |
| **Scenario C** | Baseline Staffing | Analytical M/M/c | 12 | $4,320.00 | **Unstable** | **113.3%** | **Unstable ($\rho \ge 1$)** |
| *(Peak Demand)* | ILP Optimized | PuLP ILP + M/M/c | 16 | $5,760.00 | 10.40 | 84.9% | Optimal |
| | Baseline Sim | SimPy (5 reps, seed 42) | 12 | $4,320.00 | 32.68 | 92.5% | Simulated |
| | Optimized Sim | SimPy (5 reps, seed 42) | 16 | $5,760.00 | **4.42** | 76.7% | **-86.5% Wait** |

---

## 3. Detailed Scenario Analysis

### 3.1 Scenario A: Low Demand ($\lambda = [7, 9, 8]$ pts/hr, $\mu = 5.0$)
- **Baseline Allocation:** 2 doctors in Morning, 2 in Afternoon, 2 in Evening (Total: 6 doctors).
- **Optimizer Decision:** The PuLP solver selected $\{ \text{Morning}: 2, \text{Afternoon}: 2, \text{Evening}: 2 \}$.
- **Finding:** Because the baseline allocation already satisfies the safety margin $\mu x_t \ge \lambda_t + 1.0$ at minimal integer cost, the optimizer retained the baseline schedule. Wait reduction is exactly $0.0\%$.
- **Academic Significance:** Proves that the optimization model does not artificially inflate or invent improvements when the baseline configuration is already mathematically optimal.

### 3.2 Scenario B: Moderate Demand ($\lambda = [14, 18, 15]$ pts/hr, $\mu = 5.0$)
- **Baseline Allocation:** 3 doctors across all shifts (Total: 9 doctors).
- **The Bottleneck:** In the afternoon shift (12:00–15:00), arrival rate is 18 pts/hr while 3 doctors provide only $3 \times 5 = 15$ pts/hr capacity ($\rho = 1.20$). Steady-state queueing theory flags this shift as **strictly unstable**.
- **Optimizer Decision:** The ILP allocated 3 doctors to Morning ($15 \ge 14+1$), 4 doctors to Afternoon ($20 \ge 18+1$), and 4 doctors to Evening ($20 \ge 15+1$) for a total of 11 doctors.
- **Simulation Validation:** In discrete-event simulation, average waiting time dropped from **19.75 minutes to 6.60 minutes**, achieving a **66.6% reduction**.
- **Statistical Significance:** Using Common Random Numbers (CRN), the paired 95% confidence interval of the reduction is $[6.41, 19.89]$ minutes ($p < 0.05$).

### 3.3 Scenario C: Peak Emergency Demand ($\lambda = [20, 26, 22]$ pts/hr, $\mu = 5.0$)
- **Baseline Allocation:** 4 doctors per shift (Total: 12 doctors).
- **The Bottleneck:** Severe afternoon overload ($26 > 20$ pts/hr, utilization 130%).
- **Optimizer Decision:** The ILP allocated 5 doctors to Morning ($25 \ge 20+1.5$), 6 doctors to Afternoon ($30 \ge 26+1.5$), and 5 doctors to Evening ($25 \ge 22+1.5$) for a total of 16 doctors.
- **Simulation Validation:** Average patient wait time dropped from **32.68 minutes to 4.42 minutes** (**86.5% reduction**). The 95% paired confidence interval is $[20.40, 36.12]$ minutes.

---

## 4. Analytical vs. Simulation Discrepancies Explained

Evaluators should note the expected differences between analytical steady-state M/M/c numbers and simulated metrics:
1. **Unstable Queues in Finite Horizons:** Analytical queueing theory reports infinite wait times ($W_q = \infty$) when $\rho \ge 1$. However, a finite-horizon simulation (e.g. 9 hours) reports a finite wait time (e.g., 19.75 min in Scenario B) because queues require time to accumulate.
2. **Warm-up and Transients:** Hospitals begin the day with empty waiting rooms. A warm-up period (e.g. 0.5 hours) was excluded from post-warmup metrics to reflect operational equilibrium.
3. **Non-Preemptive Shift Handover:** In SimPy, departing doctors finish active patient consultations, providing transient capacity buffers during shift handovers.

---

## 5. Cost-Benefit Trade-Off Frontier

In Scenario B:
- Baseline Cost: **$3,240.00** (Avg Wait: 19.75 min)
- ILP Cost-Minimizing Cost: **$3,960.00** (Avg Wait: 6.60 min) — Cost increase: $720.00 to save 13.15 min wait time (**$54.75 per minute of wait reduction across ~140 patients**).
- Wait-Bounded ILP Cost: **$4,680.00** (Avg Wait: 4.58 min) — Cost increase: $1,440.00 to achieve near-zero waiting.

This provides hospital administrators with an explicit, quantifiable menu of operational trade-offs between budgetary expenditure and patient quality of service.
