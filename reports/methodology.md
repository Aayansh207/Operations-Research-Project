# Operations Research Methodology and Mathematical Formulations
## SmartCare OR: Intelligent Hospital Patient Wait-Time Optimization System

**Author:** Operations Research Engineering Team  
**Evaluation:** Academic Evaluation in Operations Research  
**Date:** October 2026  

---

## 1. Executive Summary and Problem Formulation

Hospitals and outpatient clinics worldwide experience severe operational congestion characterized by long patient waiting times, crowded waiting rooms, and physician burnout. A primary root cause of this congestion is the temporal mismatch between **stochastic, time-varying patient arrivals** and **inflexible physician staffing schedules**. 

Because queueing congestion increases asymptotically as utilization approaches capacity, small imbalances in staffing during peak arrival surges can cause catastrophic waiting times. This project designs and evaluates two core Operations Research (OR) techniques:
1. **Technique 1: M/M/c Analytical Queueing Theory Model** to establish theoretical performance limits and identify instability thresholds.
2. **Technique 2: Integer Linear Programming (ILP)** solved via PuLP and the COIN-OR CBC solver to optimize doctor allocations across time-varying daily shifts.
3. **Discrete-Event Simulation (SimPy)** as an empirical evaluation engine to validate policy performance under dynamic shift handovers and finite-horizon non-equilibrium conditions.

---

## 2. Technique 1: M/M/c Queueing Theory Model

### 2.1 Theoretical Assumptions
1. **Poisson Arrivals:** Patient arrivals follow a stationary Poisson process with rate $\lambda$ (patients/hour). Interarrival times $A_i$ are independent, identically distributed (i.i.d.) Exponential random variables:
   $$f_A(t) = \lambda e^{-\lambda t}, \quad t \ge 0, \quad \mathbb{E}[A] = \frac{1}{\lambda}$$
2. **Exponential Service Times:** Physician consultation durations $S_i$ are i.i.d. Exponential random variables with rate $\mu$ (patients/hour/doctor):
   $$f_S(t) = \mu e^{-\mu t}, \quad t \ge 0, \quad \mathbb{E}[S] = \frac{1}{\mu}$$
3. **Identical Parallel Servers:** $c \ge 1$ identical doctors operating in parallel.
4. **Queue Discipline:** Single centralized First-Come, First-Served (FCFS) queue.
5. **No Reneging / Balking:** Patients wait until service without abandoning.
6. **Steady-State Stability:** Long-term equilibrium requires traffic utilization $\rho < 1$.

---

### 2.2 Analytical Equations and Derivations

Let:
- $\lambda$: Mean arrival rate (patients per hour)
- $\mu$: Mean service rate per doctor (patients per hour)
- $c$: Number of parallel doctors
- $a = \frac{\lambda}{\mu}$: Traffic intensity (offered load in Erlangs)
- $\rho = \frac{\lambda}{c\mu}$: Server utilization factor

#### Steady-State Stability Criterion
$$\rho = \frac{\lambda}{c\mu} < 1 \iff c > \frac{\lambda}{\mu}$$
When $\rho \ge 1$, the arrival rate exceeds or equals total service capacity. Under an infinite horizon, queue length diverges to infinity ($L_q \to \infty, W_q \to \infty$). SmartCare OR flags unstable states immediately and advises minimum staffing requirements.

#### Empty-System Probability ($P_0$)
Using the balance equations of the underlying birth-death Markov process:
$$P_0 = \left[ \sum_{n=0}^{c-1} \frac{a^n}{n!} + \frac{a^c}{c!(1 - \rho)} \right]^{-1}$$

#### Probability of an Arriving Patient Waiting (Erlang C Formula)
$$P_w = P(N \ge c) = \sum_{n=c}^\infty P_n = \frac{a^c}{c!(1 - \rho)} P_0$$

#### Average Queue Length ($L_q$)
$$L_q = \sum_{n=c}^\infty (n - c) P_n = \frac{P_0 a^c \rho}{c!(1 - \rho)^2} = \frac{P_w \rho}{1 - \rho}$$

#### Little's Law — Waiting Time in Queue ($W_q$) and System ($W$)
$$W_q = \frac{L_q}{\lambda} \quad (\text{hours}) = \frac{L_q}{\lambda} \times 60 \quad (\text{minutes})$$
$$W = W_q + \frac{1}{\mu} \quad (\text{total time in hospital})$$
$$L = \lambda W = L_q + a \quad (\text{total patients in system})$$

#### M/M/1 Special Case ($c = 1$)
For single-doctor clinics, substituting $c = 1$ yields:
$$P_0 = 1 - \rho, \quad P_w = \rho, \quad W_q = \frac{\lambda}{\mu(\mu - \lambda)}, \quad W = \frac{1}{\mu - \lambda}, \quad L_q = \frac{\lambda^2}{\mu(\mu - \lambda)}$$

---

## 3. Technique 2: Optimal Staff Scheduling via Integer Linear Programming

### 3.1 Problem Definition
Hospital operations are partitioned into daily shifts $T = \{1, 2, \dots, |T|\}$ (e.g., Morning 09:00–12:00, Afternoon 12:00–15:00, Evening 15:00–18:00). 

### 3.2 Parameters
- $d_t$: Duration of shift $t$ in hours.
- $\lambda_t$: Expected patient arrival rate during shift $t$ (patients/hour).
- $\mu_t$: Consultation rate per doctor during shift $t$ (patients/hour/doctor).
- $w_t$: Staffing wage/cost per doctor-hour during shift $t$ ($).
- $\delta_t$: Capacity safety margin required above nominal demand (patients/hour).
- $c_{\min, t}, c_{\max, t}$: Bounds on allowed doctors for shift $t$.
- $W_{\max}$: Peak simultaneous workforce limit across any shift.
- $B$: Total daily financial staffing budget ($).

### 3.3 Decision Variables
$$x_t \in \mathbb{Z}^+, \quad \forall t \in T$$
where $x_t$ represents the integer count of doctors scheduled on duty during shift $t$.

### 3.4 Mathematical Formulation (ILP)
$$\min_{x_t} \quad Z = \sum_{t \in T} d_t w_t x_t$$

**Subject to:**

1. **Capacity Safety Margin Constraint (Linear):**
   $$\mu_t x_t \ge \lambda_t + \delta_t, \quad \forall t \in T$$
   This linear integer inequality guarantees that service capacity exceeds expected demand with a buffer $\delta_t$:
   $$x_t \ge \left\lceil \frac{\lambda_t + \delta_t}{\mu_t} \right\rceil$$

2. **Physician Lower and Upper Bounds:**
   $$c_{\min, t} \le x_t \le c_{\max, t}, \quad \forall t \in T$$

3. **Peak Workforce Constraint:**
   $$x_t \le W_{\max}, \quad \forall t \in T$$

4. **Staffing Budget Ceiling:**
   $$\sum_{t \in T} d_t w_t x_t \le B$$

5. **Integrality Constraints:**
   $$x_t \in \{1, 2, 3, \dots\}, \quad \forall t \in T$$

### 3.5 Wait-Time Constrained Optimization Variant
In addition to the cost-minimizing ILP, SmartCare OR implements a quality-of-service constrained variant:
$$\min \sum_{t \in T} d_t w_t x_t \quad \text{s.t.} \quad W_q(x_t, \lambda_t, \mu_t) \le W_{\text{target}}, \quad \forall t \in T$$
Because $W_q$ is non-linear in $x_t$, the minimum integer staffing $c_t^*$ satisfying the target wait is determined analytically via the M/M/c equations and applied as a tightened lower bound ($x_t \ge c_t^*$) in the linear program.

---

## 4. Discrete-Event Patient Flow Simulation (SimPy)

### 4.1 Architecture and Event Mechanics
The simulation engine models the temporal dynamics of the hospital outpatient flow:
1. **Non-Homogeneous Poisson Arrival Process (NHPP):** Patients are generated dynamically with interarrival times sampled from $\text{Exponential}(\lambda_t)$.
2. **Central FIFO Waiting Store:** Patients enter a shared `simpy.Store` queue.
3. **Doctor Consultation Workers:** Idle doctors fetch patients from the queue and consult for $\text{Exponential}(\mu_t)$ duration.
4. **Non-Preemptive Shift Handover Policy:** When shift $t+1$ has fewer doctors than shift $t$, off-duty doctors complete their active patient consultation before clocking out. No patient consultation is preempted.
5. **Continuous Queue Dynamics:** The queue is never artificially reset at shift boundaries.

### 4.2 Statistical Integration
- **Time-Weighted Queue Length:**
  $$\bar{L}_q = \frac{1}{T_{\text{horizon}} - T_{\text{warmup}}} \int_{T_{\text{warmup}}}^{T_{\text{horizon}}} Q(t) \, dt$$
- **Server Utilization:**
  $$\rho_{\text{sim}} = \frac{\text{Cumulative Doctor Consultation Hours}}{\text{Total Scheduled Doctor Hours}}$$
- **Common Random Numbers (CRN):** Synchronized pseudo-random number generator seeds are utilized across baseline and optimized runs to isolate the exact treatment effect and reduce experimental variance.

---

## 5. Verified Academic Literature References

1. **Hillier, F. S., & Lieberman, G. J. (2021).** *Introduction to Operations Research* (11th ed.). McGraw-Hill Education.  
   *Foundational references for queueing theory (Chapters 17–18) and integer programming (Chapters 11–12).*
2. **Kleinrock, L. (1975).** *Queueing Systems, Volume 1: Theory*. John Wiley & Sons.  
   *Birth-death Markov processes and Erlang C formulations.*
3. **Winston, W. L. (2004).** *Operations Research: Applications and Algorithms* (4th ed.). Duxbury Press.  
   *Workforce scheduling, shift coverage models, and integer programming techniques.*
4. **Hall, R. (Ed.). (2012).** *Handbook of Healthcare System Scheduling: Predictive Modeling to Optimize Operational Performance*. Springer.  
   *Operations research applications in patient flow scheduling and hospital congestion reduction.*
5. **Law, A. M. (2015).** *Simulation Modeling and Analysis* (5th ed.). McGraw-Hill.  
   *Discrete-event simulation methodology, output analysis, warm-up periods, and Common Random Numbers (CRN).*
