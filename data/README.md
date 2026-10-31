# SmartCare OR — Data Architecture & Formats

## 1. Overview
SmartCare OR is designed to operate seamlessly without requiring sensitive protected health information (PHI) or proprietary hospital databases.
The platform includes:
1. A **Synthetic Patient Stream Generator** modeling non-homogeneous Poisson arrival processes with exponential consultation service times.
2. A **CSV Importer and Validator** supporting user-supplied arrival traces.
3. Three **Predefined Benchmark Scenarios** (Low Demand, Moderate Demand, and Peak Demand).

---

## 2. Predefined Scenarios (Synthetic Parameters)

| Scenario | Shift | Hours | Arrival Rate $\lambda$ (pts/hr) | Service Rate $\mu$ (pts/hr/doc) | Baseline Doctors | Optimized Doctors | Shift Duration |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Scenario A: Low Demand** | Morning | 09:00–12:00 | 7.0 | 5.0 | 2 | 2 | 3.0 hrs |
| | Afternoon | 12:00–15:00 | 9.0 | 5.0 | 2 | 2 | 3.0 hrs |
| | Evening | 15:00–18:00 | 8.0 | 5.0 | 2 | 2 | 3.0 hrs |
| **Scenario B: Moderate Demand** | Morning | 09:00–12:00 | 14.0 | 5.0 | 3 | 3 | 3.0 hrs |
| *(Hospital Standard)* | Afternoon | 12:00–15:00 | 18.0 | 5.0 | 3 (Unstable!) | 4 | 3.0 hrs |
| | Evening | 15:00–18:00 | 15.0 | 5.0 | 3 | 4 | 3.0 hrs |
| **Scenario C: Peak Demand** | Morning | 09:00–12:00 | 20.0 | 5.0 | 4 | 5 | 3.0 hrs |
| *(Emergency Congestion)* | Afternoon | 12:00–15:00 | 26.0 | 5.0 | 4 (Unstable!) | 6 | 3.0 hrs |
| | Evening | 15:00–18:00 | 22.0 | 5.0 | 4 | 5 | 3.0 hrs |

*All parameters are synthetic and designed for academic demonstration and evaluation.*

---

## 3. CSV Import Format
To import external arrival records, format your CSV file with the following headers:

```csv
arrival_time_hour,service_duration_min
9.12,14.5
9.25,8.2
9.31,19.0
...
```

### Validation Rules
- `arrival_time_hour` (or `arrival_time_min`): Must be non-negative numeric floats.
- `service_duration_min`: Must be strictly positive (> 0.0) numeric floats.
- No null or empty cells are accepted.
