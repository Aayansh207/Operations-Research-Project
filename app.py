
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from src.data.schemas import TimeSlot, OptimizationResult
from src.models.queueing import calculate_mmc
from src.models.scheduling import optimize_staff_ilp, evaluate_schedule


# ------------------------- CONFIGURATION -------------------------

st.set_page_config(
    page_title="SmartCare OR",
    page_icon="🏥",
    layout="wide"
)


# ------------------------- HELPER FUNCTIONS -------------------------

def mmc(arrival, service, doctors):
    return calculate_mmc(
        arrival_rate=float(arrival),
        service_rate=float(service),
        num_servers=int(doctors)
    )


def make_slots(arrivals, service, margin, baseline):
    times = {
        "Morning": (9, 12),
        "Afternoon": (12, 15),
        "Evening": (15, 18)
    }

    return [
        TimeSlot(
            name, start, end,
            arrival_rate=arrivals[name],
            service_rate=service,
            min_doctors=1,
            max_doctors=8,
            cost_per_doctor_hour=100.0,
            capacity_margin=margin,
            baseline_doctors=baseline[name]
        )
        for name, (start, end) in times.items()
    ]


def show_queue_status(result, arrival, service, doctors):
    capacity = service * doctors

    if not result.is_stable:
        required = int(np.floor(arrival / service) + 1)

        st.error(
            f"Unstable queue! Demand: {arrival:.1f}/hr, "
            f"Capacity: {capacity:.1f}/hr. "
            f"At least {required} doctors are required."
        )

    elif result.utilization >= 0.85:
        st.warning(
            f"High congestion risk. Current utilization: "
            f"{result.utilization:.1%}"
        )

    else:
        st.success(
            f"Queue is stable. Capacity: {capacity:.1f}/hr, "
            f"Demand: {arrival:.1f}/hr, "
            f"Utilization: {result.utilization:.1%}"
        )


# ------------------------- SIDEBAR -------------------------

st.sidebar.title("🏥 SmartCare OR")

page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Queueing Analysis", "Staff Optimization"]
)

st.sidebar.markdown("---")
st.sidebar.caption("Technique 1: M/M/c Queueing Theory")
st.sidebar.caption("Technique 2: Integer Linear Programming")


# ========================= OVERVIEW =========================

if page == "Overview":

    st.title("SmartCare OR")
    st.subheader("Hospital Patient Wait-Time Optimization")

    st.write(
        "An Operations Research based system to analyze patient "
        "waiting times and optimize doctor allocation."
    )

    st.markdown("---")

    st.subheader("Techniques Used")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 1. Queueing Theory (M/M/c)")
        st.write(
            "Analyzes patient arrivals, doctor service rates, "
            "queue lengths and waiting times."
        )

    with col2:
        st.markdown("### 2. Integer Linear Programming")
        st.write(
            "Optimizes doctor allocation across different shifts "
            "while satisfying capacity constraints."
        )

    st.markdown("---")

    st.subheader("Illustrative Hospital Scenario")

    result = mmc(14, 5, 3)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Patient Arrivals", "14 / hour")
    c2.metric("Service Rate", "5 / doctor / hour")
    c3.metric("Doctors", "3")
    c4.metric(
        "Average Waiting Time",
        f"{result.Wq_minutes:.2f} min" if result.is_stable else "Unstable"
    )

    show_queue_status(result, 14, 5, 3)


# ========================= QUEUEING ANALYSIS =========================

elif page == "Queueing Analysis":

    st.title("M/M/c Queueing Analysis")

    st.write(
        "Analyze patient waiting times based on arrival rates, "
        "service rates and available doctors."
    )

    st.markdown("---")

    st.subheader("Configure Parameters")

    col1, col2, col3 = st.columns(3)

    with col1:
        arrival = st.slider(
            "Arrival Rate (λ)",
            1.0, 35.0, 14.0, 0.5
        )

    with col2:
        service = st.slider(
            "Service Rate (μ)",
            1.0, 12.0, 5.0, 0.5
        )

    with col3:
        doctors = st.slider(
            "Number of Doctors (c)",
            1, 10, 3
        )

    result = mmc(arrival, service, doctors)

    st.markdown("---")

    st.subheader("Queue Performance")

    show_queue_status(result, arrival, service, doctors)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Utilization (ρ)", f"{result.utilization:.2%}")

    c2.metric(
        "Average Waiting Time",
        f"{result.Wq_minutes:.2f} min"
        if result.is_stable else "∞"
    )

    c3.metric(
        "Average Queue Length",
        f"{result.Lq:.2f}"
        if result.is_stable else "∞"
    )

    c4.metric(
        "Probability of Waiting",
        f"{result.p_wait:.2%}"
        if result.is_stable else "100%"
    )

    st.markdown("---")

    st.subheader("Sensitivity Analysis")

    col1, col2 = st.columns(2)

    # Waiting time vs number of doctors
    with col1:

        counts = list(range(1, 9))
        waits = []

        for n in counts:
            r = mmc(arrival, service, n)

            waits.append(
                r.Wq_minutes if r.is_stable else None
            )

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=counts,
            y=waits,
            name="Waiting Time"
        ))

        fig.update_layout(
            title="Waiting Time vs Number of Doctors",
            xaxis_title="Number of Doctors",
            yaxis_title="Waiting Time (minutes)"
        )

        st.plotly_chart(fig, use_container_width=True)

    # Waiting time vs arrival rate
    with col2:

        rates = np.linspace(1, max(arrival * 1.5, service * doctors * 1.1), 70)

        waits = []

        for rate in rates:
            r = mmc(rate, service, doctors)

            waits.append(
                r.Wq_minutes if r.is_stable else None
            )

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=rates,
            y=waits,
            mode="lines",
            name="Waiting Time"
        ))

        fig.add_vline(
            x=service * doctors,
            line_dash="dash",
            annotation_text="Maximum Capacity"
        )

        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Mathematical Formulation"):

        st.latex(r"\rho = \frac{\lambda}{c\mu}")
        st.latex(r"L_q = \lambda W_q")

        st.write("Where:")
        st.write("- λ: Patient arrival rate")
        st.write("- μ: Doctor service rate")
        st.write("- c: Number of doctors")
        st.write("- ρ: System utilization")
        st.write("- Lq: Average queue length")
        st.write("- Wq: Average waiting time")


# ========================= STAFF OPTIMIZATION =========================

elif page == "Staff Optimization":

    st.title("Staff Scheduling Optimization")

    st.write(
        "Optimize doctor allocation across morning, afternoon "
        "and evening shifts using Integer Linear Programming."
    )

    st.markdown("---")

    st.subheader("1. Configure Patient Demand")

    presets = {
        "Standard Outpatient Day": (12, 18, 14),
        "High Demand Day": (16, 26, 20),
        "Rural Clinic": (8, 13, 9)
    }

    preset = st.selectbox(
        "Select a demand scenario",
        list(presets.keys())
    )

    defaults = presets[preset]

    col1, col2, col3 = st.columns(3)

    arrivals = {}
    baseline = {}

    shifts = [
        ("Morning", "09:00 - 12:00", defaults[0]),
        ("Afternoon", "12:00 - 15:00", defaults[1]),
        ("Evening", "15:00 - 18:00", defaults[2])
    ]

    for col, (name, hours, default) in zip(
        [col1, col2, col3], shifts
    ):

        with col:

            st.markdown(f"### {name}")
            st.caption(hours)

            arrivals[name] = st.number_input(
                f"Patient arrivals / hour",
                min_value=0.5,
                max_value=40.0,
                value=float(default),
                step=0.5,
                key=f"arrival_{name}"
            )

            baseline[name] = st.number_input(
                "Baseline doctors",
                min_value=1,
                max_value=8,
                value=2,
                key=f"baseline_{name}"
            )

    st.markdown("---")

    st.subheader("2. Configure Constraints")

    col1, col2 = st.columns(2)

    with col1:
        service = st.slider(
            "Service Rate (patients/hour/doctor)",
            2.0, 10.0, 5.0, 0.5
        )

    with col2:
        buffer = st.slider(
            "Additional Capacity Buffer",
            0.0, 5.0, 1.0, 0.5
        )

    slots = make_slots(
        arrivals,
        service,
        buffer,
        baseline
    )

    if st.button(
        "Run Staff Optimization",
        type="primary",
        use_container_width=True
    ):

        try:
            with st.spinner("Optimizing staff allocation..."):

                st.session_state["ilp_result"] = optimize_staff_ilp(slots)

                st.session_state["ilp_signature"] = (
                    tuple(arrivals.items()),
                    service,
                    buffer,
                    tuple(baseline.items())
                )

        except Exception as exc:
            st.error(f"Optimization failed: {exc}")

    # Run again automatically when parameters change
    signature = (
        tuple(arrivals.items()),
        service,
        buffer,
        tuple(baseline.items())
    )

    if st.session_state.get("ilp_signature") != signature:
        st.session_state.pop("ilp_result", None)

    if "ilp_result" not in st.session_state:

        st.info("Configure the parameters and run optimization.")

    else:

        opt: OptimizationResult = st.session_state["ilp_result"]

        baseline_eval = evaluate_schedule(
            slots,
            baseline
        )

        allocation = {
            s.name: int(opt.allocation.get(s.name, s.baseline_doctors))
            for s in slots
        }

        total_base = sum(baseline.values())
        total_opt = sum(allocation.values())

        st.markdown("---")

        st.subheader("3. Optimization Results")

        c1, c2, c3 = st.columns(3)

        c1.metric("Baseline Doctors", total_base)
        c2.metric("Optimized Doctors", total_opt)
        c3.metric("Doctors Saved", total_base - total_opt)

        rows = []

        for slot in slots:

            name = slot.name

            bq = baseline_eval.slot_queueing_metrics[name]
            oq = opt.slot_queueing_metrics[name]

            rows.append({
                "Shift": name,
                "Demand / hour": slot.arrival_rate,
                "Baseline Doctors": baseline[name],
                "Optimized Doctors": allocation[name],
                "Baseline Wait (min)": (
                    round(bq.Wq_minutes, 2) if bq.is_stable else "Unstable"
                ),
                "Optimized Wait (min)": (
                    round(oq.Wq_minutes, 2) if oq.is_stable else "Unstable"
                ),
                "Optimized Utilization": f"{oq.utilization:.2%}"
            })

        df = pd.DataFrame(rows)

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("---")

        st.subheader("Staff Allocation Comparison")

        fig = go.Figure()

        fig.add_bar(
            x=df["Shift"],
            y=df["Baseline Doctors"],
            name="Baseline"
        )

        fig.add_bar(
            x=df["Shift"],
            y=df["Optimized Doctors"],
            name="Optimized"
        )

        fig.update_layout(
            barmode="group",
            yaxis_title="Number of Doctors"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Recommended Staffing Plan")

        cols = st.columns(3)

        for col, slot in zip(cols, slots):

            with col:

                q = opt.slot_queueing_metrics[slot.name]

                st.metric(
                    slot.name,
                    f"{allocation[slot.name]} doctors",
                    f"Utilization: {q.utilization:.1%}"
                )

        with st.expander("Mathematical Formulation"):

            st.write("Objective: Minimize total doctors assigned.")

            st.latex(r"\min \sum_{t \in T} x_t")

            st.write("Capacity constraint:")

            st.latex(
                r"\mu x_t \geq \lambda_t + \delta"
            )

            st.latex(r"x_t \in \mathbb{Z}_{+}")

        st.download_button(
            "Download Results CSV",
            df.to_csv(index=False),
            file_name="staffing_optimization.csv",
            mime="text/csv"
        )


