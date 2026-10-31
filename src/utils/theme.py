"""
Theme and Plotly Visualization Builders for SmartCare OR.
Implements a refined, high-contrast healthcare analytics styling:
- Palette: Slate / Navy / Emerald / Coral / Amber
- Crisp typography and explicit units on all axes
- Professional tooltips, legends, and grid layouts
"""

from typing import List, Dict, Optional, Tuple, Any
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np


# Brand Palette
PRIMARY_BLUE = "#1E3A8A"      # Deep Navy
SECONDARY_BLUE = "#3B82F6"    # Vibrant Cobalt
SUCCESS_EMERALD = "#10B981"   # Emerald Green
WARNING_AMBER = "#F59E0B"     # Amber / Alert
DANGER_CORAL = "#EF4444"      # Crimson / Congested
PURPLE_ACCENT = "#8B5CF6"     # Purple
NEUTRAL_DARK = "#1F2937"      # Dark Slate for text
NEUTRAL_LIGHT = "#F3F4F6"     # Light Grey background
CARD_BORDER = "#E5E7EB"

PLOTLY_LAYOUT_DEFAULTS = dict(
    font=dict(family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif", size=12, color=NEUTRAL_DARK),
    plot_bgcolor="#FFFFFF",
    paper_bgcolor="#FFFFFF",
    margin=dict(l=50, r=30, t=50, b=50),
    xaxis=dict(
        showgrid=True,
        gridcolor="#F3F4F6",
        zeroline=True,
        zerolinecolor="#E5E7EB",
        linecolor="#D1D5DB"
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor="#F3F4F6",
        zeroline=True,
        zerolinecolor="#E5E7EB",
        linecolor="#D1D5DB"
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        bgcolor="rgba(255,255,255,0.8)"
    )
)


def apply_theme_layout(fig: go.Figure, **kwargs) -> go.Figure:
    """Applies standard layout theme and cleanly merges custom keyword arguments."""
    fig.update_layout(**PLOTLY_LAYOUT_DEFAULTS)
    if kwargs:
        fig.update_layout(**kwargs)
    return fig


def plot_wait_time_vs_doctors(df_sensitivity: pd.DataFrame, arrival_rate: float, service_rate: float) -> go.Figure:
    """Chart 1: Waiting time in queue vs number of doctors (diminishing returns)."""
    fig = go.Figure()

    stable_mask = df_sensitivity["is_stable"]
    unstable_mask = ~df_sensitivity["is_stable"]

    # Stable points
    fig.add_trace(go.Scatter(
        x=df_sensitivity.loc[stable_mask, "servers"],
        y=df_sensitivity.loc[stable_mask, "waiting_time_min"],
        mode="lines+markers",
        name="Average Wait Time (M/M/c)",
        line=dict(color=SECONDARY_BLUE, width=3),
        marker=dict(size=9, color=PRIMARY_BLUE),
        hovertemplate="Doctors: %{x}<br>Avg Wait: %{y:.2f} min<extra></extra>"
    ))

    # Unstable markers if any
    if unstable_mask.any():
        fig.add_trace(go.Scatter(
            x=df_sensitivity.loc[unstable_mask, "servers"],
            y=[0] * int(unstable_mask.sum()),
            mode="markers",
            name="Unstable System (rho >= 1)",
            marker=dict(symbol="x", size=12, color=DANGER_CORAL, line=dict(width=2)),
            hovertemplate="Doctors: %{x}<br>Status: UNSTABLE (Queue grows infinitely)<extra></extra>"
        ))

    apply_theme_layout(
        fig,
        title=f"<b>Average Waiting Time vs. Number of Doctors</b> (Demand: {arrival_rate:.1f} pts/hr, Service: {service_rate:.1f} pts/hr/doc)",
        xaxis_title="Number of Parallel Doctors (c)",
        yaxis_title="Average Queue Wait Time (minutes)"
    )
    fig.update_yaxes(rangemode="tozero")
    return fig


def plot_arrival_rate_vs_wait_time(df_arrivals: pd.DataFrame, capacity: float) -> go.Figure:
    """Chart 2: Arrival rate vs waiting time (congestive hockey-stick effect)."""
    fig = go.Figure()

    stable_df = df_arrivals[df_arrivals["is_stable"]]
    fig.add_trace(go.Scatter(
        x=stable_df["arrival_rate"],
        y=stable_df["waiting_time_min"],
        mode="lines",
        name="Stable Queue Wait Time",
        line=dict(color=PRIMARY_BLUE, width=3),
        hovertemplate="Arrival Rate: %{x:.1f} pts/hr<br>Avg Wait: %{y:.2f} min<extra></extra>"
    ))

    # Add vertical line for capacity limit
    fig.add_vline(
        x=capacity,
        line_width=2,
        line_dash="dash",
        line_color=DANGER_CORAL,
        annotation_text=f"Total Capacity: {capacity:.1f} pts/hr",
        annotation_position="top left"
    )

    apply_theme_layout(
        fig,
        title=f"<b>Waiting Time Surge as Demand Approaches Capacity</b> (Capacity = {capacity:.1f} pts/hr)",
        xaxis_title="Patient Arrival Rate (patients / hour)",
        yaxis_title="Average Queue Wait Time (minutes)"
    )
    fig.update_yaxes(rangemode="tozero")
    return fig


def plot_staffing_allocation_comparison(slots: List[Any], baseline_alloc: Dict[str, int], opt_alloc: Dict[str, int]) -> go.Figure:
    """Chart 3: Grouped bar chart comparing baseline vs optimized doctor counts across slots."""
    slot_names = [s.name for s in slots]
    base_counts = [baseline_alloc.get(name, 2) for name in slot_names]
    opt_counts = [opt_alloc.get(name, 2) for name in slot_names]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=slot_names,
        y=base_counts,
        name="Baseline Manual Allocation",
        marker_color="#94A3B8",
        text=base_counts,
        textposition="auto"
    ))
    fig.add_trace(go.Bar(
        x=slot_names,
        y=opt_counts,
        name="ILP Optimized Allocation",
        marker_color=SUCCESS_EMERALD,
        text=opt_counts,
        textposition="auto"
    ))

    apply_theme_layout(
        fig,
        barmode="group",
        title="<b>Doctor Allocation by Operational Time Slot</b>",
        xaxis_title="Shift / Time Slot",
        yaxis_title="Number of Assigned Doctors"
    )
    fig.update_yaxes(tickmode="linear", tick0=0, dtick=1)
    return fig


def plot_demand_vs_capacity(slots: List[Any], allocation: Dict[str, int], label: str = "Allocation") -> go.Figure:
    """Chart 4: Grouped bar chart comparing arrival rate demand vs service capacity per slot."""
    slot_names = [s.name for s in slots]
    demands = [s.arrival_rate for s in slots]
    capacities = [allocation.get(s.name, 1) * s.service_rate for s in slots]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=slot_names,
        y=demands,
        name="Expected Demand (lambda pts/hr)",
        marker_color=SECONDARY_BLUE,
        text=[f"{d:.1f}" for d in demands],
        textposition="auto"
    ))
    fig.add_trace(go.Bar(
        x=slot_names,
        y=capacities,
        name=f"Service Capacity ({label})",
        marker_color="#059669",
        text=[f"{c:.1f}" for c in capacities],
        textposition="auto"
    ))

    apply_theme_layout(
        fig,
        barmode="group",
        title=f"<b>Patient Demand vs. Service Capacity ({label})</b>",
        xaxis_title="Shift / Time Slot",
        yaxis_title="Patients per Hour"
    )
    return fig


def plot_slot_wait_time_comparison(
    slots: List[Any],
    baseline_q_metrics: Dict[str, Any],
    opt_q_metrics: Dict[str, Any]
) -> go.Figure:
    """Chart 5: Grouped bar chart comparing analytical waiting time by slot."""
    slot_names = [s.name for s in slots]
    base_waits = [
        round(baseline_q_metrics[name].Wq_minutes, 1) if baseline_q_metrics[name].is_stable else 120.0
        for name in slot_names
    ]
    opt_waits = [
        round(opt_q_metrics[name].Wq_minutes, 1) if opt_q_metrics[name].is_stable else 120.0
        for name in slot_names
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=slot_names,
        y=base_waits,
        name="Baseline Wait Time",
        marker_color=DANGER_CORAL,
        text=[f"{w:.1f}m" if baseline_q_metrics[s].is_stable else "Unstable" for s, w in zip(slot_names, base_waits)],
        textposition="auto"
    ))
    fig.add_trace(go.Bar(
        x=slot_names,
        y=opt_waits,
        name="Optimized Wait Time",
        marker_color=SUCCESS_EMERALD,
        text=[f"{w:.1f}m" if opt_q_metrics[s].is_stable else "Unstable" for s, w in zip(slot_names, opt_waits)],
        textposition="auto"
    ))

    apply_theme_layout(
        fig,
        barmode="group",
        title="<b>Expected Wait Time by Time Slot (M/M/c Evaluation)</b>",
        xaxis_title="Shift / Time Slot",
        yaxis_title="Waiting Time in Queue (minutes)"
    )
    return fig


def plot_simulation_wait_distribution(df_baseline: pd.DataFrame, df_optimized: Optional[pd.DataFrame] = None) -> go.Figure:
    """Chart 6: Histogram of simulated individual patient waiting times."""
    fig = go.Figure()

    base_completed = df_baseline[df_baseline["completed"] & (df_baseline["waiting_time_min"].notnull())]
    fig.add_trace(go.Histogram(
        x=base_completed["waiting_time_min"],
        name="Baseline Policy",
        marker_color="#DC2626",
        opacity=0.6,
        nbinsx=25,
        hovertemplate="Wait: %{x:.1f} min<br>Count: %{y}<extra></extra>"
    ))

    if df_optimized is not None and not df_optimized.empty:
        opt_completed = df_optimized[df_optimized["completed"] & (df_optimized["waiting_time_min"].notnull())]
        fig.add_trace(go.Histogram(
            x=opt_completed["waiting_time_min"],
            name="Optimized Policy",
            marker_color=SUCCESS_EMERALD,
            opacity=0.6,
            nbinsx=25,
            hovertemplate="Wait: %{x:.1f} min<br>Count: %{y}<extra></extra>"
        ))

    apply_theme_layout(
        fig,
        barmode="overlay",
        title="<b>Simulated Patient Wait-Time Distribution</b>",
        xaxis_title="Waiting Time (minutes)",
        yaxis_title="Number of Patients"
    )
    return fig


def plot_pareto_frontier(df_pareto: pd.DataFrame) -> go.Figure:
    """Chart 7: Trade-off scatter plot showing Staffing Cost vs Expected Wait Time."""
    fig = go.Figure()

    # All feasible allocations
    non_pareto = df_pareto[~df_pareto["is_pareto_optimal"]]
    fig.add_trace(go.Scatter(
        x=non_pareto["total_cost"],
        y=non_pareto["avg_wait_minutes"],
        mode="markers",
        name="Feasible Allocation",
        marker=dict(size=8, color="#94A3B8", opacity=0.7),
        text=non_pareto["allocation_label"],
        hovertemplate="Cost: $%{x:,.0f}<br>Avg Wait: %{y:.2f} min<br>Schedule: %{text}<extra></extra>"
    ))

    # Pareto optimal frontier
    pareto_pts = df_pareto[df_pareto["is_pareto_optimal"]].sort_values(by="total_cost")
    fig.add_trace(go.Scatter(
        x=pareto_pts["total_cost"],
        y=pareto_pts["avg_wait_minutes"],
        mode="lines+markers",
        name="Pareto Optimal Frontier",
        line=dict(color=PRIMARY_BLUE, width=3, dash="solid"),
        marker=dict(size=11, color=SUCCESS_EMERALD, symbol="diamond"),
        text=pareto_pts["allocation_label"],
        hovertemplate="<b>Pareto Optimal</b><br>Cost: $%{x:,.0f}<br>Avg Wait: %{y:.2f} min<br>Schedule: %{text}<extra></extra>"
    ))

    apply_theme_layout(
        fig,
        title="<b>Operations Research Trade-Off: Staffing Expenditure vs. Patient Wait Time</b>",
        xaxis_title="Total Daily Staffing Expenditure ($)",
        yaxis_title="Arrival-Weighted Expected Wait Time (minutes)"
    )
    return fig


def plot_queue_length_timeline(df_queue_baseline: pd.DataFrame, df_queue_opt: Optional[pd.DataFrame] = None) -> go.Figure:
    """Chart 8: Queue size over time step-plot."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_queue_baseline["sim_time_hr"],
        y=df_queue_baseline["queue_length"],
        mode="lines",
        name="Baseline Queue Q(t)",
        line=dict(shape="hv", color=DANGER_CORAL, width=2),
        hovertemplate="Time: %{x:.2f} hr<br>Queue Length: %{y} pts<extra></extra>"
    ))

    if df_queue_opt is not None and not df_queue_opt.empty:
        fig.add_trace(go.Scatter(
            x=df_queue_opt["sim_time_hr"],
            y=df_queue_opt["queue_length"],
            mode="lines",
            name="Optimized Queue Q(t)",
            line=dict(shape="hv", color=SUCCESS_EMERALD, width=2),
            hovertemplate="Time: %{x:.2f} hr<br>Queue Length: %{y} pts<extra></extra>"
        ))

    apply_theme_layout(
        fig,
        title="<b>Simulated Patient Queue Dynamics Q(t) Over Operating Horizon</b>",
        xaxis_title="Simulation Time (hours elapsed)",
        yaxis_title="Number of Waiting Patients"
    )
    return fig


def plot_cross_scenario_wait_comparison(
    scenario_names: List[str],
    baseline_waits: List[float],
    opt_waits: List[float],
    ci_lowers: Optional[List[float]] = None,
    ci_uppers: Optional[List[float]] = None,
    reductions: Optional[List[float]] = None
) -> go.Figure:
    """
    High-impact grouped bar chart comparing simulated waiting times across demand scenarios
    with 95% Confidence Interval error bars and percentage reduction callouts.
    """
    fig = go.Figure()

    # Short scenario labels for clean presentation
    clean_labels = [s.split(":")[0] for s in scenario_names]

    # Baseline Bars
    fig.add_trace(go.Bar(
        x=clean_labels,
        y=baseline_waits,
        name="Baseline Policy (Manual)",
        marker=dict(color="#DC2626", opacity=0.85),
        text=[f"<b>{w:.1f}m</b>" for w in baseline_waits],
        textposition="auto",
        hovertemplate="<b>%{x} — Baseline</b><br>Avg Wait: %{y:.2f} min<extra></extra>"
    ))

    # Error bar calculation for optimized runs
    error_y_args = None
    if ci_lowers and ci_uppers:
        error_plus = [max(0.0, u - m) for u, m in zip(ci_uppers, opt_waits)]
        error_minus = [max(0.0, m - l) for l, m in zip(ci_lowers, opt_waits)]
        error_y_args = dict(
            type="data",
            symmetric=False,
            array=error_plus,
            arrayminus=error_minus,
            color="#065F46",
            thickness=2,
            width=5
        )

    # Optimized Bars
    fig.add_trace(go.Bar(
        x=clean_labels,
        y=opt_waits,
        name="Optimized Policy (ILP)",
        marker=dict(color="#10B981", opacity=0.9),
        text=[f"<b>{w:.1f}m</b>" for w in opt_waits],
        textposition="auto",
        error_y=error_y_args,
        hovertemplate="<b>%{x} — Optimized</b><br>Avg Wait: %{y:.2f} min<extra></extra>"
    ))

    # Add reduction callout annotations above the bars
    if reductions:
        for idx, (label, b_w, o_w, red) in enumerate(zip(clean_labels, baseline_waits, opt_waits, reductions)):
            y_pos = max(b_w, o_w) * 1.12 + 1.0
            badge_text = f"<b>{red:+.1f}%</b>" if red != 0 else "<b>0.0% (Optimal)</b>"
            badge_color = "#047857" if red > 0 else "#64748B"
            fig.add_annotation(
                x=label,
                y=y_pos,
                text=badge_text,
                showarrow=False,
                font=dict(color=badge_color, size=12),
                bgcolor="#ECFDF5" if red > 0 else "#F1F5F9",
                bordercolor=badge_color,
                borderwidth=1,
                borderpad=4
            )

    apply_theme_layout(
        fig,
        barmode="group",
        title="<b>Patient Waiting Time Reduction Across Demand Tiers (SimPy Discrete-Event Validation)</b>",
        xaxis_title="Operational Scenario",
        yaxis_title="Average Patient Waiting Time (minutes)"
    )
    return fig


def plot_cross_scenario_cost_tradeoff(bench_data: List[Any]) -> go.Figure:
    """
    Connected trajectory visualization illustrating the Operations Research intervention:
    Shows how moving from Baseline to Optimized shifts expenditure and cuts waiting time.
    """
    fig = go.Figure()

    palette = ["#3B82F6", "#8B5CF6", "#F59E0B"]

    for idx, (exp_res, _) in enumerate(bench_data):
        sc_name = exp_res["scenario_name"].split(":")[0]
        color = palette[idx % len(palette)]

        b_cost = exp_res["baseline_analytical"].total_cost
        b_wait = exp_res["baseline_sim_metrics"].mean_wait_minutes
        b_docs = sum(exp_res["baseline_allocation"].values())

        opt_cost = exp_res["opt_ilp_result"].total_cost if exp_res["opt_ilp_result"].is_feasible else b_cost
        opt_wait = exp_res["opt_sim_metrics"].mean_wait_minutes if exp_res["opt_sim_metrics"] else b_wait
        opt_docs = sum(exp_res["opt_ilp_result"].allocation.values()) if exp_res["opt_ilp_result"].is_feasible else b_docs

        # Draw connecting intervention vector (dashed line)
        fig.add_trace(go.Scatter(
            x=[b_cost, opt_cost],
            y=[b_wait, opt_wait],
            mode="lines",
            line=dict(color=color, width=2.5, dash="dash"),
            name=f"{sc_name} Path",
            showlegend=False,
            hoverinfo="skip"
        ))

        # Baseline Marker (Red square)
        fig.add_trace(go.Scatter(
            x=[b_cost],
            y=[b_wait],
            mode="markers+text",
            marker=dict(size=14, color="#DC2626", symbol="square"),
            name="Baseline Policy" if idx == 0 else None,
            showlegend=(idx == 0),
            text=[f"{sc_name} Base ({b_docs} docs)"],
            textposition="top right",
            hovertemplate=f"<b>{sc_name} — Baseline</b><br>Cost: ${b_cost:,.0f}<br>Wait: {b_wait:.2f} min<br>Doctors: {b_docs}<extra></extra>"
        ))

        # Optimized Marker (Green diamond)
        fig.add_trace(go.Scatter(
            x=[opt_cost],
            y=[opt_wait],
            mode="markers+text",
            marker=dict(size=15, color="#10B981", symbol="diamond"),
            name="Optimized Policy" if idx == 0 else None,
            showlegend=(idx == 0),
            text=[f"{sc_name} Opt ({opt_docs} docs)"],
            textposition="bottom right",
            hovertemplate=f"<b>{sc_name} — Optimized (ILP)</b><br>Cost: ${opt_cost:,.0f}<br>Wait: {opt_wait:.2f} min<br>Doctors: {opt_docs}<extra></extra>"
        ))

    apply_theme_layout(
        fig,
        title="<b>Operations Research Intervention Space: Staffing Cost vs. Simulated Wait Time</b>",
        xaxis_title="Total Daily Staffing Expenditure ($)",
        yaxis_title="Average Patient Waiting Time (minutes)"
    )
    return fig

