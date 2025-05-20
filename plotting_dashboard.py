import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.colors
from datetime import timedelta

def scale_blade_angle(x):
    if x > 300:
        return abs(360 - x) / 90
    elif abs(x) > 100:
        return (abs(x) - 360) / 90
    else:
        return abs(x) / 90

def find_gaps(timestamps, freq='10T'):
    gaps = []
    expected_delta = pd.Timedelta(freq)
    for i in range(len(timestamps) - 1):
        delta = timestamps[i + 1] - timestamps[i]
        if delta > expected_delta:
            gaps.append((timestamps[i], timestamps[i + 1]))
    return gaps

def create_all_turbines_figure(df, dark_mode):
    turbines = df["Power Plant"].unique()
    colors = plotly.colors.qualitative.Plotly
    color_map = {turbine: colors[i % len(colors)] for i, turbine in enumerate(turbines)}

    timestamps = pd.date_range(start=df["Timestamp"].min(), end=df["Timestamp"].max(), freq="10T")
    df_resampled = df.set_index("Timestamp").groupby("Power Plant").apply(lambda g: g.reindex(timestamps)).reset_index(level=0)
    df_resampled = df_resampled.rename(columns={"level_1": "Timestamp"})
    df_resampled["Timestamp"] = pd.to_datetime(df_resampled["Timestamp"])

    fig = go.Figure()

    for turbine in turbines:
        df_turbine = df_resampled[df_resampled["Power Plant"] == turbine]

        fig.add_trace(go.Scatter(
            x=df_turbine["Timestamp"],
            y=df_turbine["Power Avg [kW]"],
            mode='lines',
            name=f"{turbine} Power Avg [kW]",
            line=dict(color=color_map[turbine])
        ))

        fig.add_trace(go.Scatter(
            x=df_turbine["Timestamp"],
            y=df_turbine["Wind Speed Avg [m/s]"],
            mode='lines',
            name=f"{turbine} Wind Speed Avg [m/s]",
            line=dict(color=color_map[turbine], dash='dot')
        ))

        scaled_blade_angle = df_turbine["Blade Angle 1 [°]"].apply(scale_blade_angle)
        fig.add_trace(go.Scatter(
            x=df_turbine["Timestamp"],
            y=scaled_blade_angle,
            mode='lines',
            name=f"{turbine} Blade Angle 1 (scaled)",
            line=dict(color=color_map[turbine], dash='dash')
        ))

        fig.add_trace(go.Scatter(
            x=df_turbine["Timestamp"],
            y=df_turbine["Generator Speed Avg [min-1]"],
            mode='lines',
            name=f"{turbine} Generator Speed Avg [min-1]",
            line=dict(color=color_map[turbine], dash='dashdot')
        ))

    gap_times = find_gaps(timestamps)
    shapes = []
    for start, end in gap_times:
        shapes.append({
            'type': 'rect',
            'xref': 'x',
            'yref': 'paper',
            'x0': start,
            'x1': end,
            'y0': 0,
            'y1': 1,
            'fillcolor': 'rgba(200, 200, 200, 0.3)',
            'line': {'width': 0},
            'layer': 'below',
        })

    template = 'plotly_dark' if dark_mode else 'plotly_white'

    fig.update_layout(
        title="All Turbines Data",
        xaxis_title="Timestamp",
        yaxis_title="Values",
        legend_title="Metrics",
        shapes=shapes,
        margin=dict(t=50, b=50, l=50, r=50),
        hovermode='x unified',
        template=template
    )
    return fig

def create_single_turbine_figure(df, selected_turbine, dark_mode):
    if selected_turbine not in df["Power Plant"].unique():
        return go.Figure()

    df_turbine = df[df["Power Plant"] == selected_turbine]

    timestamps = pd.date_range(start=df_turbine["Timestamp"].min(), end=df_turbine["Timestamp"].max(), freq="10T")
    df_resampled = df_turbine.set_index("Timestamp").reindex(timestamps).reset_index()
    df_resampled = df_resampled.rename(columns={"index": "Timestamp"})

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_resampled["Timestamp"],
        y=df_resampled["Power Avg [kW]"],
        mode='lines',
        name="Power Avg [kW]",
        line=dict(color='blue')
    ))

    fig.add_trace(go.Scatter(
        x=df_resampled["Timestamp"],
        y=df_resampled["Wind Speed Avg [m/s]"],
        mode='lines',
        name="Wind Speed Avg [m/s]",
        line=dict(color='orange', dash='dot')
    ))

    scaled_blade_angle = df_resampled["Blade Angle 1 [°]"].apply(scale_blade_angle)
    fig.add_trace(go.Scatter(
        x=df_resampled["Timestamp"],
        y=scaled_blade_angle,
        mode='lines',
        name="Blade Angle 1 (scaled)",
        line=dict(color='green', dash='dash')
    ))

    fig.add_trace(go.Scatter(
        x=df_resampled["Timestamp"],
        y=df_resampled["Generator Speed Avg [min-1]"],
        mode='lines',
        name="Generator Speed Avg [min-1]",
        line=dict(color='red', dash='dashdot')
    ))

    gap_times = find_gaps(timestamps)
    shapes = []
    for start, end in gap_times:
        shapes.append({
            'type': 'rect',
            'xref': 'x',
            'yref': 'paper',
            'x0': start,
            'x1': end,
            'y0': 0,
            'y1': 1,
            'fillcolor': 'rgba(200, 200, 200, 0.3)',
            'line': {'width': 0},
            'layer': 'below',
        })

    template = 'plotly_dark' if dark_mode else 'plotly_white'

    fig.update_layout(
        title=f"Data for Turbine: {selected_turbine}",
        xaxis_title="Timestamp",
        yaxis_title="Values",
        shapes=shapes,
        margin=dict(t=50, b=50, l=50, r=50),
        hovermode='x unified',
        template=template
    )

    return fig

st.title("Turbine Data Visualization")

uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

dark_mode = st.checkbox("Enable Dark Mode 🌙")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, sep=';', skiprows=1, parse_dates=["Timestamp"])
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()

    turbines = df["Power Plant"].unique()
    selected_turbine = st.selectbox("Select turbine", turbines)

    st.plotly_chart(create_all_turbines_figure(df, dark_mode), use_container_width=True)
    st.plotly_chart(create_single_turbine_figure(df, selected_turbine, dark_mode), use_container_width=True)
else:
    st.info("Please upload a CSV file to get started.")
