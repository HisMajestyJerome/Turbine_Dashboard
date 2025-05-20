import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.colors
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(layout="wide")

# App title
st.title("Turbine Data Visualization")

# --- Utility Functions ---

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

def create_all_turbines_figure(df, selected_metric):
    fig = go.Figure()
    turbines = df["Power Plant"].unique()
    columns_to_scale = ["Power Avg [kW]", "Wind Speed Avg [m/s]", "Blade Angle 1 [°]", "Generator Speed Avg [min-1]"]
    colors = ["red", "blue", "green", "orange", "purple", "yellow", "cyan", "magenta"]
    color_map = {turbine: colors[i % len(colors)] for i, turbine in enumerate(turbines)}
    timestamps = pd.date_range(start=df["Timestamp"].min(), end=df["Timestamp"].max(), freq="10T")
    offset = 0

    for i, turbine in enumerate(turbines):
        df_turbine = df[df["Power Plant"] == turbine].copy()
        if df_turbine.empty:
            continue

        scalers = {col: MinMaxScaler() for col in columns_to_scale}
        for col in columns_to_scale:
            if col == "Blade Angle 1 [°]":
                df_turbine[col] = df_turbine[col].apply(lambda x: abs(x) - 360 if abs(x) > 100 else abs(x))
            df_turbine[f"{col}_Scaled"] = scalers[col].fit_transform(df_turbine[[col]])

        df_resampled = df_turbine.set_index("Timestamp").reindex(timestamps).reset_index()
        df_resampled.rename(columns={"index": "Timestamp"}, inplace=True)

        if selected_metric in columns_to_scale:
            col = selected_metric
            units = {
                "Power Avg [kW]":"kW",
                "Wind Speed Avg [m/s]":"m/s",
                "Blade Angle 1 [°]":"°",
                "Generator Speed Avg [min-1]":"min-1"
            }
            values_true = df_resampled[col].values
            fig.add_trace(go.Scatter(
                x=df_resampled["Timestamp"],
                y=df_resampled[f"{col}_Scaled"] + offset,
                mode="lines",
                name=f"{turbine}",
                line=dict(color=color_map[turbine]),
                hovertemplate=f"{turbine}<br>{col}: %{{customdata}} {units[selected_metric]}",
                customdata=values_true,
                connectgaps=False
            ))

        offset += 1

    fig.update_layout(
        title=f"All Turbines - {selected_metric}",
        xaxis_title="Timestamp",
        yaxis_title="Scaled Values",
        font=dict(color="white"),
        hovermode="x unified"
    )
    return fig

def create_single_turbine_figure(df, selected_turbine):
    fig = go.Figure()
    df_plant = df[df["Power Plant"] == selected_turbine].copy()
    if df_plant.empty:
        return fig

    columns_to_scale = ["Power Avg [kW]", "Wind Speed Avg [m/s]", "Generator Speed Avg [min-1]", "Blade Angle 1 [°]"]
    scalers = {col: MinMaxScaler() for col in columns_to_scale}
    for col in columns_to_scale:
        if col == "Blade Angle 1 [°]":
            df_plant["Blade Angle 1 [°]_Scaled"] = df_plant["Blade Angle 1 [°]"].apply(
                lambda x: abs(360 - x) if x > 300 else abs(x)
            ) / 90
        else:
            df_plant[f"{col}_Scaled"] = scalers[col].fit_transform(df_plant[[col]])

    timestamps = pd.date_range(start=df_plant["Timestamp"].min(), end=df_plant["Timestamp"].max(), freq="10T")
    df_resampled = df_plant.set_index("Timestamp").reindex(timestamps).reset_index()
    df_resampled.rename(columns={"index": "Timestamp"}, inplace=True)

    # Define your manual colors here (same order as traces)
    colors = {
        "Wind Speed Avg [m/s]": "blue",
        "Power Avg [kW]": "red",
        "Blade Angle 1 [°]": "green",
        "Generator Speed Avg [min-1]": "purple"
    }

    traces = [
        ("Wind Speed Avg [m/s]", df_resampled["Wind Speed Avg [m/s]"], df_resampled["Wind Speed Avg [m/s]_Scaled"], "Wind Speed Avg: %{customdata} m/s"),
        ("Power Avg [kW]", df_resampled["Power Avg [kW]"], df_resampled["Power Avg [kW]_Scaled"], "Power Avg: %{customdata} kW"),
        ("Blade Angle 1 [°]", df_resampled["Blade Angle 1 [°]"], df_resampled["Blade Angle 1 [°]_Scaled"], "Blade Angle: %{customdata}°"),
        ("Generator Speed Avg [min-1]", df_resampled["Generator Speed Avg [min-1]"], df_resampled["Generator Speed Avg [min-1]_Scaled"], "Generator Speed: %{customdata} 1/min")
    ]

    for name, original, scaled, hover_text in traces:
        fig.add_trace(go.Scatter(
            x=df_resampled["Timestamp"],
            y=scaled,
            mode="lines",
            name=f"{name}",
            hovertemplate=f"{hover_text}<extra></extra>",
            customdata=original,
            connectgaps=False,
            line=dict(color=colors.get(name, "black"))  # use color from dict or black as default
        ))

    fig.update_layout(
        title=f"Parameters - {selected_turbine}",
        xaxis_title="Timestamp",
        yaxis=dict(title="Scaled Values (Relative Comparison)", showgrid=True),
        hovermode="x unified"
    )
    return fig


# --- Upload Section ---
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, sep=';', skiprows=1, parse_dates=["Timestamp"])
        df.columns = df.columns.str.strip()
        df.columns = df.columns.str.replace('\u202f', '', regex=True)
        st.success("File loaded successfully.")
    except Exception as e:
        st.error(f"Failed to load file: {e}")
        st.stop()

    turbines = df["Power Plant"].unique()
    selected_turbine = st.selectbox("Select a turbine", turbines)

    # Metric selector for all turbines plot
    selected_metric = st.selectbox(
        "Select metric to visualize for all turbines",
        ["Power Avg [kW]", "Wind Speed Avg [m/s]", "Blade Angle 1 [°]", "Generator Speed Avg [min-1]"]
    )

    st.subheader(f"Single Turbine: {selected_turbine}")
    st.plotly_chart(create_single_turbine_figure(df, selected_turbine), use_container_width=True)

    st.subheader("All Turbines Overview")
    st.plotly_chart(create_all_turbines_figure(df, selected_metric), use_container_width=True)

else:
    st.info("Please upload a CSV file to begin.")
