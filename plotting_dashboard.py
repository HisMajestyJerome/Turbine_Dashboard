import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.colors
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import math
from datetime import datetime
from datetime import timedelta


credentials = st.secrets["users"]
# Initialize session state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = ""

# Authentication logic
def login():
    username = st.session_state["login_user"]
    password = st.session_state["login_pass"]
    if username in credentials and credentials[username] == password:
        st.session_state.authenticated = True
        st.session_state.username = username
    else:
        st.error("Invalid username or password")

# Show login fields only if not logged in
if not st.session_state.authenticated:
    st.text_input("Username", key="login_user")
    st.text_input("Password", type="password", key="login_pass")
    st.button("Login", on_click=login)
    st.stop()
# App title
st.set_page_config(layout="wide")
st.title("Turbine Data Visualization")



#constant color for single turbine plot
default_colors = {
        "Wind Speed Avg [m/s]": "blue",
        "Power Avg [kW]": "red",
        "Blade Angle 1 [°]": "green",
        "Generator Speed Avg [min-1]": "purple"
    }


# -------------------------------------------------------------------------- Utility Functions -----------------------------------------------------

def wind_cutoff_calculator(df, selected_turbine):
    df = df[df["Power Plant"] == selected_turbine].copy()
    df['Generator Speed Avg [min-1]'] = df['Generator Speed Avg [min-1]'].astype(int)
    df = df[df['Generator Speed Avg [min-1]'] > 300]

    x = df['Wind Speed Avg [m/s]']
    y = df['Power Avg [kW]']
    coeffs = np.polyfit(x, y, deg=2)
    fit_fn = np.poly1d(coeffs)

    def solve_x(coeffs):
        A, B, C = coeffs
        delta = B**2 - 4*A*C
        if delta < 0:
            return []
        elif delta == 0:
            return [-B / (2*A)]
        else:
            return [(-B - math.sqrt(delta)) / (2*A), (-B + math.sqrt(delta)) / (2*A)]

    zeroes = solve_x(coeffs)

    # Plot 1: Wind vs Power with polynomial fit
    x_line = np.linspace(min(x), max(x), 100)
    trace_data = go.Scatter(x=x, y=y, mode='markers', name='Data Points')
    trace_fit = go.Scatter(x=x_line, y=fit_fn(x_line), mode='lines', name='Quadratic Fit', line=dict(color='red'))

    traces = [trace_data, trace_fit]
    if zeroes:
        traces.append(go.Scatter(
            x=[max(zeroes), max(zeroes)],
            y=[min(y), max(y)],
            mode='lines',
            name=f'Zero Point = {max(zeroes):.2f}',
            line=dict(color='green', dash='dash')
        ))

    fig1 = go.Figure(data=traces)
    fig1.update_layout(
        title='Polynomial Fit: Wind Speed vs Power',
        xaxis_title='Wind Speed [m/s]',
        yaxis_title='Power [kW]',
        template='plotly_white'
    )

    # Scale and resample
    df_original = df.copy()
    columns_to_scale = ["Power Avg [kW]", "Wind Speed Avg [m/s]", "Blade Angle 1 [°]", "Generator Speed Avg [min-1]"]
    scalers = {col: MinMaxScaler() for col in columns_to_scale}

    for col in columns_to_scale:
        if col == "Blade Angle 1 [°]":
            df_original[col] = df_original[col].apply(lambda x: abs(360 - x) if x > 300 else x)
            df_original[f"{col}_Scaled"] = df_original[col] / 90
        else:
            df_original[f"{col}_Scaled"] = scalers[col].fit_transform(df_original[[col]])

    new_timestamps = pd.date_range(start=df_original["Timestamp"].min(), end=df_original["Timestamp"].max(), freq="10T")
    df_resampled = df_original.set_index("Timestamp").reindex(new_timestamps).reset_index()
    df_resampled.rename(columns={"index": "Timestamp"}, inplace=True)

    time = df_resampled["Timestamp"]
    wind = df_resampled["Wind Speed Avg [m/s]_Scaled"]
    power = df_resampled["Power Avg [kW]_Scaled"]

    zero_point_scaled = scalers["Wind Speed Avg [m/s]"].transform(
    pd.DataFrame([[max(zeroes)]], columns=["Wind Speed Avg [m/s]"])
    )[0][0]


    power_off = [p if w < zero_point_scaled else None for w, p in zip(wind, power)]

    # Plot 2: Scaled time series
    fig2 = go.Figure()
    
    #fig2.add_trace(go.Scatter(x=time, y=power, mode='lines', name='Power [kW] (Scaled)', line=dict(color='pink')))
    #fig2.add_trace(go.Scatter(x=time, y=wind, mode='lines', name='Wind Speed [m/s] (Scaled)', line=dict(color='blue')))
    #fig2.add_trace(go.Scatter(x=time, y=power_off, mode='lines', name='Power Off', line=dict(color='red')))
    
    colors = {
        "Wind Speed Avg [m/s]": "blue",
        "Power Avg [kW]": "yellow",
        "Power Off": "red"
    }
    traces = [
        ("Wind Speed Avg [m/s]", wind , "Wind Speed Avg: %{customdata} m/s"),
        ("Power Avg [kW]", power, "Power Avg: %{customdata} kW"),
        ("Power Off", power_off, "Power off due to wind")
    ]
    
    
    for name, scaled, hover_text in traces:
        fig2.add_trace(go.Scatter(
            x=df_resampled["Timestamp"],
            y=scaled,
            mode="lines",
            name=f"{name}",
            hovertemplate=f"{hover_text}<extra></extra>",
            customdata=scaled,
            connectgaps=False,
            line=dict(color=colors.get(name, "grey"))  # use color from dict or grey as default
        ))
    fig2.add_shape(
        type="line",
        x0=df_resampled["Timestamp"].min(),
        x1=df_resampled["Timestamp"].max(),
        y0=zero_point_scaled,
        y1=zero_point_scaled,
        line=dict(color="green", width=2, dash="dash"),
    )
    fig2.update_layout(
        title='Scaled Time Series: Power & Wind Speed',
        xaxis_title='Timestamp',
        yaxis_title='Scaled Values',
        template='plotly_white',
        hovermode='x unified'  # 👈 this is the key
    )

    return fig1, fig2

# ================================================================================NEW EDIT ==========================================================
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

def create_single_turbine_figure(df, selected_turbine, colors=None):
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

    # Default colors if none provided
    
    if colors is None:
        colors = default_colors

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

def _set_last_edited_to_manual(): #sets updated for table
    st.session_state.last_edited = "manual"

def _set_last_edited_to_slider():
    st.session_state.last_edited = "slider"
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
    
    if "turbine_index" not in st.session_state:
        st.session_state.turbine_index = 0

    col1,col2 = st.columns([4,1])
    with col1:
        selected_turbine = st.selectbox("Select a turbine", turbines, index=st.session_state.turbine_index)
    with col2:
        with st.expander("Color Customizer", expanded=False):
                # Color picker widget
                col1, col2 = st.columns([2,1])
                with col1:
                    default_colors["Wind Speed Avg [m/s]"] = st.color_picker("Wind Speed", "#5542fa") # wind
                    default_colors["Power Avg [kW]"] = st.color_picker("Power Avg", "#ff4040") # wind
                with col2:
                    default_colors["Blade Angle 1 [°]"] = st.color_picker("Blade Angle", "#58aa48") # wind
                    default_colors["Generator Speed Avg [min-1]"] = st.color_picker("Generator Speed Avg", "#a139aa") # wind
                    
    col1,col2,col3 = st.columns([1,1,12])
    with col1:
        if st.button("Previous"):
            st.session_state.turbine_index = (st.session_state.turbine_index - 1) % len(turbines)
            st.rerun()  # Force rerun to update the selectbox
    with col2:
        if st.button("Next"):
            st.session_state.turbine_index = (st.session_state.turbine_index + 1) % len(turbines)
            st.rerun()
    # Add this below your turbine selection and before plotting:
    show_wind = st.checkbox("Run Wind Cutoff Calculator")
    

    if show_wind:
        fig1, fig2 = wind_cutoff_calculator(df, selected_turbine)
        st.plotly_chart(fig1, use_container_width=True)
        st.plotly_chart(fig2, use_container_width=True)

    if not show_wind:
        col1,col2 = st.columns([2,1])
        with col1:
            show_table = st.checkbox("Show as Table")
        with col2:
            pass
                

        if show_table:
            # initialize if missing
            if "last_edited" not in st.session_state:
                st.session_state.last_edited = None

            df_single = df[df["Power Plant"] == selected_turbine].copy()
            df_single["Timestamp"] = pd.to_datetime(df_single["Timestamp"])
            min_time = df_single["Timestamp"].min().to_pydatetime()
            max_time = df_single["Timestamp"].max().to_pydatetime()

            st.write(f"### Table for {selected_turbine}")
            col1, col2, col3 = st.columns([1,5,1])

            with col1:
                # manual start date/time
                start_date = st.date_input(
                    "Start Date",
                    min_value=min_time.date(),
                    max_value=max_time.date(),
                    value=min_time.date(),
                    key="man_start_date",
                    on_change=_set_last_edited_to_manual
                )
                start_clock = st.time_input(
                    "Start Time",
                    value=min_time.time(),
                    key="man_start_time",
                    on_change=_set_last_edited_to_manual
                )
                manual_start = datetime.combine(start_date, start_clock)

            with col2:
                # slider
                slider_start, slider_end = st.slider(
                    "Or use the slider",
                    value=(min_time, max_time),
                    min_value=min_time,
                    max_value=max_time,
                    format="YYYY-MM-DD HH:mm:ss",
                    step=timedelta(minutes=10),
                    key="time_slider",
                    on_change=_set_last_edited_to_slider
                )

            with col3:
                # manual end date/time
                end_date = st.date_input(
                    "End Date",
                    min_value=min_time.date(),
                    max_value=max_time.date(),
                    value=max_time.date(),
                    key="man_end_date",
                    on_change=_set_last_edited_to_manual
                )
                end_clock = st.time_input(
                    "End Time",
                    value=max_time.time(),
                    key="man_end_time",
                    on_change=_set_last_edited_to_manual
                )
                manual_end = datetime.combine(end_date, end_clock)

            # choose which to use based on last interaction
            if st.session_state.last_edited == "manual":
                start_time, end_time = manual_start, manual_end
            else:
                start_time, end_time = slider_start, slider_end

            st.write(f"Showing data from **{start_time}** to **{end_time}**")

            df_filtered = df_single[
                (df_single["Timestamp"] >= start_time) &
                (df_single["Timestamp"] <= end_time)
            ]
            st.dataframe(df_filtered, height=400)


        else:
            st.subheader(f"Single Turbine: {selected_turbine}")
            st.plotly_chart(create_single_turbine_figure(df, selected_turbine, colors=default_colors), use_container_width=True)
            # All turbines plot
            selected_metric = st.selectbox(
                "Select metric to visualize for all turbines",
                ["Power Avg [kW]", "Wind Speed Avg [m/s]", "Blade Angle 1 [°]", "Generator Speed Avg [min-1]"]
            )
            st.subheader("All Turbines Overview")
            st.plotly_chart(create_all_turbines_figure(df, selected_metric), use_container_width=True)

else:
    st.info("Please upload a CSV file to begin.")
