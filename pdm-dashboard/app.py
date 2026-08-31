import time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Industrial IoT Sensor Monitor", page_icon="⚙️", layout="wide"
)

st.title("⚙️ Real-Time Equipment Health & Anomaly Dashboard")
st.markdown(
    "Simulated live IoT telemetry stream for predictive maintenance monitoring."
)

# Sidebar Controls
st.sidebar.header("Simulation Settings")
refresh_rate = st.sidebar.slider(
    "Sensor Streaming Speed (seconds)", 0.5, 3.0, 1.0
)
temp_threshold = st.sidebar.number_input(
    "Air Temp Warning Threshold (°K)", value=302.0
)
torque_threshold = st.sidebar.number_input(
    "Torque Warning Threshold (Nm)", value=60.0
)

# Initialize Session State
if "step" not in st.session_state:
  st.session_state.step = 0

if "df" not in st.session_state:
  st.session_state.df = pd.DataFrame(
      columns=[
          "Timestamp",
          "Air Temp (°K)",
          "Process Temp (°K)",
          "Speed (RPM)",
          "Torque (Nm)",
          "Tool Wear (min)",
          "Health Score",
      ]
  )


# Generate synthetic sensor telemetry
def get_latest_telemetry(step):
  np.random.seed(step)
  air_temp = 298.1 + (step * 0.05) + np.random.normal(0, 0.5)
  process_temp = 308.1 + (step * 0.04) + np.random.normal(0, 0.4)
  rotational_speed = 1500 + np.random.normal(0, 50) - (step * 2)
  torque = 40 + (step * 0.3) + np.random.normal(0, 2)
  tool_wear = step * 1.5

  # Health score calculation
  health_score = max(0, 100 - (step * 0.8) - np.random.normal(0, 1))

  return {
      "Timestamp": pd.Timestamp.now().strftime("%H:%M:%S"),
      "Air Temp (°K)": round(air_temp, 2),
      "Process Temp (°K)": round(process_temp, 2),
      "Speed (RPM)": int(rotational_speed),
      "Torque (Nm)": round(torque, 2),
      "Tool Wear (min)": round(tool_wear, 1),
      "Health Score": round(health_score, 1),
  }


# Append new stream reading
new_data = get_latest_telemetry(st.session_state.step)
st.session_state.df = pd.concat(
    [st.session_state.df, pd.DataFrame([new_data])]
).tail(20)
st.session_state.step += 1

# KPI Header Metrics
col1, col2, col3, col4 = st.columns(4)
latest = st.session_state.df.iloc[-1]

if (
    latest["Health Score"] < 50
    or latest["Air Temp (°K)"] > temp_threshold
    or latest["Torque (Nm)"] > torque_threshold
):
  status_color = "🔴 FAILURE IMMINENT"
  st.error(
      f"ALERT: Sensor anomaly detected! Air Temp: {latest['Air Temp (°K)']}°K,"
      f" Torque: {latest['Torque (Nm)']}Nm"
  )
elif latest["Health Score"] < 75:
  status_color = "🟡 WARNING"
  st.warning("Maintenance Warning: Equipment degradation observed.")
else:
  status_color = "🟢 NORMAL"

col1.metric("Equipment Status", status_color)
col2.metric("Health Score", f"{latest['Health Score']}%")
col3.metric("Air Temperature", f"{latest['Air Temp (°K)']} °K")
col4.metric("Torque", f"{latest['Torque (Nm)']} Nm")

st.divider()

# Charts
c1, c2 = st.columns(2)

with c1:
  st.subheader("Temperature & Torque Telemetry")
  fig1 = px.line(
      st.session_state.df,
      x="Timestamp",
      y=["Air Temp (°K)", "Torque (Nm)"],
      markers=True,
  )
  st.plotly_chart(fig1, use_container_width=True)

with c2:
  st.subheader("Equipment Health Gauge")
  fig2 = go.Figure(
      go.Indicator(
          mode="gauge+number",
          value=latest["Health Score"],
          title={"text": "Health Index (%)"},
          gauge={
              "axis": {"range": [0, 100]},
              "bar": {"color": "darkblue"},
              "steps": [
                  {"range": [0, 50], "color": "red"},
                  {"range": [50, 75], "color": "yellow"},
                  {"range": [75, 100], "color": "green"},
              ],
          },
      )
  )
  st.plotly_chart(fig2, use_container_width=True)

# Stream auto-refresh
time.sleep(refresh_rate)
st.rerun()