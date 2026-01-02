import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# --------------------------------------
# Page setup
# --------------------------------------
st.set_page_config(page_title="Air Quality Prediction Dashborad", layout="centered")

st.title("🌍 Air Quality Prediction Dashboard")
st.caption("AI-based air quality monitoring and trend visualization")

color_map = {
    "Good": "#2ECC71",
    "Moderate": "#F1C40F",
    "Poor": "#E67E22",
    "Hazardous": "#E74C3C"
}

advice_map = {
    "Good": "Air quality is satisfactory. Enjoy outdoor activities.",
    "Moderate": "Sensitive individuals should reduce prolonged exertion.",
    "Poor": "Limit ourdoor activity. Consider wearing a mask.",
    "Hazardous": "Stay indoors. Avoid outdoor exposure."
}

# --------------------------------------
# Load data
# --------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("prediction_history.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df
df = load_data()

# --------------------------------------
# Mapping for visualization
# --------------------------------------
aqi_map = {
    "Good": 1,
    "Moderate": 2,
    "Poor": 3,
    "Hazardous": 4
}

df["aqi_level"]= df["Prediction"].map(aqi_map)

# --------------------------------------
# Latest status
# --------------------------------------

st.subheader("📌 Current Air Quality Status")

latest = df.sort_values("Date").iloc[-1]

status_color = color_map[latest["Prediction"]]
with st.container():
    col1, col2 = st.columns([2, 1])

    col1.metric("Air Quality", latest['Prediction'])
    col2.metric("Confidence", f"{latest['Confidence']*100:.1f}%")

st.markdown(
    f"""
    <div style="
        padding:20px;
        border-radius:15px;
        background-color:{status_color};
        color:white;
        font-size:24px;
        text-align:center;
    ">
        <strong>{latest['Prediction']}</strong><br>
        Reliability: {latest['Confidence']*100:.2f}%
    </div>
    """,

    unsafe_allow_html=True
)

st.info(advice_map[latest["Prediction"]])

selected_days = st.slider(
    "Show predictions for last (days)",
    min_value=1,
    max_value=30,
    value=7
)
# --------------------------------------
# Air Quality Forecast
# --------------------------------------

st.subheader("Air Quality Forcast(Next 5 Days)")
aqi_map = {
    "Good": 1,
    "Moderate": 2,
    "Poor": 3,
    "Hazardous": 4
}

df["aqi_level"] = df["Prediction"].map(aqi_map)

daily_avg = df.groupby("Date")["aqi_level"].mean().reset_index()

last_value = daily_avg["aqi_level"].iloc[-1]
trend = daily_avg["aqi_level"].diff().mean()

future_days = 5
future_dates = pd.date_range(
    start=daily_avg["Date"].iloc[-1] + pd.Timedelta(days=1),
    periods=future_days
)

future_levels = [
    max(1, min(4, round(last_value + trend * (i+1))))
    for i in range (future_days)
]

forecast_df = pd.DataFrame({
    "Date": future_dates,
    "aqi_level": future_levels
})

# --------------------------------------
# Daily trend
# --------------------------------------
st.button("🔄 Refresh Data")

st.subheader("📈 Air Quality Trend Over Time")
trend_fig = px.line(
    df,
    x="Date",
    y="aqi_level",
    markers=True,
    labels={
        "data": "Date",
        "aqi_level": "Air Quality Level"
    }
)

trend_fig.update_yaxes(
    tickvals=[1, 2, 3, 4],
    ticktext=["Good", "Moderate", "Poor", "Hazardous"]
)

st.plotly_chart(trend_fig, use_container_width=True)

# --------------------------------------
# Distribution
# --------------------------------------
st.subheader("📊 Air Quality Distribution")

dist_df = df["Prediction"].value_counts().reset_index()
dist_df.columns= ["Air_Quality", "Count"]

dist_fig = px.bar(
    dist_df,
    x="Air_Quality",
    y="Count",
    title="Distribution of Air Quality Prediction"
)

st.plotly_chart(dist_fig, use_container_width=True)

# --------------------------------------
# Historical records
# --------------------------------------
with st.expander("View Prediction History"):

    st.dataframe(df.tail(30))
