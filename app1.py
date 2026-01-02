# --------------------------------------
# AI-BASED AIR QUALITY MONITORING DASHBOARD (AUTOMATIC)
# --------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import joblib 
import random
import os
from datetime import datetime
import plotly.express as px
import time
from sklearn.linear_model import LogisticRegression

# --------------------------------------
# LOAD TRAINED MODELS
# --------------------------------------
rf_model = joblib.load("rf_model.pkl")
lr_model = joblib.load("lr_model.pkl")
scaler = joblib.load("scaler.pkl")

reverse_mapping = {
    0: "Good",
    1: "Moderate",
    2: "Poor",
    3: "Hazardous"
}

st.markdown("""
<style>
/* Main container width */
.block-container {
    padding-top: 2rem;
    padding-left: 3rem;
    padding-right: 3rem;
}

/* Make cards responsive */
@media(max-width: 768px) {
    .block-container {
        padding-left: 1.5rem;
        padding-right: 1.5rem;
    }
}

/* Center big status card text */
.status-card {
    text-align: center;
}

/* Improve font scaling */
h1, h2, h3 {
    word-break: break-word;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------
# SIMULATED SENSOR DATA GENERATOR
# --------------------------------------
def generate_simulated_data():
    return [
        random.uniform(20, 38),         # Temperature
        random.uniform(30, 90),         # Humidity
        random.uniform(5, 150),         # PM2.5
        random.uniform(10, 200),        # PM10
        random.uniform(5, 100),         # NO2
        random.uniform(2, 60),          # SO2
        random.uniform(0.1, 5),         # CO
        random.uniform(0, 5),           # Industrial proximity
        random.uniform(300, 3000)       # Population density
    ]

# --------------------------------------
# AI PREDICTION FUNCTION (HYBRID MODEL)
# --------------------------------------
def predict_air_quality():
    simulated = generate_simulated_data()

    df = pd.DataFrame([simulated], columns=[
        "Temperature", "Humidity", "PM2.5", "PM10", "NO2", "SO2", "CO",
        "Proximity_to_Industrial_Areas", "Population_Density"
    ])

    scaled = scaler.transform(df)

    rf_prob = rf_model.predict_proba(scaled)
    lr_prob = lr_model.predict_proba(scaled)

    hybrid_prob = (rf_prob + lr_prob) / 2

    predicted_class = np.argmax(hybrid_prob)
    confidence = np.max(hybrid_prob)* 100

    return reverse_mapping[predicted_class], confidence

# --------------------------------------
# SAVE PREDICTION HISTORY (DATE + TIME)
# --------------------------------------

def save_prediction(label, confidence):
    file = "pred_hist.csv"

    data = pd.DataFrame([{
        "Timestamp": datetime.now(), # Full datetime
        "Prediction": label,
        "Confidence": confidence
    }])

    if os.path.exists(file):
        data.to_csv(file, mode="a", header=False, index=False)
    else:
        data.to_csv(file, index=False)

# --------------------------------------
# AQI COLOR THEMES
# --------------------------------------
def aqi_style(label):
    colors = {
        "Good": ("#2ECC71",
                 "Air quality is satisfactory. Enjoy outdoor activities."),
        "Moderate": ("#F1C40F",
                     "Sensitive individuals should reduce prolonged exertion."),
        "Poor": ("#E67E22", "Limit ourdoor activity. Consider wearing a mask."),
        "Hazardous": ("#E74C3C", "Stay indoors. Avoid outdoor exposure.")
    }
    return colors[label]

# --------------------------------------
# STREAMLIT APP DESIGN
# --------------------------------------
st.set_page_config(page_title="AI Air Quality Dashboard",
                   page_icon="🌍", layout="wide")
left, right = st.columns([3, 1])

with left:
    st.title("🌍 AI-Based Air Quality Monitoring Dashboard")

with right:
    st.markdown(
        f"**Last Updated**  \n{datetime.now().strftime('%b %d, %Y %H:%M')}"
    )
    if "last_update" not in st.session_state:
        st.session_state.last_update = datetime.now()

    if(datetime.now() - st.session_state.last_update).seconds > 60:
        st.session_state.last_update = datetime.now()
        st.rerun()

# Auto prediction
air_quality, Confidence = predict_air_quality()
save_prediction(air_quality, Confidence)

color, message = aqi_style(air_quality)

# ============= DISPLAY MAIN STATUS ===============
st.markdown(
    f"""
    <div style="background:{color};
    padding:10px;
        border-radius:12px;
        color:white;
        text-align:center;">
        <h2 style="font-size:32px;">{air_quality}</h2>
        <p>Reliabilty: <b>{Confidence:.2f}%</b></p>
        <hr style="border:1px solid white;">
        <p>⚠️ {message}</p>
    </div>
    """,
    unsafe_allow_html=True
)

# --------------------------------------
# LOAD AND DISPLAY HISTORY
# --------------------------------------
if os.path.exists("pred_hist.csv"):
    history = pd.read_csv("pred_hist.csv")
    history["Timestamp"] = pd.to_datetime(history["Timestamp"])
    history = history.sort_values("Timestamp")
    
col1, col2 = st.columns(2)
with col1:
    st.subheader("📈 Air Quality Trend")
    # Trend chart
    trend_chart = px.scatter(
        history,
        x="Timestamp",
        y="Confidence",
        color="Prediction",
        title="📈 Air Quality Trend (Confidence Over Time)",
    )
    
    trend_chart.update_traces(mode="lines+markers")
    
    st.plotly_chart(trend_chart, use_container_width=True)
    
with col2:
    st.subheader("📊 AQI Distribution")
    # Distribution chart
    dist_chart = px.histogram(
        history,
        x="Prediction",
        title="📊 Air Quality Distribution",
        color="Prediction",
        color_discrete_map={
            "Good": "#2ECC71",
            "Moderate": "#F1C40F",
            "Poor": "#E67E22", 
            "Hazardous": "#E74C3C"
        }
    )
    st.plotly_chart(dist_chart, use_container_width=True)

    # Recent history table
st.subheader("🕒 Recent Predictions")
st.dataframe(history.tail(10).iloc[::-1], use_container_width=True)


from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import plotly.express as px

st.markdown("## 🔮 Future Air Quality Forecast")
run_forecast = st.button("Run 5-Day Forecast")

if run_forecast:

    # --- Prepare history ---
    history["Timestamp"] = pd.to_datetime(history["Timestamp"])
    history = history.sort_values("Timestamp")

    daily_history = (
        history
        .groupby(history["Timestamp"].dt.date)
        .agg({"Prediction": lambda x: x.mode()[0]})
        .reset_index()
        .rename(columns={"Timestamp": "Date"})
    )

    # --- Encode AQI ---
    aqi_numeric = {
        "Good": 1,
        "Moderate": 2,
        "Poor": 3,
        "Hazardous": 4
    }

    reverse_numeric = {v: k for k, v in aqi_numeric.items()}

    daily_history["aqi_value"] = daily_history["Prediction"].map(aqi_numeric)

    if len(daily_history) < 3:
        st.warning("Not enough historical data to generate forecast.")
    else:
        # --- Train transition model ---
        X = daily_history["aqi_value"].values[:-1].reshape(-1, 1)
        y = daily_history["aqi_value"].values[1:]

        forecast_model = LogisticRegression(
            multi_class="multinomial",
            max_iter=1000
        )
        forecast_model.fit(X, y)

        # --- Generate future predictions ---
        future_days = 5
        last_value = daily_history["aqi_value"].iloc[-1]

        future_values = []
        current_value = last_value

        for _ in range(future_days):
            next_value = forecast_model.predict(
                np.array([[current_value]])
            )[0]

            future_values.append(next_value)
            current_value = next_value

        # --- Create future dates ---
        last_date = pd.to_datetime(daily_history["Date"].iloc[-1])

        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=future_days,
            freq="D"
        )

        forecast_df = pd.DataFrame({
            "Date": future_dates,
            "Forecast_value": future_values
        })

        forecast_df["Forecast"] = forecast_df["Forecast_value"].map(reverse_numeric)

        # --- Layout ---
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📈 5-Day Air Quality Forecast")

            fig = px.line(
                forecast_df,
                x="Date",
                y="Forecast_value",
                markers=True,
                title="Predicted Air Quality Trend"
            )

            fig.update_yaxes(
                tickvals=[1, 2, 3, 4],
                ticktext=["Good", "Moderate", "Poor", "Hazardous"]
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("📅 Forecast Summary")
            st.dataframe(
                forecast_df[["Date", "Forecast"]],
                use_container_width=True
            )

# --------------------------------------
# FOOTER
# --------------------------------------
st.markdown(
    """
    <hr>
    <center>🌱 Powered by AI Monitoring System | Developed for Education Use </center>
    """,
    unsafe_allow_html=True
)
