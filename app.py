import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Aadhaar-Setu",
    layout="wide"
)

# ---------------- TITLE ----------------
st.title("🇮🇳 Aadhaar-Setu: Predictive Infrastructure Planner")
st.subheader("Biometric Update Trend Analysis, Anomaly Detection & Forecasting")

st.markdown("""
Aadhaar-Setu analyzes biometric update data to:
- Detect **service gaps**
- Visualize **district-level demand**
- Predict **future biometric update surges**
- Help UIDAI plan infrastructure proactively
""")

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload Aadhaar Biometric Dataset (CSV)",
    type="csv"
)

if uploaded_file:

    # ---------------- LOAD DATA ----------------
    df = pd.read_csv(uploaded_file)
    st.success("Dataset loaded successfully!")

    # ---------------- DATA PREVIEW ----------------
    st.subheader("📄 Raw Data Preview")
    st.dataframe(df.head())

    # ---------------- BASIC STATISTICS ----------------
    st.subheader("📊 Basic Statistics")
    st.dataframe(df.describe())

    # ---------------- FEATURE ENGINEERING ----------------
    st.subheader("⚙️ Feature Engineering")

    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month

    st.write("Created features: **day_of_week**, **month**")

    # ---------------- ANOMALY DETECTION ----------------
    st.subheader("🚨 Anomaly Detection: Service Gaps")

    district_summary = (
        df.groupby(["state", "district"])["bio_age_5_17"]
        .sum()
        .reset_index()
    )

    threshold = district_summary["bio_age_5_17"].quantile(0.10)

    anomalies = district_summary[
        district_summary["bio_age_5_17"] <= threshold
    ]

    st.metric("🚩 High-Risk Districts Identified", len(anomalies))

    st.markdown("### 🔴 Districts with Critically Low Biometric Updates (Age 5–17)")
    st.dataframe(anomalies.sort_values("bio_age_5_17"))

    # ---------------- HEATMAP VISUALIZATION ----------------
    st.subheader("🗺️ District-Level Biometric Update Heatmap (Age 5–17)")

    heatmap_data = (
        df.groupby(["state", "district"])["bio_age_5_17"]
        .sum()
        .reset_index()
    )

    fig = px.treemap(
        heatmap_data,
        path=["state", "district"],
        values="bio_age_5_17",
        color="bio_age_5_17",
        color_continuous_scale="Reds",
        title="Biometric Update Intensity Across Districts"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ---------------- MODEL PREPARATION ----------------
    st.subheader("🔮 Predictive Modeling (Age 5–17 Updates)")

    le_state = LabelEncoder()
    le_district = LabelEncoder()

    df["state_enc"] = le_state.fit_transform(df["state"])
    df["district_enc"] = le_district.fit_transform(df["district"])

    X = df[["state_enc", "district_enc", "day_of_week", "month"]]
    y = df["bio_age_5_17"]

    model = RandomForestRegressor(
        n_estimators=50,
        random_state=42
    )
    model.fit(X, y)

    st.success("Random Forest model trained successfully!")

    # ---------------- PREDICTION UI ----------------
    st.subheader("📈 Predict Next 7-Day Biometric Update Surge")

    selected_state = st.selectbox(
        "Select State",
        sorted(df["state"].unique())
    )

    selected_district = st.selectbox(
        "Select District",
        sorted(df[df["state"] == selected_state]["district"].unique())
    )

    if st.button("Predict Next 7 Days"):

        state_val = le_state.transform([selected_state])[0]
        district_val = le_district.transform([selected_district])[0]

        future_data = []

        for i in range(7):
            future_data.append([
                state_val,
                district_val,
                (pd.Timestamp.today().dayofweek + i) % 7,
                pd.Timestamp.today().month
            ])

        future_df = pd.DataFrame(
            future_data,
            columns=["state_enc", "district_enc", "day_of_week", "month"]
        )

        predictions = model.predict(future_df)

        st.success("✅ Prediction Complete")

        for i, value in enumerate(predictions, 1):
            st.write(f"**Day {i}:** {int(value)} expected biometric updates")

else:
    st.info("⬆️ Please upload the Aadhaar biometric CSV file to begin.")
