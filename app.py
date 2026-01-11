import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="UIDAI | Aadhaar-Setu",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================== UIDAI STYLE CSS ==================
st.markdown("""
<style>
body {
    background-color: #F4F6F9;
}

.uidai-header {
    font-size: 36px;
    font-weight: 700;
    color: #0B3C5D;
}

.uidai-subheader {
    font-size: 18px;
    color: #334155;
}

.section-title {
    font-size: 22px;
    font-weight: 600;
    color: #0B3C5D;
    margin-top: 20px;
}

.card {
    background-color: #FFFFFF;
    padding: 18px;
    border-radius: 8px;
    border-left: 6px solid #0B3C5D;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
}

.metric-box {
    background-color: #FFFFFF;
    padding: 15px;
    border-radius: 8px;
    border: 1px solid #E5E7EB;
    text-align: center;
}

.footer {
    color: #64748B;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# ================== HEADER ==================
st.markdown('<div class="uidai-header">Unique Identification Authority of India</div>', unsafe_allow_html=True)
st.markdown('<div class="uidai-subheader">Aadhaar-Setu: Predictive Infrastructure Planning Dashboard</div>', unsafe_allow_html=True)

st.divider()

# ================== SIDEBAR ==================
st.sidebar.title("📂 Data Management")
uploaded_file = st.sidebar.file_uploader(
    "Upload Aadhaar Biometric CSV",
    type="csv"
)

st.sidebar.markdown("""
### About Aadhaar-Setu
This internal analytics tool assists UIDAI in:
- Identifying service gaps
- Visualizing biometric update pressure
- Predicting upcoming demand
- Improving citizen service delivery
""")

# ================== MAIN LOGIC ==================
if uploaded_file:

    df = pd.read_csv(uploaded_file)

    # ================== FEATURE ENGINEERING ==================
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month

    # ================== KPI DASHBOARD ==================
    st.markdown('<div class="section-title">National Overview</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Biometric Updates (5–17)", f"{int(df['bio_age_5_17'].sum()):,}")
    col2.metric("States Covered", df["state"].nunique())
    col3.metric("Districts Covered", df["district"].nunique())

    st.divider()

    # ================== DATA PREVIEW ==================
    st.markdown('<div class="section-title">Dataset Snapshot</div>', unsafe_allow_html=True)
    st.dataframe(df.head(), use_container_width=True)

    # ================== ANOMALY DETECTION ==================
    st.markdown('<div class="section-title">Service Gap Identification</div>', unsafe_allow_html=True)

    district_summary = (
        df.groupby(["state", "district"])["bio_age_5_17"]
        .sum()
        .reset_index()
    )

    threshold = district_summary["bio_age_5_17"].quantile(0.10)
    anomalies = district_summary[district_summary["bio_age_5_17"] <= threshold]

    st.markdown("""
    <div class="card">
    Districts listed below show critically low biometric update activity for children (Age 5–17).
    These areas may require awareness campaigns or deployment of mobile enrolment units.
    </div>
    """, unsafe_allow_html=True)

    st.metric("High-Risk Districts Identified", len(anomalies))
    st.dataframe(anomalies.sort_values("bio_age_5_17"), use_container_width=True)

    st.divider()

    # ================== HEATMAP ==================
    st.markdown('<div class="section-title">Biometric Update Pressure Map</div>', unsafe_allow_html=True)

    fig = px.treemap(
        district_summary,
        path=["state", "district"],
        values="bio_age_5_17",
        color="bio_age_5_17",
        color_continuous_scale="Reds"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ================== PREDICTIVE MODEL ==================
    st.markdown('<div class="section-title">Predictive Demand Forecasting</div>', unsafe_allow_html=True)

    le_state = LabelEncoder()
    le_district = LabelEncoder()

    df["state_enc"] = le_state.fit_transform(df["state"])
    df["district_enc"] = le_district.fit_transform(df["district"])

    X = df[["state_enc", "district_enc", "day_of_week", "month"]]
    y = df["bio_age_5_17"]

    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)

    colA, colB = st.columns(2)
    selected_state = colA.selectbox("State", sorted(df["state"].unique()))
    selected_district = colB.selectbox(
        "District",
        sorted(df[df["state"] == selected_state]["district"].unique())
    )

    if st.button("Generate 7-Day Forecast"):

        state_val = le_state.transform([selected_state])[0]
        district_val = le_district.transform([selected_district])[0]

        future = []
        for i in range(7):
            future.append([
                state_val,
                district_val,
                (pd.Timestamp.today().dayofweek + i) % 7,
                pd.Timestamp.today().month
            ])

        future_df = pd.DataFrame(
            future,
            columns=["state_enc", "district_enc", "day_of_week", "month"]
        )

        preds = model.predict(future_df)

        st.success("Forecast Generated Successfully")

        for i, p in enumerate(preds, 1):
            st.write(f"Day {i}: **{int(p)} expected updates**")

        st.markdown("""
        <div class="card">
        <b>Policy Insight:</b> Forecast-based deployment of biometric operators and kits
        can significantly reduce queue times and improve Aadhaar service delivery.
        </div>
        """, unsafe_allow_html=True)

else:
    st.info("Please upload the Aadhaar biometric dataset to access the dashboard.")

# ================== FOOTER ==================
st.divider()
st.markdown('<div class="footer">© UIDAI Hackathon 2026 | Aadhaar-Setu Prototype</div>', unsafe_allow_html=True)

