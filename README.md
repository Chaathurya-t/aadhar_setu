# UIDAI Hackathon 2026: Aadhaar Biometric Analytics Dashboard

### Team ID: UIDAI_2958
### Project Category: Analytics & Predictive Modeling

## 📌 Project Overview
This project is a web-based dashboard built for the *UIDAI Data Hackathon 2026*. It analyzes Aadhaar biometric update trends (specifically for the 5-17 age group) and uses Machine Learning to predict future demand at the district level.

The goal is to help UIDAI move from *reactive* to *proactive* resource allocation (Camp Mode) by identifying high-demand regions before queues form.

## 🚀 Key Features
* *Interactive Map:* Visualizes update density across districts using Plotly.
* *Demand Prediction:* Uses a *Random Forest Regressor* to predict update volume for the next 30 days.
* *Demographic Insights:* Analyzes the ratio of child vs. adult updates to target school-level interventions.
* *Temporal Trends:* Tracks weekly and monthly spikes in enrollment data.

## 🛠️ Tech Stack
* *Frontend:* Streamlit (Python)
* *Data Manipulation:* Pandas, NumPy
* *Machine Learning:* Scikit-Learn (RandomForestRegressor)
* *Visualization:* Plotly Express

## 📂 Repository Structure
* app.py: The main Streamlit application file that runs the dashboard.
* model.py: Contains the Machine Learning logic for demand forecasting.
* analysis.py: Handles data cleaning and statistical analysis of the raw CSVs.
* requirements.txt: List of all Python dependencies required to run the project.
