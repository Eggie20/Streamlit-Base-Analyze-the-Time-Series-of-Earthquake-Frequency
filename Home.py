
"""
Analyze the Time Series of Earthquake Frequency - Home Page
----------------------------------------------------------
An interactive Streamlit application for analyzing temporal patterns in earthquake frequency.
"""

import streamlit as st
import pandas as pd
import time
from datetime import datetime

from utils import load_data

# Page configuration
st.set_page_config(
    page_title="Earthquake Frequency Analysis",
    page_icon="📈",
    layout="wide"
)

# Apply custom styling
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stMetric {
        background-color: #1E1E1E;
        padding: 15px;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Main title
st.title("📈 Analyze the Time Series of Earthquake Frequency")

# Dashboard Introduction
st.markdown("""
Welcome to the **Earthquake Frequency Analysis Dashboard**!  
This interactive application focuses on:

> 📈 **"Analyzing the Time Series of Earthquake Frequency"**

This dashboard provides powerful visualization and analysis tools for understanding how 
earthquake frequency changes over time, revealing temporal patterns and trends in seismic activity rates.

---

### 🧪 Objectives of the Analysis:
- Visualize earthquake frequency changes over time using animated time series
- Detect trends and patterns in earthquake occurrence rates
- Analyze cyclical and seasonal variations in frequency
- Understand regional differences in earthquake frequency patterns

---

### 🗂️ Pages in this App:
- **📈 Seismic Activity Trends** – Interactive animated visualization of frequency trends
- **🗺️ Map View** – Geographic visualization of frequency distribution with animations
- **📊 Regional Analysis** – Animated comparison of frequency patterns across regions
- **📊 Dashboard Overview** – Interactive summary of frequency analysis with animations

---

### 📊 Time Series Analysis Focus:
- **Frequency Analysis**: Analyzing counts of earthquakes over time periods
- **Trend Detection**: Identifying increasing or decreasing patterns in occurrence rates
- **Seasonality Analysis**: Uncovering cyclical patterns in earthquake frequency
- **Anomaly Detection**: Highlighting unusual spikes or drops in earthquake rates

---

### 🔬 Analysis Methodology:
- Time series aggregation at different temporal scales (daily, weekly, monthly)
- Moving averages and smoothing techniques to reveal underlying trends
- Seasonal decomposition to separate trend, seasonal, and residual components
- Animation-based visualization to enhance pattern recognition

---

Feel free to explore the application to interact with the frequency analysis tools!
""")

# Load the data
df = load_data()

if len(df) == 0:
    st.error("No data available. Please check that the data file exists and contains valid data.")
    st.stop()

# Basic data metrics without filtering
total_earthquakes = len(df)
date_range_days = (df['DATETIME'].max() - df['DATETIME'].min()).days if len(df) > 1 else 1
avg_daily = total_earthquakes / max(1, date_range_days)

# Display metrics in columns
st.markdown("### Key Frequency Metrics")
col1, col2 = st.columns(2)

with col1:
    st.metric("Total Earthquake Count", f"{total_earthquakes:,}")
with col2:
    st.metric("Average Daily Frequency", f"{avg_daily:.2f}")

# Footer
st.markdown("---")
st.markdown("""
<p style='text-align: center; color: #888888;'>
Earthquake Frequency Analysis Dashboard | Data Explorer<br>
</p>
""", unsafe_allow_html=True)
