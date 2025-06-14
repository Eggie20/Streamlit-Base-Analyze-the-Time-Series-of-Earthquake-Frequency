
import streamlit as st
import pandas as pd
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
import calendar
import io

st.set_page_config(page_title="Analyze Time Series of Earthquake Frequency", page_icon="📈", layout="wide")

# Header and Study Overview
st.title("📈 Analyze Time Series of Earthquake Frequency")
st.sidebar.header("Earthquake Data Filter")

# Function to assign months to seasons
def assign_season(month):
    if month in [12, 1, 2, 3, 4, 5]:  # Dry Season (Dec-May)
        return "Dry Season"
    else:  # Wet Season (June-November)
        return "Wet Season"

# Load and preprocess data
@st.cache_data
def load_data():
    file_path = "Earthquake_Data.csv"
    try:
        df = pd.read_csv(file_path)

        # Convert DATE column
        df["DATE"] = pd.to_datetime(df["DATE"], errors='coerce')
        df = df.dropna(subset=["DATE"])
        df["MONTH"] = df["DATE"].dt.month_name()
        df["MONTH_NUM"] = df["DATE"].dt.month
        df["YEAR"] = df["DATE"].dt.year
        df["SEASON"] = df["MONTH_NUM"].apply(assign_season)  # Add season column
        return df
    except FileNotFoundError:
        st.error(f"❌ File '{file_path}' not found.")
        return None

# Generate downloadable report
@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

df = load_data()

if df is not None:
    # Sidebar filters
    # Date range filter
    if 'DATE' in df.columns:
        min_date = df['DATE'].min().date()
        max_date = df['DATE'].max().date()
        
        date_range = st.sidebar.date_input(
            "Select date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        if len(date_range) == 2:
            start_date, end_date = date_range
            df = df[(df['DATE'] >= pd.Timestamp(start_date)) & 
                    (df['DATE'] <= pd.Timestamp(end_date))]
    
    # Province filter
    if 'PROVINCE' in df.columns:
        provinces = sorted(df["PROVINCE"].dropna().unique())
        selected_provinces = st.sidebar.multiselect("Select Provinces (Optional)", provinces)
        if selected_provinces:
            df = df[df["PROVINCE"].isin(selected_provinces)]

    # Magnitude filter if available
    if 'MAGNITUDE' in df.columns:
        mag_range = st.sidebar.slider(
            "Magnitude Range", 
            float(df['MAGNITUDE'].min()), 
            float(df['MAGNITUDE'].max()), 
            (float(df['MAGNITUDE'].min()), float(df['MAGNITUDE'].max()))
        )
        
        df = df[(df['MAGNITUDE'] >= mag_range[0]) & 
                (df['MAGNITUDE'] <= mag_range[1])]
    
    # Filter by month and year for map
    st.sidebar.subheader("Filter by Month and Year")
    unique_months = sorted(df["MONTH"].dropna().unique(), key=lambda m: list(calendar.month_name).index(m) if m in calendar.month_name else 0)
    unique_years = sorted(df["YEAR"].dropna().unique())
    
    if unique_months and unique_years:
        selected_month = st.sidebar.selectbox("Select Month", unique_months)
        selected_year = st.sidebar.selectbox("Select Year", unique_years)
        filtered_map_df = df[(df["MONTH"] == selected_month) & (df["YEAR"] == selected_year)]
    else:
        filtered_map_df = pd.DataFrame()
    
    # Add download option in sidebar
    st.sidebar.subheader("Download Data")
    csv = convert_df(df)
    st.sidebar.download_button(
        label="Download Filtered Data as CSV",
        data=csv,
        file_name='earthquake_time_series_data.csv',
        mime='text/csv',
    )
    
    # OVERVIEW SECTION (Previously Tab 1)
    st.header("Study Overview")
    
    # Basic metrics
    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
    with metrics_col1:
        st.metric("Total Earthquakes", f"{len(df):,}")
    with metrics_col2:
        st.metric("Date Range", f"{df['DATE'].min().date()} to {df['DATE'].max().date()}")
    with metrics_col3:
        if 'MAGNITUDE' in df.columns:
            st.metric("Average Magnitude", f"{df['MAGNITUDE'].mean():.2f}")
    with metrics_col4:
        if 'MAGNITUDE' in df.columns:
            st.metric("Max Magnitude", f"{df['MAGNITUDE'].max():.2f}")
    
    st.markdown("---")
    
    st.subheader("📌 Project Description")
    st.write("""
    This project analyzes earthquake data in the Philippines to identify **temporal patterns** and seasonality in seismic activity.
    We examine the relationship between earthquake occurrences and the Philippines' two main seasons: **Dry Season** (December-May) 
    and **Wet Season** (June-November).
    """)
    
    st.markdown("""
    The Philippines experiences an average of 20 earthquakes per day, with approximately 100-150 felt earthquakes annually.
    Understanding temporal patterns can help with:
    - Resource allocation for emergency response
    - Public awareness campaigns during high-risk periods
    - Infrastructure planning and maintenance scheduling
    """)
    
    st.subheader("🔍 Methodology Summary")
    st.write("""
    Our time series analysis follows these steps:
    1. **Data collection** from Philippine Institute of Volcanology and Seismology (PHIVOLCS)
    2. **Data preprocessing** including time-based feature extraction
    3. **Time series decomposition** to identify trends, seasonality, and residuals
    4. **Seasonal pattern analysis** across different time scales (daily, monthly, yearly)
    5. **Statistical testing** to validate observed patterns
    """)
    
    st.markdown("""
    The analysis employs various visualization techniques:
    - **Bar charts** for frequency analysis
    - **Line graphs** for trend identification
    - **Box plots** for distribution analysis
    - **Heatmaps** for pattern recognition
    - **Geographic mapping** for spatial-temporal correlation
    """)
    
    # About this application
    st.subheader("📊 About This Application")
    st.markdown("""
    This interactive dashboard provides comprehensive time series analysis of earthquake data, allowing users to:
    
    - Filter data by date range, province, and magnitude
    - Visualize earthquake distribution patterns
    - Compare seasonal differences in earthquake frequency and intensity
    - Export filtered data for further analysis
    """)
    
    # Additional content section for data exploration
    st.header("Data Exploration")
    
    # Show data preview
    st.subheader("Data Preview")
    st.dataframe(df.head(10))
    
    # Basic data statistics
    st.subheader("Data Statistics")
    st.write(df.describe().T)

    # Monthly earthquake frequency visualization
    st.subheader("Monthly Earthquake Frequency")
    
    # Group by month
    monthly_data = df.groupby(['MONTH_NUM', 'MONTH']).size().reset_index(name='Count')
    monthly_data = monthly_data.sort_values('MONTH_NUM')
    
    # Create bar chart
    fig = plt.figure(figsize=(12, 6))
    plt.bar(monthly_data['MONTH'], monthly_data['Count'], color=plt.cm.tab10.colors)
    
    plt.title('Earthquake Frequency by Month')
    plt.xlabel('Month')
    plt.ylabel('Number of Earthquakes')
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45)
    
    st.pyplot(fig)

    # Footer
    st.markdown("---")
    st.caption("Time Series Analysis of Earthquake Frequency | Developed by Earthquake Research Team")
else:
    st.error("Could not load earthquake data. Please check the data source and file format.")
