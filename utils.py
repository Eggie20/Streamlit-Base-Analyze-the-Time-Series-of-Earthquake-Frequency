"""
Utility functions for the Earthquake Time Series Analysis Dashboard.
Contains common functions for data loading, preprocessing, and time series analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.nonparametric.smoothers_lowess import lowess

# Caching the data loading to improve performance
@st.cache_data
def load_data():
    """
    Load and preprocess the earthquake data from the CSV file.
    
    Returns:
        pandas.DataFrame: Processed earthquake data
    """
    try:
        df = pd.read_csv("Earthquake_Data.csv")
        
        # Convert column names to uppercase for consistency
        df.columns = df.columns.str.upper()
        
        # Parse the DATE & TIME column using the format: "31 January 2023 - 11:58 PM"
        df['DATETIME'] = pd.to_datetime(df['DATE & TIME'], format="%d %B %Y - %I:%M %p", errors='coerce')
        
        # Convert numeric columns
        df['LATITUDE'] = pd.to_numeric(df['LATITUDE'], errors='coerce')
        df['LONGITUDE'] = pd.to_numeric(df['LONGITUDE'], errors='coerce')
        df['MAGNITUDE'] = pd.to_numeric(df['MAGNITUDE'], errors='coerce')
        df['DEPTH (KM)'] = pd.to_numeric(df['DEPTH (KM)'], errors='coerce')
        
        # Create standardized column names for easier access
        df['Magnitude'] = df['MAGNITUDE']
        df['Depth'] = df['DEPTH (KM)']
        
        # Ensure Province column is standardized
        if 'PROVINCE' in df.columns:
            df['Province'] = df['PROVINCE']
            
        # Add date components for time series analysis
        df['Date'] = df['DATETIME'].dt.date
        df['Year'] = df['DATETIME'].dt.year
        df['Month'] = df['DATETIME'].dt.month
        df['Week'] = df['DATETIME'].dt.isocalendar().week
        df['DayOfYear'] = df['DATETIME'].dt.dayofyear
        df['DayOfWeek'] = df['DATETIME'].dt.dayofweek
        df['Hour'] = df['DATETIME'].dt.hour
        
        # Drop rows with missing critical data
        df = df.dropna(subset=['DATETIME', 'MAGNITUDE'])
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

def create_sidebar_filters(df):
    """
    Creates common sidebar filters for the dashboard.
    
    Args:
        df (pandas.DataFrame): The earthquake data
        
    Returns:
        tuple: Selected date range, minimum magnitude, and province
    """
    st.sidebar.header("Time Series Filters")
    
    # Date range filter
    min_date = df['DATETIME'].min().date()
    max_date = df['DATETIME'].max().date()
    
    # Default to showing the last 90 days
    default_start = max(min_date, max_date - timedelta(days=90))
    
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(default_start, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Magnitude filter
    min_magnitude = st.sidebar.slider(
        "Minimum Magnitude", 
        min_value=float(max(1.0, df['MAGNITUDE'].min())),
        max_value=float(min(9.0, df['MAGNITUDE'].max())), 
        value=2.0,
        step=0.1
    )
    
    # Province selection
    all_provinces = sorted(df['PROVINCE'].unique())
    default_provinces = []
    
    # Get the provinces with the most events to use as defaults
    if len(all_provinces) > 0:
        top_provinces = df['PROVINCE'].value_counts().head(5).index.tolist()
        default_provinces = top_provinces[:3] if len(top_provinces) >= 3 else top_provinces
    
    selected_provinces = st.sidebar.multiselect(
        "Select Provinces",
        options=["All"] + all_provinces,
        default=["All"]
    )
    
    # Handle "All" selection
    if "All" in selected_provinces or not selected_provinces:
        selected_provinces = all_provinces
    
    return date_range, min_magnitude, selected_provinces

def apply_data_filters(df, date_range, min_magnitude, provinces):
    """
    Apply the selected filters to the data.
    
    Args:
        df (pandas.DataFrame): The earthquake data
        date_range (tuple): Selected start and end dates
        min_magnitude (float): Minimum magnitude
        provinces (list): Selected provinces
        
    Returns:
        pandas.DataFrame: Filtered data
    """
    filtered_df = df.copy()
    
    # Apply magnitude filter
    filtered_df = filtered_df[filtered_df['MAGNITUDE'] >= min_magnitude]
    
    # Apply date filter if we have a valid range
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = filtered_df[
            (filtered_df['DATETIME'].dt.date >= start_date) & 
            (filtered_df['DATETIME'].dt.date <= end_date)
        ]
    
    # Apply province filter
    if provinces and len(provinces) > 0:
        filtered_df = filtered_df[filtered_df['PROVINCE'].isin(provinces)]
    
    return filtered_df

def aggregate_by_time(df, freq='D'):
    """
    Aggregate earthquake data by time period.
    
    Args:
        df (pandas.DataFrame): Earthquake data
        freq (str): Frequency for aggregation ('D'=daily, 'W'=weekly, 'M'=monthly, 'Y'=yearly)
        
    Returns:
        pandas.DataFrame: Aggregated data
    """
    # Create a time series with the count of earthquakes
    ts = df.set_index('DATETIME')
    
    # Resample by the specified frequency and count events
    counts = ts.resample(freq).size().reset_index(name='Count')
    
    # Also calculate average magnitude
    magnitude = ts.resample(freq)['MAGNITUDE'].mean().reset_index(name='AvgMagnitude')
    
    # Merge the two series
    result = pd.merge(counts, magnitude, on='DATETIME')
    
    return result

def plot_time_series(df, y_column='Count', title='Earthquake Frequency Over Time'):
    """
    Create a time series plot of earthquake data.
    
    Args:
        df (pandas.DataFrame): Aggregated time series data
        y_column (str): Column to plot on y-axis
        title (str): Plot title
        
    Returns:
        plotly.graph_objects.Figure: Time series plot
    """
    fig = px.line(
        df, 
        x='DATETIME', 
        y=y_column,
        title=title,
        labels={'DATETIME': 'Date', y_column: y_column},
        line_shape='linear'
    )
    
    fig.update_layout(
        hovermode='x unified',
        plot_bgcolor='white',
        xaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
        )
    )
    
    return fig

def calculate_rolling_stats(df, window=7):
    """
    Calculate rolling statistics for time series data.
    
    Args:
        df (pandas.DataFrame): Time series data with 'DATETIME' and 'Count' columns
        window (int): Window size for rolling calculations
        
    Returns:
        pandas.DataFrame: DataFrame with rolling statistics added
    """
    # Make a copy to avoid modifying the original
    result = df.copy()
    
    # Calculate rolling mean (moving average)
    result['RollingMean'] = result['Count'].rolling(window=window).mean()
    
    # Calculate rolling standard deviation
    result['RollingStd'] = result['Count'].rolling(window=window).std()
    
    # Calculate upper and lower bounds (mean ± 2*std)
    result['UpperBound'] = result['RollingMean'] + 2 * result['RollingStd']
    result['LowerBound'] = result['RollingMean'] - 2 * result['RollingStd']
    result['LowerBound'] = result['LowerBound'].clip(lower=0)  # Ensure non-negative
    
    return result

def plot_rolling_statistics(df, window=7, title='Rolling Statistics of Earthquake Frequency'):
    """
    Create a plot with rolling statistics.
    
    Args:
        df (pandas.DataFrame): Time series data with rolling statistics
        window (int): Window size used for calculations
        title (str): Plot title
        
    Returns:
        plotly.graph_objects.Figure: Plot with rolling statistics
    """
    fig = go.Figure()
    
    # Add raw count
    fig.add_trace(go.Scatter(
        x=df['DATETIME'],
        y=df['Count'],
        mode='lines',
        name='Daily Count',
        line=dict(color='rgba(0, 123, 255, 0.5)', width=1)
    ))
    
    # Add rolling mean
    fig.add_trace(go.Scatter(
        x=df['DATETIME'],
        y=df['RollingMean'],
        mode='lines',
        name=f'{window}-Day Moving Average',
        line=dict(color='rgba(255, 0, 0, 1)', width=2)
    ))
    
    # Add upper and lower bounds
    fig.add_trace(go.Scatter(
        x=df['DATETIME'],
        y=df['UpperBound'],
        mode='lines',
        name='Upper Bound (2σ)',
        line=dict(color='rgba(0, 0, 0, 0)', width=0)
    ))
    
    fig.add_trace(go.Scatter(
        x=df['DATETIME'],
        y=df['LowerBound'],
        mode='lines',
        name='Lower Bound (2σ)',
        fill='tonexty',
        fillcolor='rgba(255, 0, 0, 0.1)',
        line=dict(color='rgba(0, 0, 0, 0)', width=0)
    ))
    
    # Add anomalies (points outside the bounds)
    anomalies = df[(df['Count'] > df['UpperBound']) | (df['Count'] < df['LowerBound'])]
    
    if not anomalies.empty:
        fig.add_trace(go.Scatter(
            x=anomalies['DATETIME'],
            y=anomalies['Count'],
            mode='markers',
            name='Anomalies',
            marker=dict(color='black', size=8, symbol='circle')
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Number of Earthquakes',
        hovermode='x unified',
        plot_bgcolor='white',
        xaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
        )
    )
    
    return fig

def decompose_time_series(ts_data, period=7, model='additive'):
    """
    Decompose a time series into trend, seasonal, and residual components.
    
    Args:
        ts_data (pandas.Series): Time series data indexed by date
        period (int): Period for seasonal decomposition
        model (str): Type of decomposition ('additive' or 'multiplicative')
        
    Returns:
        statsmodels.tsa.seasonal.DecomposeResult: Decomposition result
    """
    # Ensure we have a complete series without gaps
    idx = pd.date_range(ts_data.index.min(), ts_data.index.max())
    ts_data = ts_data.reindex(idx, fill_value=0)
    
    # Decompose the time series
    result = seasonal_decompose(ts_data, model=model, period=period)
    
    return result

def plot_decomposition(decomp_result, title='Time Series Decomposition'):
    """
    Plot the decomposition of a time series.
    
    Args:
        decomp_result: Result of seasonal_decompose
        title (str): Plot title
        
    Returns:
        plotly.graph_objects.Figure: Decomposition plot
    """
    # Create subplots
    fig = go.Figure()
    
    # Original time series
    fig.add_trace(go.Scatter(
        x=decomp_result.observed.index,
        y=decomp_result.observed.values,
        mode='lines',
        name='Observed',
        line=dict(color='blue')
    ))
    
    # Trend component
    fig.add_trace(go.Scatter(
        x=decomp_result.trend.index,
        y=decomp_result.trend.values,
        mode='lines',
        name='Trend',
        line=dict(color='red')
    ))
    
    # Seasonal component
    fig.add_trace(go.Scatter(
        x=decomp_result.seasonal.index,
        y=decomp_result.seasonal.values,
        mode='lines',
        name='Seasonal',
        line=dict(color='green')
    ))
    
    # Residual component
    fig.add_trace(go.Scatter(
        x=decomp_result.resid.index,
        y=decomp_result.resid.values,
        mode='lines',
        name='Residual',
        line=dict(color='purple')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Value',
        hovermode='x unified',
        plot_bgcolor='white',
        xaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
        )
    )
    
    return fig

def create_heatmap(df, value_col='Count', title='Earthquake Frequency Heatmap'):
    """
    Create a heatmap of earthquake frequency by day of week and hour.
    
    Args:
        df (pandas.DataFrame): Earthquake data with datetime information
        value_col (str): Column to use for the heatmap values
        title (str): Plot title
        
    Returns:
        plotly.graph_objects.Figure: Heatmap figure
    """
    # Extract day of week and hour
    df_heatmap = df.copy()
    df_heatmap['DayOfWeek'] = df_heatmap['DATETIME'].dt.day_name()
    df_heatmap['Hour'] = df_heatmap['DATETIME'].dt.hour
    
    # Pivot the data to create a day-of-week by hour heatmap
    heatmap_data = df_heatmap.pivot_table(
        index='DayOfWeek', 
        columns='Hour',
        values='DATETIME',
        aggfunc='count'
    )
    
    # Reorder days of week
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_data = heatmap_data.reindex(day_order)
    
    # Create a heatmap figure
    fig = px.imshow(
        heatmap_data, 
        title=title,
        labels=dict(x="Hour of Day", y="Day of Week", color="Frequency"),
        x=[str(i) for i in range(24)],
        y=day_order,
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        xaxis=dict(
            tickmode='linear',
            tick0=0,
            dtick=1
        ),
        plot_bgcolor='white'
    )
    
    return fig

def create_monthly_comparison(df):
    """
    Create a DataFrame for comparing earthquake frequency across months and years.
    
    Args:
        df (pandas.DataFrame): Earthquake data with datetime information
        
    Returns:
        pandas.DataFrame: Reshaped data for monthly comparison
    """
    # Extract month and year
    df_monthly = df.copy()
    df_monthly['Month'] = df_monthly['DATETIME'].dt.strftime('%b')
    df_monthly['Year'] = df_monthly['DATETIME'].dt.year
    
    # Count earthquakes by month and year
    monthly_counts = df_monthly.groupby(['Year', 'Month']).size().reset_index(name='Count')
    
    # Pivot to create a table with years as rows and months as columns
    pivot_table = monthly_counts.pivot_table(
        values='Count',
        index='Year',
        columns='Month',
        fill_value=0
    )
    
    # Reorder columns to be in calendar order
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    pivot_table = pivot_table.reindex(columns=month_order)
    
    return pivot_table

def plot_monthly_heatmap(monthly_pivot, title='Monthly Earthquake Frequency by Year'):
    """
    Create a heatmap of monthly earthquake frequency by year.
    
    Args:
        monthly_pivot (pandas.DataFrame): Pivoted data with years as rows and months as columns
        title (str): Plot title
        
    Returns:
        plotly.graph_objects.Figure: Heatmap figure
    """
    # Create heatmap
    fig = px.imshow(
        monthly_pivot,
        title=title,
        labels=dict(x="Month", y="Year", color="Frequency"),
        color_continuous_scale='Viridis',
        aspect="auto"
    )
    
    fig.update_layout(
        xaxis=dict(
            title="Month",
            tickangle=-45
        ),
        yaxis=dict(
            title="Year",
            dtick=1
        ),
        plot_bgcolor='white'
    )
    
    return fig

def detect_trend(time_series):
    """
    Detect the trend in a time series using smoothing techniques.
    
    Args:
        time_series (pandas.Series): Time series data
        
    Returns:
        tuple: Smoothed time series and trend description
    """
    # Apply LOWESS smoothing
    y = time_series.values
    x = np.arange(len(y))
    
    # Perform LOWESS smoothing
    filtered = lowess(y, x, frac=0.1)
    
    # Create a series with the smoothed values
    smoothed = pd.Series(filtered[:, 1], index=time_series.index)
    
    # Determine trend
    start_value = smoothed.iloc[0]
    end_value = smoothed.iloc[-1]
    
    if end_value > start_value * 1.1:
        trend = "increasing"
    elif end_value < start_value * 0.9:
        trend = "decreasing"
    else:
        trend = "stable"
    
    # Calculate percentage change
    percent_change = ((end_value - start_value) / start_value) * 100 if start_value != 0 else 0
    
    return smoothed, trend, percent_change

def compare_regions_over_time(df, regions, freq='M'):
    """
    Compare earthquake frequency across different regions over time.
    
    Args:
        df (pandas.DataFrame): Earthquake data
        regions (list): List of regions to compare
        freq (str): Frequency for aggregation
        
    Returns:
        pandas.DataFrame: Time series data for each region
    """
    result = pd.DataFrame()
    
    for region in regions:
        # Filter data for this region
        region_data = df[df['PROVINCE'] == region]
        
        # Skip if no data for this region
        if len(region_data) == 0:
            continue
        
        # Create time series
        ts = region_data.set_index('DATETIME')
        counts = ts.resample(freq).size().reset_index(name=region)
        
        if result.empty:
            result = counts
        else:
            # Merge with existing result
            result = pd.merge(result, counts, on='DATETIME', how='outer')
    
    # Fill NaN values with 0
    result = result.fillna(0)
    
    return result

def plot_region_comparison(region_data, title='Regional Comparison of Earthquake Frequency'):
    """
    Create a line plot comparing earthquake frequency across regions.
    
    Args:
        region_data (pandas.DataFrame): Time series data for each region
        title (str): Plot title
        
    Returns:
        plotly.graph_objects.Figure: Comparison plot
    """
    fig = go.Figure()
    
    # Add a line for each region
    for column in region_data.columns:
        if column != 'DATETIME':
            fig.add_trace(go.Scatter(
                x=region_data['DATETIME'],
                y=region_data[column],
                mode='lines',
                name=column
            ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Number of Earthquakes',
        hovermode='x unified',
        plot_bgcolor='white',
        legend_title='Region',
        xaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
        )
    )
    
    return fig
