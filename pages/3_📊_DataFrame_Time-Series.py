
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import re
from datetime import datetime, timedelta

st.set_page_config(page_title="Earthquake Time Series Analysis", layout="wide")
st.title("🌍 Analyze the Time Series of Earthquake Frequency")

@st.cache_data
def load_data():
    # Load the new CSV format
    df = pd.read_csv("Earthquake_Data.csv")
    
    # Convert date and time columns
    df['DateTime'] = pd.to_datetime(df['DATE & TIME'], errors='coerce')
    
    # Extract separate date components for time series analysis
    df['Year'] = df['DateTime'].dt.year
    df['Month'] = df['DateTime'].dt.month
    df['Day'] = df['DateTime'].dt.day
    df['WeekDay'] = df['DateTime'].dt.day_name()
    
    # Ensure numeric values
    df['LATITUDE'] = pd.to_numeric(df['LATITUDE'], errors='coerce')
    df['LONGITUDE'] = pd.to_numeric(df['LONGITUDE'], errors='coerce')
    df['MAGNITUDE'] = pd.to_numeric(df['MAGNITUDE'], errors='coerce')
    df['DEPTH (KM)'] = pd.to_numeric(df['DEPTH (KM)'], errors='coerce')
    
    return df

df = load_data()

# Time Series Analysis - Direct approach without sidebar navigation
st.header("📈 Earthquake Frequency Over Time")

# Time grouping options
time_grouping = st.radio(
    "Select Time Grouping:",
    ["Daily", "Weekly", "Monthly", "Yearly"],
    horizontal=True
)

# Group data based on selection
if time_grouping == "Daily":
    df_time = df.groupby(pd.Grouper(key='DateTime', freq='D')).size().reset_index(name='Count')
    time_column = 'DateTime'
    title = "Daily Earthquake Frequency"
elif time_grouping == "Weekly":
    df_time = df.groupby(pd.Grouper(key='DateTime', freq='W')).size().reset_index(name='Count')
    time_column = 'DateTime'
    title = "Weekly Earthquake Frequency"
elif time_grouping == "Monthly":
    df_time = df.groupby(pd.Grouper(key='DateTime', freq='M')).size().reset_index(name='Count')
    time_column = 'DateTime'
    title = "Monthly Earthquake Frequency"
else:
    df_time = df.groupby(pd.Grouper(key='DateTime', freq='Y')).size().reset_index(name='Count')
    time_column = 'DateTime'
    title = "Yearly Earthquake Frequency"

# Check if we have valid time series data
if not df_time.empty:
    # Create animated time series plot
    fig = px.line(
        df_time, 
        x=time_column, 
        y='Count',
        title=title,
        labels={'Count': 'Number of Earthquakes', time_column: 'Date'},
        markers=True
    )
    
    # Add range slider for interactive time range selection
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        ),
        hovermode="x unified",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add automatic animation of earthquake occurrences over time
    st.subheader("🔄 Animated Earthquake Occurrences")
    
    # Create date range for animation
    min_date = df['DateTime'].min().date()
    max_date = df['DateTime'].max().date()
    
    # Create animation frame frequency options
    frame_freq = st.select_slider(
        "Animation Time Frame",
        options=["Daily", "Weekly", "Monthly", "Quarterly"],
        value="Weekly"
    )
    
    # Generate animation frames based on selected frequency
    if frame_freq == "Daily":
        delta = timedelta(days=1)
        date_sequence = pd.date_range(start=min_date, end=max_date, freq='D')
    elif frame_freq == "Weekly":
        delta = timedelta(days=7)
        date_sequence = pd.date_range(start=min_date, end=max_date, freq='W')
    elif frame_freq == "Monthly":
        date_sequence = pd.date_range(start=min_date, end=max_date, freq='M')
    else:  # Quarterly
        date_sequence = pd.date_range(start=min_date, end=max_date, freq='Q')
    
    # Create animation of earthquakes on map
    animation_data = []
    
    for i, frame_date in enumerate(date_sequence):
        if frame_freq == "Daily":
            frame_df = df[df['DateTime'].dt.date == frame_date.date()]
        elif frame_freq == "Weekly":
            end_date = frame_date + timedelta(days=6)
            frame_df = df[(df['DateTime'].dt.date >= frame_date.date()) & 
                          (df['DateTime'].dt.date <= end_date.date())]
        elif frame_freq == "Monthly":
            frame_df = df[(df['DateTime'].dt.year == frame_date.year) & 
                          (df['DateTime'].dt.month == frame_date.month)]
        else:  # Quarterly
            quarter_end = pd.Timestamp(frame_date) + pd.tseries.offsets.QuarterEnd()
            frame_df = df[(df['DateTime'] >= frame_date) & 
                          (df['DateTime'] <= quarter_end)]
        
        frame_data = dict(
            type='scattergeo',
            lon=frame_df['LONGITUDE'],
            lat=frame_df['LATITUDE'],
            text=frame_df.apply(lambda row: f"Magnitude: {row['MAGNITUDE']}<br>Depth: {row['DEPTH (KM)']} km<br>Date: {row['DateTime']}", axis=1),
            mode='markers',
            marker=dict(
                size=frame_df['MAGNITUDE'] * 2,
                color=frame_df['MAGNITUDE'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Magnitude'),
                cmin=df['MAGNITUDE'].min(),
                cmax=df['MAGNITUDE'].max(),
                opacity=0.8
            ),
            name=str(frame_date.date())
        )
        animation_data.append(frame_data)
    
    # Create figure with animation
    animation_fig = go.Figure(
        data=[animation_data[0]],
        layout=go.Layout(
            title=f'Earthquake Events ({frame_freq} Animation)',
            geo=dict(
                scope='asia',
                projection_type='mercator',
                center=dict(lon=df['LONGITUDE'].mean(), lat=df['LATITUDE'].mean()),
                projection_scale=30,  # Adjust based on your data's geographic spread
                showland=True,
                landcolor='rgb(243, 243, 243)',
                countrycolor='rgb(204, 204, 204)',
            ),
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [{
                    'label': '▶️ Play',
                    'method': 'animate',
                    'args': [None, {
                        'frame': {'duration': 1000, 'redraw': True},
                        'fromcurrent': True,
                        'transition': {'duration': 500}
                    }]
                }, {
                    'label': '⏸️ Pause',
                    'method': 'animate',
                    'args': [[None], {
                        'frame': {'duration': 0, 'redraw': False},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                }]
            }],
            sliders=[{
                'active': 0,
                'steps': [{'args': [[f.name], {
                    'frame': {'duration': 1000, 'redraw': True},
                    'mode': 'immediate',
                    'transition': {'duration': 500}
                }],
                    'label': f.name,
                    'method': 'animate'
                } for f in [go.Frame(name=d['name']) for d in animation_data]],
                'x': 0.1,
                'len': 0.9,
                'xanchor': 'left',
                'y': 0,
                'yanchor': 'top'
            }]
        ),
        frames=[go.Frame(data=[d], name=d['name']) for d in animation_data]
    )
    
    st.plotly_chart(animation_fig, use_container_width=True)
    
    # Add frequency patterns analysis
    st.subheader("📊 Earthquake Frequency Patterns")
    
    pattern_col1, pattern_col2 = st.columns(2)
    
    with pattern_col1:
        # Monthly distribution of earthquakes
        monthly_counts = df.groupby(df['DateTime'].dt.month).size().reset_index(name='Count')
        monthly_counts['Month'] = monthly_counts['DateTime'].apply(lambda x: datetime(2000, x, 1).strftime('%B'))
        
        month_order = [datetime(2000, i, 1).strftime('%B') for i in range(1, 13)]
        monthly_counts['Month'] = pd.Categorical(monthly_counts['Month'], categories=month_order, ordered=True)
        monthly_counts = monthly_counts.sort_values('Month')
        
        fig = px.bar(
            monthly_counts,
            x='Month',
            y='Count',
            title='Monthly Pattern of Earthquakes',
            color='Count',
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(xaxis_title='Month', yaxis_title='Number of Earthquakes')
        st.plotly_chart(fig, use_container_width=True)
    
    with pattern_col2:
        # Day of week distribution
        weekday_counts = df.groupby(df['WeekDay']).size().reset_index(name='Count')
        
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekday_counts['WeekDay'] = pd.Categorical(weekday_counts['WeekDay'], categories=weekday_order, ordered=True)
        weekday_counts = weekday_counts.sort_values('WeekDay')
        
        fig = px.bar(
            weekday_counts,
            x='WeekDay',
            y='Count',
            title='Day of Week Pattern of Earthquakes',
            color='Count',
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(xaxis_title='Day of Week', yaxis_title='Number of Earthquakes')
        st.plotly_chart(fig, use_container_width=True)

# Add footer with information
st.markdown("---")
st.markdown("### 🌍 Analyze the Time Series of Earthquake Frequency")
st.markdown("""
This interactive dashboard focuses on time-based patterns of earthquake frequency:
- Temporal analysis with adjustable time granularity (daily, weekly, monthly, yearly)
- Animated visualization of earthquake occurrences over time
- Distribution patterns by month and day of week
""")
