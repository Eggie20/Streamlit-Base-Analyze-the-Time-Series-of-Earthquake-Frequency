
"""
Animated Time Series Analysis of Earthquake Frequency
----------------------------------------------------
An interactive animated visualization showing how earthquake frequency changes over time.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
from datetime import datetime

from utils import load_data, create_sidebar_filters, apply_data_filters
from utils import aggregate_by_time, detect_trend

# Page configuration
st.set_page_config(
    page_title="Animated Earthquake Frequency Analysis",
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
    .stAlert {
        padding: 20px;
        border-radius: 10px;
    }
    .plot-container {
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        border-radius: 10px;
        padding: 10px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 24px;
        border: none;
        cursor: pointer;
        transition-duration: 0.4s;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
""", unsafe_allow_html=True)

# Page header
st.title("📈 Animated Earthquake Frequency Time Series Analysis")
st.markdown("<p style='font-size: 1.2em; color: #888888;'>Watch the evolution of earthquake frequency patterns over time</p>", unsafe_allow_html=True)

# Introduction text
st.write("""
    This page provides an interactive animated visualization of how earthquake frequency changes over time.
    The animation helps identify temporal patterns, trends, and cycles in earthquake occurrence rates that
    might not be immediately apparent in static visualizations.
    
    Use the controls in the sidebar to customize the visualization and filter the data to focus on specific
    time periods, regions, or magnitude ranges.
""")

# Load data
try:
    df = load_data()
    
    if len(df) == 0:
        st.error("No data available. Please check that the data file exists and contains valid data.")
        st.stop()
    
    # Create sidebar filters
    date_range, min_magnitude, selected_provinces = create_sidebar_filters(df)
    
    # Sidebar options for animation and analysis
    st.sidebar.header("Frequency Analysis Options")
    time_aggregation = st.sidebar.selectbox(
        "Time Aggregation",
        options=["Daily", "Weekly", "Monthly", "Yearly"],
        index=1,  # Default to weekly
        help="Select time period for aggregating earthquake counts"
    )
    
    # Animation settings
    st.sidebar.header("Animation Settings")
    animation_speed = st.sidebar.slider(
        "Animation Speed", 
        min_value=0.1, 
        max_value=2.0, 
        value=1.0, 
        step=0.1,
        help="Control how fast new data points appear"
    )
    
    animation_frame_skipping = st.sidebar.slider(
        "Frame Resolution", 
        min_value=1, 
        max_value=10, 
        value=1, 
        step=1,
        help="Higher values improve performance by skipping frames (smoother animation but less granular)"
    )
    
    use_smooth_transitions = st.sidebar.checkbox(
        "Smooth Transitions", 
        True,
        help="Enable curved lines between data points"
    )
    
    show_area_fill = st.sidebar.checkbox(
        "Area Fill Effect", 
        True,
        help="Fill area under the frequency line"
    )
    
    # Performance optimization
    enable_performance_mode = st.sidebar.checkbox(
        "Enable Performance Mode", 
        True,
        help="Optimize animation for smoother performance"
    )
    
    # Apply filters
    filtered_df = apply_data_filters(df, date_range, min_magnitude, selected_provinces)
    
    if len(filtered_df) == 0:
        st.warning("No data matches your filter criteria. Please adjust your selections.")
        st.stop()
    
    # Determine frequency for aggregation
    freq_map = {
        "Daily": "D",
        "Weekly": "W",
        "Monthly": "M", 
        "Yearly": "Y"
    }
    
    # Aggregate data
    agg_freq = freq_map[time_aggregation]
    time_series_data = aggregate_by_time(filtered_df, freq=agg_freq)
    
    # Display key metrics
    st.markdown("### Frequency Analysis Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_earthquakes = len(filtered_df)
        st.metric("Total Events", f"{total_earthquakes:,}")
    
    with col2:
        date_range_days = (filtered_df['DATETIME'].max() - filtered_df['DATETIME'].min()).days
        avg_daily = total_earthquakes / max(1, date_range_days)
        st.metric("Average Daily Frequency", f"{avg_daily:.2f}")
    
    with col3:
        if len(time_series_data) > 0:
            max_count = time_series_data['Count'].max()
            max_date = time_series_data.loc[time_series_data['Count'].idxmax(), 'DATETIME'].strftime('%Y-%m-%d')
            st.metric("Peak Frequency", f"{max_count}")
            st.caption(f"on {max_date}")
        else:
            st.metric("Peak Frequency", "N/A")
    
    with col4:
        # Apply trend detection if enough data
        if len(time_series_data) > 3:
            _, trend_direction, percent_change = detect_trend(time_series_data['Count'])
            trend_icon = "↗️" if trend_direction == "increasing" else "↘️" if trend_direction == "decreasing" else "→"
            st.metric("Frequency Trend", f"{trend_icon} {trend_direction.capitalize()}")
            st.caption(f"{percent_change:.1f}% change over period")
        else:
            st.metric("Frequency Trend", "Insufficient data")
    
    # Create the animated visualization
    st.markdown("---")
    st.subheader("Earthquake Frequency Time Series Animation")
    
    # Create placeholder for chart
    chart_placeholder = st.empty()
    
    # Add explanation
    with st.expander("About this Visualization", expanded=False):
        st.markdown("""
        This animation shows how earthquake frequency changes over time. The visualization includes:
        
        - **Time series line**: Shows the count of earthquakes per time period
        - **Smooth transitions**: Natural movement between data points
        - **Color gradient**: Changes color based on frequency intensity
        - **Area fill**: Highlights the volume under the frequency curve
        
        Use the sidebar controls to adjust the visualization settings and filters.
        """)
    
    # Run the animation when button is clicked
    if st.button("Run Frequency Analysis Animation", use_container_width=True):
        # Create a function to generate color based on count
        def get_count_color(count, max_count):
            # Color changes from blue (low) to red (high)
            ratio = min(1.0, count / max_count) if max_count > 0 else 0
            r = int(55 + ratio * 200)  # More red as count increases
            g = int(100 - ratio * 70)  # Less green as count increases
            b = int(200 - ratio * 170)  # Less blue as count increases
            return f'rgba({r}, {g}, {b}, 0.8)'
        
        # Precompute colors for better performance
        if len(time_series_data) > 0:
            max_count_value = time_series_data['Count'].max()
            time_series_data['color'] = time_series_data['Count'].apply(
                lambda count: get_count_color(count, max_count_value)
            )
            
            # Animation steps - adjust based on performance mode
            steps = len(time_series_data)
            
            # Initialize display with empty plot to avoid redrawing
            if enable_performance_mode:
                fig = go.Figure()
                fig.update_layout(
                    title=f'Earthquake Frequency Over Time ({time_aggregation})',
                    xaxis_title='Time',
                    yaxis_title='Number of Earthquakes',
                    hovermode='x unified',
                    margin=dict(l=20, r=20, t=50, b=20),
                    xaxis=dict(
                        showgrid=True,
                        gridcolor='lightgray',
                        # Set fixed range for entire animation to prevent rescaling
                        range=[time_series_data['DATETIME'].min(), time_series_data['DATETIME'].max()]
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridcolor='lightgray',
                        # Set fixed range to prevent rescaling
                        range=[0, max(max_count_value * 1.1, 1)]
                    )
                )
                # Display the initial empty chart
                chart_placeholder.plotly_chart(fig, use_container_width=True)
            
            # Determine frame skipping for performance optimization
            frame_step = max(1, animation_frame_skipping)
            
            # Animation frames
            for i in range(1, steps + 1, frame_step):
                # Performance optimization: only render final frame if near the end
                if enable_performance_mode and i < steps and i + frame_step > steps:
                    i = steps  # Skip to final frame
                
                # Create display dataframe with increasing number of points
                display_df = time_series_data.iloc[:i]
                
                # Create the animated figure
                fig = go.Figure()
                
                # Optimization: Set fixed layout parameters for consistent view
                line_shape = 'spline' if use_smooth_transitions else 'linear'
                
                # Add area fill if enabled - optimization: only one trace
                if show_area_fill:
                    fig.add_trace(go.Scatter(
                        x=display_df['DATETIME'],
                        y=display_df['Count'],
                        fill='tozeroy',
                        fillcolor='rgba(55, 100, 200, 0.2)',
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                
                # Main trend line - single trace
                fig.add_trace(go.Scatter(
                    x=display_df['DATETIME'],
                    y=display_df['Count'],
                    mode='lines+markers',
                    line=dict(
                        shape=line_shape,
                        smoothing=1.3,
                        width=3,
                        color='rgba(55, 100, 200, 0.8)'
                    ),
                    marker=dict(
                        size=8,
                        color=display_df['color'].tolist(),
                        line=dict(width=1, color='white')
                    ),
                    name='Earthquake Frequency',
                    hovertemplate='<b>%{y} earthquakes</b><br>%{x}<br>'
                ))
                
                # Highlight just the newest point with a larger marker
                # Only add this if we have at least one point
                if i > 0:
                    newest_point = display_df.iloc[-1]
                    fig.add_trace(go.Scatter(
                        x=[newest_point['DATETIME']],
                        y=[newest_point['Count']],
                        mode='markers',
                        marker=dict(
                            size=12,
                            color=newest_point['color'],
                            line=dict(
                                color='white',
                                width=2
                            )
                        ),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                
                # Update layout - consistent settings for smoother transitions
                fig.update_layout(
                    title=f'Earthquake Frequency Over Time ({time_aggregation})',
                    xaxis_title='Time',
                    yaxis_title='Number of Earthquakes',
                    hovermode='x unified',
                    margin=dict(l=20, r=20, t=50, b=20),
                    xaxis=dict(
                        showgrid=True,
                        gridcolor='lightgray',
                        # Set fixed range for entire animation to prevent rescaling
                        range=[time_series_data['DATETIME'].min(), time_series_data['DATETIME'].max()]
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridcolor='lightgray',
                        # Set fixed range to prevent rescaling
                        range=[0, max(max_count_value * 1.1, 1)]
                    ),
                    transition_duration=100  # Smooth transition between updates
                )
                
                # Display the updated chart
                chart_placeholder.plotly_chart(fig, use_container_width=True)
                
                # Control animation speed - non-blocking for better performance
                if i < steps:  # Don't sleep after the final frame
                    # Progressive sleep timing for smoother acceleration
                    sleep_time = 0.1 / animation_speed
                    # Reduce sleep time as animation progresses for smoother effect
                    if enable_performance_mode and i > steps / 2:
                        sleep_time = max(0.01, sleep_time * 0.8)
                    time.sleep(sleep_time)
        else:
            st.warning("No data available for animation. Adjust filters to see results.")
    
    # Static visualization as fallback
    else:
        # Create a static version of the plot
        fig = go.Figure()
        
        if len(time_series_data) > 0:
            # Add area fill if enabled
            if show_area_fill:
                fig.add_trace(go.Scatter(
                    x=time_series_data['DATETIME'],
                    y=time_series_data['Count'],
                    fill='tozeroy',
                    fillcolor='rgba(55, 100, 200, 0.2)',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            # Add the main line
            line_shape = 'spline' if use_smooth_transitions else 'linear'
            fig.add_trace(go.Scatter(
                x=time_series_data['DATETIME'],
                y=time_series_data['Count'],
                mode='lines+markers',
                line=dict(
                    shape=line_shape,
                    smoothing=1.3,
                    width=3,
                    color='rgba(55, 100, 200, 0.8)'
                ),
                marker=dict(
                    size=8,
                    color='rgba(55, 100, 200, 0.8)'
                ),
                name='Earthquake Frequency'
            ))
            
            # Update layout
            fig.update_layout(
                title=f'Earthquake Frequency Over Time ({time_aggregation})',
                xaxis_title='Time',
                yaxis_title='Number of Earthquakes',
                hovermode='x unified',
                margin=dict(l=20, r=20, t=50, b=20),
                xaxis=dict(
                    showgrid=True,
                    gridcolor='lightgray'
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor='lightgray'
                )
            )
            
            # Display the static chart
            chart_placeholder.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available for visualization. Adjust your filters to see results.")
    
    # Download options
    st.markdown("---")
    st.subheader("Download Frequency Data")
    
    # Offer download of frequency time series data
    if len(time_series_data) > 0:
        # Remove the color column before exporting
        if 'color' in time_series_data.columns:
            export_data = time_series_data.drop(columns=['color'])
        else:
            export_data = time_series_data
            
        csv = export_data.to_csv(index=False)
        st.download_button(
            label="Download Frequency Time Series as CSV",
            data=csv,
            file_name=f"earthquake_frequency_{agg_freq}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            help="Download the earthquake frequency time series data as a CSV file"
        )
    else:
        st.info("No data available for download. Adjust your filters to generate time series data.")
    
    # Additional analysis section
    st.markdown("---")
    st.subheader("Frequency Trend Analysis")
    
    if len(time_series_data) > 5:
        # Apply trend detection and analysis
        smoothed_series, trend_direction, percent_change = detect_trend(time_series_data['Count'])
        
        # Display trend analysis
        st.markdown(f"""
        ### Overall Trend: {trend_direction.capitalize()}
        
        The analysis shows an **{trend_direction}** trend in earthquake frequency over the selected time period,
        with a **{percent_change:.1f}%** change from the beginning to the end of the period.
        
        #### Key Observations:
        - The highest frequency period occurred on **{time_series_data.loc[time_series_data['Count'].idxmax(), 'DATETIME'].strftime('%B %d, %Y')}**
        - Average frequency over the time period: **{time_series_data['Count'].mean():.1f}** earthquakes per {time_aggregation.lower()}
        - Variability in frequency (coefficient of variation): **{time_series_data['Count'].std() / time_series_data['Count'].mean():.2f}**
        
        The animated visualization helps identify these patterns by showing the progression of earthquake frequency over time.
        """)
    else:
        st.info("Insufficient data for trend analysis. Adjust filters to include more data points.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <p style='text-align: center; color: #888888;'>
    Earthquake Frequency Analysis Dashboard | Interactive Time Series Visualization<br>
    Use the navigation sidebar to explore different frequency analysis views.
    </p>
    """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.write("Please check that the data file exists and is properly formatted.")
