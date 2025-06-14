import streamlit as st
import pandas as pd
import pydeck as pdk
import numpy as np
import time
from datetime import datetime
import colorsys

st.set_page_config(page_title="Sequential Earthquake Visualization", layout="wide")
st.title("🗺️ Sequential Earthquake Animation with Direction Indicators")

# ------------------------------------------------
# System Overview
# ------------------------------------------------
st.markdown("""
This interactive visualization system provides a dynamic, sequential animation of earthquake events with the following capabilities:

- **Temporal Sequence Analysis**: Visualizes earthquakes in chronological order to reveal patterns and progression
- **Magnitude-Based Representation**: Scales visual elements proportionally to earthquake magnitude
- **Directional Indicators**: Shows movement patterns between sequential events with animated arrows
- **Intensity Categories**: Color-codes earthquakes based on their intensity classification
- **Realistic Seismic Wave Simulation**: Models P-waves and S-waves with physics-based propagation patterns
- **Depth Visualization**: Represents earthquake depth through visual effects and camera angles
- **Interactive Filtering**: Allows selection by province, intensity category, and date
- **Comprehensive Statistics**: Provides detailed information about the earthquake sequence

The system processes earthquake data from CSV files containing location, magnitude, depth, and intensity information to create an immersive, informative visualization experience for seismological analysis.
""")

# ------------------------------------------------
# Data Loading and Preprocessing
# ------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Earthquake_Data.csv")
    # Convert column names to uppercase for consistency
    df.columns = df.columns.str.upper()
    # Parse the DATE & TIME column using the new format: "31 January 2023 - 11:58 PM"
    df['DATETIME'] = pd.to_datetime(df['DATE & TIME'], format="%d %B %Y - %I:%M %p", errors='coerce')
    # Convert numeric columns
    df['LATITUDE'] = pd.to_numeric(df['LATITUDE'], errors='coerce')
    df['LONGITUDE'] = pd.to_numeric(df['LONGITUDE'], errors='coerce')
    df['MAGNITUDE'] = pd.to_numeric(df['MAGNITUDE'], errors='coerce')
    df['DEPTH (KM)'] = pd.to_numeric(df['DEPTH (KM)'], errors='coerce')
    # Ensure CATEGORY is uppercase
    df["CATEGORY"] = df["CATEGORY"].str.upper()
    
    # Handle potential NaN values in string columns
    string_columns = ['PROVINCE', 'AREA', 'CATEGORY', 'LOCATION']
    for col in string_columns:
        if col in df.columns:
            # Replace NaN with "Unknown" and ensure all values are strings
            df[col] = df[col].fillna("Unknown").astype(str)
    
    return df

try:
    df = load_data()
    
    # Display data info for debugging
    with st.expander("Data Information", expanded=False):
        st.write(f"Total records: {len(df)}")
        st.write(f"Columns: {df.columns.tolist()}")
        st.write("Sample data:")
        st.dataframe(df.head(3))

    required_columns = {'LATITUDE', 'LONGITUDE', 'MAGNITUDE', 'CATEGORY', 'AREA', 'PROVINCE', 'DEPTH (KM)', 'DATE & TIME', 'DATETIME'}
    if not required_columns.issubset(df.columns):
        missing_cols = required_columns - set(df.columns)
        st.error(f"CSV file does not contain all required columns. Missing: {', '.join(missing_cols)}")
        st.stop()

    # ------------------------------------------------
    # Enhanced Color Mapping with Intensity Levels
    # ------------------------------------------------
    # Define a function to create a gradient of colors for each intensity level
    def create_color_gradient(base_color, num_steps=5):
        """Create a gradient of colors from the base color to a lighter version"""
        r, g, b, a = base_color
        h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
        colors = []
        for i in range(num_steps):
            # Decrease saturation and increase value for a "fading" effect
            new_s = max(0, s - (i * 0.15))
            new_v = min(1, v + (i * 0.15))
            new_r, new_g, new_b = colorsys.hsv_to_rgb(h, new_s, new_v)
            new_a = max(0, a - (i * 40))  # Gradually decrease opacity
            colors.append((int(new_r*255), int(new_g*255), int(new_b*255), int(new_a)))
        return colors
    
    intensity_base_colors = {
        "SCARCELY PERCEPTIBLE": (255, 255, 255, 200),    # White
        "SLIGHTLY FELT": (223, 230, 254, 200),           # GreenYellow
        "WEAK": (130, 249, 251, 200),                    # Yellow
        "MODERATELY STRONG": (130, 250, 224, 200),       # Gold
        "STRONG": (152, 247, 130, 200),                  # Orange
        "VERY STRONG": (247, 246, 80, 200),              # DarkOrange
        "DESTRUCTIVE": (252, 199, 67, 200),              # OrangeRed
        "VERY DESTRUCTIVE": (252, 109, 44, 200),         # Red
        "DEVASTATING": (232, 37, 29, 200),               # DarkRed
        "COMPLETELY DEVASTATING": (196, 31, 24, 200),    # Maroon
        "UNKNOWN": (128, 128, 128, 200),                 # Gray
    }
    
    # Create gradients for each intensity category
    intensity_color_gradients = {category: create_color_gradient(color) 
                              for category, color in intensity_base_colors.items()}
    
    # Safely map base colors with a default for any unmapped categories
    df["COLOR"] = df["CATEGORY"].apply(lambda x: intensity_base_colors.get(x, (128, 128, 128, 200)))

    # ------------------------------------------------
    # Sidebar Filtering with "ALL" Options
    # ------------------------------------------------
    st.sidebar.header("Filters")

    # Province Filter: add "ALL" option
    # Fix: Convert all values to strings and handle NaN values before sorting
    all_provinces = sorted([p for p in df['PROVINCE'].unique() if p != "Unknown"])
    if "Unknown" in df['PROVINCE'].unique():
        all_provinces.append("Unknown")  # Add Unknown at the end if it exists
        
    selected_provinces = st.sidebar.multiselect("Select Province(s)", options=["ALL"] + all_provinces, default=["ALL"])
    if "ALL" in selected_provinces:
        selected_provinces = all_provinces

    # Intensity Category Filter: add "ALL" option
    all_categories = sorted([c for c in df["CATEGORY"].unique() if c != "Unknown"])
    if "Unknown" in df["CATEGORY"].unique():
        all_categories.append("Unknown")  # Add Unknown at the end if it exists
        
    selected_categories = st.sidebar.multiselect("Select Intensity Categories", options=["ALL"] + all_categories, default=["ALL"])
    if "ALL" in selected_categories:
        selected_categories = all_categories

    # Filter by date only (ignoring time)
    # Find the earliest and latest dates in the data
    valid_dates = df['DATETIME'].dropna()
    if len(valid_dates) > 0:
        min_date = valid_dates.dt.date.min()
        max_date = valid_dates.dt.date.max()
        default_date = max_date  # Default to the most recent date
    else:
        min_date = datetime.now().date()
        max_date = datetime.now().date()
        default_date = datetime.now().date()
        
    selected_date = st.sidebar.date_input("Select Date", value=default_date, min_value=min_date, max_value=max_date)

    # Apply filters
    filtered_df = df[
        df['PROVINCE'].isin(selected_provinces) &
        df["CATEGORY"].isin(selected_categories)
    ]
    
    # Safely filter by date
    filtered_df = filtered_df[filtered_df['DATETIME'].dt.date == selected_date]

    if filtered_df.empty:
        st.warning("No earthquake data for the selected filters.")
        st.stop()

    # Sort events in chronological order
    sorted_quakes = filtered_df.sort_values('DATETIME').reset_index(drop=True)
    
    # ------------------------------------------------
    # Enhanced Sequence Information
    # ------------------------------------------------
    st.sidebar.markdown("---")
    st.sidebar.subheader("Animation Sequence")
    
    # Show detailed sequence statistics
    total_events = len(sorted_quakes)
    st.sidebar.write(f"**Total Events:** {total_events} earthquakes")
    
    if total_events > 0:
        st.sidebar.write(f"**Magnitude Range:** {sorted_quakes['MAGNITUDE'].min():.1f} - {sorted_quakes['MAGNITUDE'].max():.1f}")
        st.sidebar.write(f"**Depth Range:** {sorted_quakes['DEPTH (KM)'].min():.1f} - {sorted_quakes['DEPTH (KM)'].max():.1f} km")
        st.sidebar.write(f"**Time Span:** {sorted_quakes['DATETIME'].min().strftime('%H:%M')} - {sorted_quakes['DATETIME'].max().strftime('%H:%M')}")
        
        # Show sequence preview
        with st.sidebar.expander("Sequence Preview", expanded=False):
            st.write("**Events in animation sequence:**")
            
            # Create a readable preview of events to be animated
            preview_data = sorted_quakes.head(min(10, len(sorted_quakes)))
            
            # Format the preview table
            preview_rows = []
            for i, row in preview_data.iterrows():
                # Truncate location for display clarity
                location = row['LOCATION']
                location_display = location[:15] + "..." if len(location) > 15 else location
                
                preview_rows.append({
                    '#': i + 1,
                    'M': f"{row['MAGNITUDE']:.1f}",
                    'Depth': f"{row['DEPTH (KM)']:.0f}km",
                    'Location': location_display
                })
            
            # Display the preview table
            st.table(pd.DataFrame(preview_rows))
            
            if len(sorted_quakes) > 10:
                st.caption(f"... and {len(sorted_quakes) - 10} more events will be animated in sequence")
                
            # Show estimated animation duration
            total_duration = len(sorted_quakes) * 5.0  # 5 seconds per event
            minutes = int(total_duration // 60)
            seconds = int(total_duration % 60)
            
            if minutes > 0:
                st.write(f"**Total Animation Time:** {minutes}m {seconds}s")
            else:
                st.write(f"**Total Animation Time:** {seconds}s")

    # ------------------------------------------------
    # Enhanced Animation Settings
    # ------------------------------------------------
    st.sidebar.subheader("Advanced Animation Settings")
    
    # Shockwave Parameters
    st.sidebar.markdown("#### Shockwave Parameters")
    base_radius = st.sidebar.slider("Base Radius Multiplier", 1000, 15000, 5000, step=100)
    max_ripples = st.sidebar.slider("Number of Ripple Rings", 1, 8, 5, step=1)
    shockwave_speed = st.sidebar.slider("Shockwave Speed", 0.5, 10.0, 3.0, step=0.1)
    
    # Visual Effects
    st.sidebar.markdown("#### Visual Effects")
    pulse_amplitude = st.sidebar.slider("Pulse Amplitude", 0.1, 1.0, 0.5, step=0.05)
    pulse_frequency = st.sidebar.slider("Pulse Frequency", 0.05, 1.0, 0.2, step=0.05)
    motion_blur = st.sidebar.slider("Motion Blur Effect", 0.0, 1.0, 0.5, step=0.05)
    
    # Epicenter Effects
    st.sidebar.markdown("#### Epicenter Effects")
    epicenter_glow = st.sidebar.slider("Epicenter Glow Intensity", 0.1, 2.0, 1.0, step=0.1)
    initial_burst_size = st.sidebar.slider("Initial Burst Size", 0.5, 5.0, 2.0, step=0.1)
    use_burst_effect = st.sidebar.checkbox("Use Burst Effect (Instead of Needle)", False)
    burst_particles = st.sidebar.slider("Burst Particles", 0, 20, 0, step=1)
    burst_intensity = st.sidebar.slider("Burst Intensity", 0.1, 2.0, 1.0, step=0.1)
    
    # Arrow Settings for Direction Indicators
    st.sidebar.markdown("#### Direction Indicators")
    arrow_size = st.sidebar.slider("Arrow Size", 0.1, 3.0, 1.0, step=0.1)
    arrow_color = st.sidebar.color_picker("Arrow Color", "#FF5733")
    show_arrow_path = st.sidebar.checkbox("Show Complete Path", True)
    
    # Timing
    st.sidebar.markdown("#### Timing")
    event_duration = 5  # Fixed at 5 seconds per earthquake
    st.sidebar.write("Duration per Event: 5 seconds (fixed)")
    transition_speed = st.sidebar.slider("Transition Speed", 0.1, 2.0, 0.8, step=0.1)
    
    # Animation physics
    show_depth_effect = st.sidebar.checkbox("Show Depth Effect", True)
    enable_terrain_interaction = st.sidebar.checkbox("Enable Terrain Interaction", True)
    
    # Camera Settings
    camera_options = ["Top-down", "Tilted View", "Dynamic Camera"]
    selected_camera = st.sidebar.selectbox("Camera Angle", camera_options, index=1)
    
    # ------------------------------------------------
    # Sequential Earthquake Animation Setup
    # ------------------------------------------------
    map_container = st.empty()

    st.info("Click 'Start Animation' to visualize earthquakes sequentially with directional indicators. Each earthquake will be shown for 5 seconds, with an arrow pointing to the next event.")

    # Display a summary of events for the selected date
    with st.expander("Events on Selected Date", expanded=True):
        st.subheader(f"Events on {selected_date}")
        summary_cols = ["DATETIME", "AREA", "PROVINCE", "MAGNITUDE", "CATEGORY", "DEPTH (KM)"]
        st.dataframe(sorted_quakes[summary_cols], use_container_width=True)

    # Enhanced animation controls with better layout
    st.markdown("### Animation Controls")
    col1, col2, col3 = st.columns([2, 2, 3])
    
    with col1:
        start_animation = st.button("▶️ Start Animation", type="primary", use_container_width=True, 
                               help="Start the sequential animation from the beginning")
    
    with col2:
        stop_animation = st.button("⏹️ Stop Animation", use_container_width=True,
                             help="Stop the current animation")
                             
    with col3:
        # Display animation sequence information
        if len(sorted_quakes) > 0:
            total_duration = len(sorted_quakes) * event_duration
            minutes = int(total_duration // 60)
            seconds = int(total_duration % 60)
            time_text = f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"
            st.info(f"Sequence: {len(sorted_quakes)} events • Total duration: {time_text}")

    # Initialize animation state
    if "animation_running" not in st.session_state:
        st.session_state.animation_running = False
    
    if "current_quake_index" not in st.session_state:
        st.session_state.current_quake_index = 0

    # Update animation state based on button clicks
    if start_animation:
        st.session_state.animation_running = True
        st.session_state.current_quake_index = 0
        st.toast("Starting sequential animation from the beginning")
    
    if stop_animation:
        st.session_state.animation_running = False
        st.toast("Animation stopped")

    # Display current earthquake info
    if len(sorted_quakes) > 0:
        current_idx = min(st.session_state.current_quake_index, len(sorted_quakes) - 1)
        current_quake = sorted_quakes.iloc[current_idx]
        
        st.sidebar.markdown("#### Current Earthquake")
        st.sidebar.markdown(f"**Time:** {current_quake['DATETIME']}")
        st.sidebar.markdown(f"**Location:** {current_quake['LOCATION']}")
        st.sidebar.markdown(f"**Magnitude:** {current_quake['MAGNITUDE']:.1f}")
        st.sidebar.markdown(f"**Category:** {current_quake['CATEGORY']}")
        
        # Progress indicator
        st.sidebar.progress((current_idx + 1) / len(sorted_quakes))
        st.sidebar.text(f"Event {current_idx + 1} of {len(sorted_quakes)}")

    # ------------------------------------------------
    # Advanced Seismic Animation Functions
    # ------------------------------------------------
    def calculate_shockwave_parameters(magnitude, depth):
        """Calculate parameters for the shockwave based on earthquake properties"""
        # Base intensity affects the strength of the visual effect
        # More exponential scaling for magnitude to better distinguish varying intensities
        base_intensity = np.clip(magnitude**1.5 / 30, 0.1, 1.0)
        
        # Depth affects how the waves propagate (deeper = slower, more spread out)
        # Refined depth factor calculation to better simulate wave propagation physics
        depth_factor = 1.0 - min(depth, 200) / 250  # Normalize depth effect with wider range
        
        # Calculate wave speed based on magnitude and depth using empirical model
        # Incorporates research on seismic wave velocity relations to quake properties
        # P-waves typically travel 1.7 times faster than S-waves
        p_wave_factor = 1.7 * (0.6 + base_intensity * 0.4)
        s_wave_factor = 1.0 * (0.7 + base_intensity * 0.3)
        
        # Combined wave speed with depth-dependent attenuation
        wave_speed = shockwave_speed * (0.6 + base_intensity) * (0.3 + depth_factor * 0.7) * p_wave_factor
        
        # Secondary wave speed (slower, follows primary)
        s_wave_speed = wave_speed / p_wave_factor * s_wave_factor
        
        # Number of visible ripples proportional to magnitude with physics-based scaling
        # Larger quakes generate more distinct wave patterns
        ripple_count = min(max(int(magnitude * 1.2), 2), max_ripples)
        
        # Initial burst size is larger for bigger earthquakes with improved scaling
        burst_multiplier = initial_burst_size * (0.4 + base_intensity * 1.8)
        
        # Particles count scaled by magnitude^2 for more dramatic visual at higher magnitudes
        particle_count = min(int(magnitude**2), burst_particles)
        
        # Directional bias factor - stronger for shallow quakes
        # This simulates how shallow earthquakes tend to have more directional energy radiation
        directional_bias = 1.0 - min(depth, 100) / 150  # 0 to 1 scale (deeper = less directional)
        
        # Fault orientation simulation - determines primary direction of energy release
        # In real earthquakes, energy propagates more strongly along fault lines
        # Use a deterministic method based on quake data to ensure consistent visualization
        # This creates the appearance of consistent fault line orientations
        import hashlib
        # Use just magnitude and depth since lat/lon aren't available here
        hash_input = f"{magnitude:.2f}_{depth:.2f}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        fault_angle = (hash_value % 360) * (np.pi / 180)  # Consistent angle based on quake properties
        
        # Wave attenuation parameters based on magnitude and depth
        # Controls how quickly waves fade with distance
        if magnitude >= 7.0:
            # Major earthquakes have waves that travel further with less attenuation
            attenuation_factor = 0.6
        elif magnitude >= 5.0:
            # Moderate earthquakes
            attenuation_factor = 0.75
        else:
            # Minor earthquakes attenuate more quickly
            attenuation_factor = 0.85
            
        # Depth also affects attenuation - deeper quakes attenuate more gradually
        attenuation_factor *= (0.8 + depth_factor * 0.2)
        
        return {
            "intensity": base_intensity,
            "depth_factor": depth_factor,
            "wave_speed": wave_speed,
            "s_wave_speed": s_wave_speed,
            "ripple_count": ripple_count,
            "burst_multiplier": burst_multiplier,
            "particle_count": particle_count,
            "directional_bias": directional_bias,
            "fault_angle": fault_angle,
            "attenuation_factor": attenuation_factor
        }
    
    def generate_ripple_layers(quake_data, animation_time, params):
        """Generate multiple ripple layers with realistic wave physics and directional indicators"""
        layers = []
        
        # Extract earthquake properties
        lat = quake_data['LATITUDE']
        lon = quake_data['LONGITUDE']
        magnitude = quake_data['MAGNITUDE']
        depth = quake_data['DEPTH (KM)']
        category = quake_data['CATEGORY']
        
        # Get base color for this earthquake's intensity category
        base_color = intensity_base_colors.get(category, (128, 128, 128, 200))
        color_gradient = intensity_color_gradients.get(category, create_color_gradient((128, 128, 128, 200)))
        
        # Calculate time-dependent factors
        animation_progress = min(animation_time / event_duration, 1.0)
        
        # Generate directional ripple effect parameters
        # Use depth to determine ripple directionality - deeper = more symmetrical
        depth_symmetry = max(0.2, min(1.0, depth / 100))
        # Magnitude influences ripple intensity
        ripple_intensity = magnitude / 10.0
        
        # ------------------------------------------------
        # Initial Burst Effect (if enabled)
        # ------------------------------------------------
        if use_burst_effect:
            burst_phase = min(animation_time * 2, 1.0)  # Quick phase in
            burst_opacity = max(0, 1 - (animation_time / (event_duration * 0.3)))  # Quick fade out
            
            if burst_opacity > 0.05:  # Only render if still visible
                burst_layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=[{
                        "position": [lon, lat],
                        "radius": magnitude * 2500 * params["burst_multiplier"] * burst_phase,
                        "color": [base_color[0], base_color[1], base_color[2], int(base_color[3] * burst_opacity)]
                    }],
                    get_position="position",
                    get_radius="radius",
                    get_fill_color="color",
                    pickable=False,
                    opacity=0.6,
                    stroked=False,
                    filled=True,
                )
                layers.append(burst_layer)
        
        # ------------------------------------------------
        # Column/Needle Effect (if not using burst)
        # ------------------------------------------------
        if not use_burst_effect:
            column_height = depth * 500  # Scale based on depth
            column_opacity = 0.7
            
            column_layer = pdk.Layer(
                "ColumnLayer",
                data=[{
                    "position": [lon, lat],
                    "elevation": column_height,
                    "color": [base_color[0], base_color[1], base_color[2], int(base_color[3] * column_opacity)]
                }],
                get_position="position",
                get_elevation="elevation",
                get_fill_color="color",
                get_radius=magnitude * 2000,
                pickable=False,
                opacity=0.7,
                stroked=False,
                filled=True,
                extruded=True,
            )
            layers.append(column_layer)
        
        # ------------------------------------------------
        # Epicenter Marker
        # ------------------------------------------------
        epicenter_layer = pdk.Layer(
            "ScatterplotLayer",
            data=[{
                "position": [lon, lat],
                "radius": magnitude * 2000 * (1.0 + pulse_amplitude * np.sin(animation_time * pulse_frequency * 10)),
                "color": [base_color[0], base_color[1], base_color[2], 180]
            }],
            get_position="position",
            get_radius="radius",
            get_fill_color="color",
            pickable=False,
            opacity=0.8,
            stroked=True,
            filled=True,
        )
        layers.append(epicenter_layer)
        
        # ------------------------------------------------
        # Advanced Ripple Rings with Directional Indicators
        # ------------------------------------------------
        # Create multiple ripple rings that expand outward with realistic physics
        max_radius = base_radius * magnitude * params["wave_speed"]
        
        for i in range(params["ripple_count"]):
            ring_delay = i * (1.0 / params["ripple_count"])  # Stagger start times
            ring_progress = max(0, animation_progress - ring_delay)
            
            if ring_progress > 0:
                # Calculate the expanding radius with physics-based damping
                # Deeper earthquakes have different wave propagation characteristics
                damping_factor = 1.0 - depth_symmetry * 0.5
                wave_speed_modifier = 1.0 + 0.2 * np.sin(ring_progress * np.pi * 2)  # Realistic wave speed variations
                
                # Physics-based radius calculation with non-linear propagation
                # Incorporate attenuation factor for more realistic wave physics
                # P-waves and S-waves travel at different speeds
                p_wave_component = 0.7  # Primary wave contribution
                s_wave_component = 0.3  # Secondary wave contribution (follows primary)
                
                # P-wave radius calculation (faster wave)
                p_wave_radius = max_radius * ring_progress * wave_speed_modifier
                
                # S-wave radius calculation (slower wave)
                # S-waves travel at approx 60-70% of P-wave speed
                s_wave_delay = 0.2  # S-waves lag behind P-waves
                s_wave_progress = max(0, ring_progress - s_wave_delay)
                s_wave_radius = max_radius * s_wave_progress * wave_speed_modifier * 0.65
                
                # Apply realistic attenuation based on magnitude, depth, and distance
                # Waves attenuate (lose energy) as they travel - this creates natural fading
                # Formula based on simplified seismic attenuation models
                p_wave_attenuation = np.exp(-ring_progress * params["attenuation_factor"])
                s_wave_attenuation = np.exp(-s_wave_progress * params["attenuation_factor"] * 0.8)  # S-waves attenuate less
                
                # Combine P and S wave components for final ring radius
                # This creates a more complex, realistic wave pattern
                if s_wave_progress > 0:
                    ring_radius = (p_wave_radius * p_wave_component * p_wave_attenuation + 
                                  s_wave_radius * s_wave_component * s_wave_attenuation)
                else:
                    ring_radius = p_wave_radius * p_wave_attenuation
                
                # Add non-linear damping based on depth factor
                ring_radius *= (1.0 - ring_progress * damping_factor * 0.2)
                
                # Fade opacity using realistic attenuation model based on depth and distance
                # Combine attenuation from both wave types
                if s_wave_progress > 0:
                    ring_opacity = max(0, 0.8 * (p_wave_attenuation * p_wave_component + 
                                                s_wave_attenuation * s_wave_component))
                else:
                    ring_opacity = max(0, 0.8 * p_wave_attenuation)
                
                # Get the color for this ring (fading as they expand)
                color_idx = min(int(ring_progress * len(color_gradient)), len(color_gradient) - 1)
                ring_color = color_gradient[color_idx]
                
                # For directional indicators, calculate points around the circle with varying intensity/size
                directional_points = []
                num_points = 24  # Number of points around the circle
                
                # Create direction-based ripple distortion
                for j in range(num_points):
                    angle = j * (2 * np.pi / num_points)
                    
                    # The actual directional effect varies based on directional_bias parameter
                    # Deep earthquakes: more symmetrical waves (lower directional_bias)
                    # Shallow earthquakes: more directional waves (higher directional_bias)
                    
                    # Create directional bias - stronger in some directions than others
                    # This mimics fault plane orientation and energy radiation patterns
                    # Calculate angle relative to the simulated fault orientation
                    relative_angle = angle - params["fault_angle"]
                    
                    # Apply directional bias using empirical seismic energy radiation pattern
                    # This creates a realistic radiation pattern based on fault mechanics
                    direction_factor = 1.0 - params["directional_bias"] * 0.7 * (
                        0.5 + 0.5 * np.cos(2 * relative_angle)
                    )
                    
                    # Calculate position with directional bias
                    # Add subtle wave-like variations to simulate complex seismic interference patterns
                    oscillation = 0.9 + 0.2 * np.sin(angle * 4 + animation_time * 4)
                    
                    # Scale radius by direction factor for directional effect
                    point_radius = ring_radius * direction_factor * oscillation
                    
                    # Convert radius (meters) to geographic offsets (degrees)
                    lat_offset = (point_radius / 111000) * np.sin(angle)  # Convert meters to degrees
                    lon_offset = (point_radius / (111000 * np.cos(np.radians(lat)))) * np.cos(angle)
                    
                    # Point intensity varies with direction to create visual directional indicators
                    # Increase contrast between strong and weak directions based on magnitude
                    mag_contrast = min(0.7, magnitude / 10)  # Higher magnitudes = more contrast in directional indicators
                    point_intensity = max(0.3, direction_factor ** (1.0 - mag_contrast)) * ring_opacity
                    
                    # Magnitude influences the size and prominence of directional indicators
                    # Higher magnitudes make the directional effects more prominent
                    magnitude_factor = np.clip(magnitude / 5.5, 0.5, 2.0)**1.5
                    
                    # Calculate adaptive point size based on magnitude, direction and distance from epicenter
                    # This makes the arrows proportionate to earthquake magnitude as required
                    point_size = magnitude * 900 * magnitude_factor * max(0.1, point_intensity) * ripple_intensity
                    
                    directional_points.append({
                        "position": [lon + lon_offset, lat + lat_offset],
                        "radius": point_size,
                        "color": [
                            ring_color[0], 
                            ring_color[1], 
                            ring_color[2], 
                            int(ring_color[3] * point_intensity)
                        ]
                    })
                
                # Main ripple outline
                ripple_layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=[{
                        "position": [lon, lat],
                        "radius": ring_radius,
                        "color": [
                            ring_color[0], 
                            ring_color[1], 
                            ring_color[2], 
                            int(ring_color[3] * ring_opacity * 0.5)  # Reduced opacity for base ring
                        ]
                    }],
                    get_position="position",
                    get_radius="radius",
                    get_fill_color="color",
                    pickable=False,
                    opacity=ring_opacity * 0.6,  # Slightly transparent
                    stroked=True,
                    filled=False,
                    line_width_min_pixels=2,
                )
                layers.append(ripple_layer)
                
                # Directional indicator points
                directional_layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=directional_points,
                    get_position="position",
                    get_radius="radius",
                    get_fill_color="color",
                    pickable=False,
                    opacity=ring_opacity,
                    stroked=False,
                    filled=True,
                )
                layers.append(directional_layer)
        
        # ------------------------------------------------
        # Burst Particles
        # ------------------------------------------------
        # Only add particles if enabled and in the initial phase of animation
        if burst_particles > 0 and animation_time < (event_duration * 0.3):
            particle_data = []
            
            # Scale the number of particles based on magnitude and settings
            num_particles = int(params["particle_count"] * burst_intensity)
            
            # Create particles with random offsets
            for _ in range(num_particles):
                # Random angle and distance from epicenter
                angle = np.random.random() * 2 * np.pi
                distance = np.random.random() * max_radius * animation_progress * 0.5
                
                # Calculate lat/lon offset
                lat_offset = (distance / 111000) * np.sin(angle)
                lon_offset = (distance / (111000 * np.cos(np.radians(lat)))) * np.cos(angle)
                
                # Particle properties
                particle_size = np.random.random() * 2000 * magnitude / 5
                particle_opacity = max(0, 1 - (animation_time / (event_duration * 0.3)))
                
                particle_data.append({
                    "position": [lon + lon_offset, lat + lat_offset],
                    "radius": particle_size,
                    "color": [255, 255, 200, int(255 * particle_opacity)]
                })
            
            if particle_data:
                particles_layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=particle_data,
                    get_position="position",
                    get_radius="radius",
                    get_fill_color="color",
                    pickable=False,
                    opacity=0.7,
                    stroked=False,
                    filled=True,
                )
                layers.append(particles_layer)
        
        return layers
    
    def create_arrow_layer(current_quake, next_quake, animation_progress):
        """Create an arrow layer pointing from current to next earthquake"""
        if current_quake is None or next_quake is None:
            return None
            
        start_lat = current_quake['LATITUDE']
        start_lon = current_quake['LONGITUDE']
        end_lat = next_quake['LATITUDE']
        end_lon = next_quake['LONGITUDE']
        
        # Convert hex color to RGB
        arrow_rgb = [int(arrow_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)]
        
        # Calculate path points for a curved trajectory between earthquakes
        path_points = []
        
        # Create a curved path with control points for more appealing visuals
        num_points = 50  # Higher number for smoother animation
        
        # Calculate midpoint for curve control
        mid_lon = (start_lon + end_lon) / 2
        mid_lat = (start_lat + end_lat) / 2
        
        # Calculate perpendicular offset for curve control point
        dx = end_lon - start_lon
        dy = end_lat - start_lat
        distance = np.sqrt(dx**2 + dy**2)
        
        # Longer distance = more curve
        curve_factor = min(0.3, distance * 0.1)
        
        # Calculate perpendicular offset direction
        perpendicular_x = -dy
        perpendicular_y = dx
        
        # Normalize and apply curve factor
        norm = np.sqrt(perpendicular_x**2 + perpendicular_y**2)
        if norm > 0:
            perpendicular_x = perpendicular_x / norm * distance * curve_factor
            perpendicular_y = perpendicular_y / norm * distance * curve_factor
        
        # Create control point off the direct path for curved trajectory
        control_lon = mid_lon + perpendicular_x
        control_lat = mid_lat + perpendicular_y
        
        # Generate curved path using quadratic Bezier curve
        for i in range(num_points + 1):
            t = i / num_points
            # Quadratic Bezier curve formula
            # B(t) = (1-t)²P₀ + 2(1-t)tP₁ + t²P₂
            # where P₀ is start, P₁ is control, P₂ is end
            t_inv = 1.0 - t
            point_lon = t_inv**2 * start_lon + 2 * t_inv * t * control_lon + t**2 * end_lon
            point_lat = t_inv**2 * start_lat + 2 * t_inv * t * control_lat + t**2 * end_lat
            path_points.append([point_lon, point_lat])
        
        # Show the arrow later in the animation cycle as specified in requirements
        # Arrow appears at 80% of animation and completes quickly
        if animation_progress < 0.8:
            # No arrow until 80% of animation is complete
            return None
        
        # Normalize progress to 0-1 range for the 80%-100% section
        arrow_progress = min(1.0, (animation_progress - 0.8) / 0.2)
        
        # Calculate visible portion of the path based on progress
        visible_points = int(arrow_progress * len(path_points))
        visible_path = path_points[:visible_points]
        
        if len(visible_path) < 2:
            return None
        
        # Enhance pulsing effect with composite sine waves for more organic feel
        base_pulse = 0.7 + 0.3 * np.sin(animation_progress * 10)
        secondary_pulse = 0.9 + 0.1 * np.sin(animation_progress * 7 + 1)
        pulse_effect = base_pulse * secondary_pulse
        
        # Color modifier to shift hue slightly during animation
        pulse_hue = np.sin(animation_progress * 3) * 20  # -20 to +20 range
        
        # Adjust RGB color based on pulse hue (simple approximation)
        r_mod = max(0, min(255, arrow_rgb[0] + pulse_hue * 0.5))
        g_mod = max(0, min(255, arrow_rgb[1] - pulse_hue * 0.3))
        b_mod = max(0, min(255, arrow_rgb[2] + pulse_hue * 0.7))
        
        # Get magnitude from the current earthquake to scale arrow properties
        magnitude = current_quake['MAGNITUDE']
        
        # Scale arrow width proportionally to earthquake magnitude but keep modest size
        # Per requirements: "Keep the arrow size modest so it doesn't dominate the visualization"
        magnitude_scale = np.clip(magnitude / 6.0, 0.6, 1.8)  # Reduced scaling range
        arrow_width = 6 * arrow_size * pulse_effect * magnitude_scale  # Smaller base width
        
        # Create the path data with enhanced color
        path_data = [{
            "path": visible_path,
            "color": [int(r_mod), int(g_mod), int(b_mod), int(230 * pulse_effect)]  # RGB + alpha with pulsing
        }]
        
        # Create enhanced line layer for the path with glow effect
        path_layer = pdk.Layer(
            "PathLayer",
            data=path_data,
            get_path="path",
            get_color="color",
            get_width=arrow_width,  # Width scaled by magnitude
            pickable=False,
            width_scale=20,
            width_min_pixels=4,
            cap_rounded=True,
            joint_rounded=True,
        )
        
        # Add glow effect by adding a wider, more transparent path underneath
        # Scale the glow effect with magnitude as well for consistency
        glow_width = 15 * arrow_size * pulse_effect * magnitude_scale
        
        glow_layer = pdk.Layer(
            "PathLayer",
            data=[{
                "path": visible_path,
                "color": [int(r_mod), int(g_mod), int(b_mod), int(80 * pulse_effect)]  # More transparent
            }],
            get_path="path",
            get_color="color",
            get_width=glow_width,  # Wider than main path, scaled by magnitude
            pickable=False,
            width_scale=20,
            width_min_pixels=7,
            cap_rounded=True,
            joint_rounded=True,
        )
        
        # Create the arrow head (visible as soon as path is visible)
        arrow_layers = [glow_layer, path_layer]  # Order matters for visual layering
        
        # Add arrow head if path is sufficiently visible
        if visible_points > 3:  # Even with minimal path, show the arrow
            # Calculate the last two points to determine direction
            if len(visible_path) >= 2:
                p1 = visible_path[-2]
                p2 = visible_path[-1]
                
                # Calculate direction angle
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                angle = np.arctan2(dy, dx)
                
                # Scale arrow head proportionally to earthquake magnitude but keep modest size
                # Per requirements: "Keep the arrow size modest so it doesn't dominate the visualization"
                magnitude_scale = np.clip(magnitude / 6.0, 0.6, 1.8)  # Reduced scaling range - same as path width
                arrow_size_meters = 4500 * arrow_size * (0.8 + 0.4 * pulse_effect) * magnitude_scale  # Smaller base size
                
                # Add slight rotation animation to arrow head
                rotation_wobble = np.sin(animation_progress * 8) * 10  # -10 to +10 degrees
                
                # Create enhanced arrow head layer using ScatterplotLayer instead of IconLayer
                # This avoids the "Icon is missing" error with Deck.gl
                angle_rad = np.radians(np.degrees(angle) + rotation_wobble)
                
                # Generate triangle points for arrow head
                triangle_points = []
                # Calculate three points forming a triangle for the arrow head
                # Calculate the triangle vertices based on direction angle
                head_size = arrow_size_meters / 6000  # Scale down for geographic coordinates
                
                # Calculate the point at the front of the arrow (tip)
                tip_lon = p2[0]
                tip_lat = p2[1]
                
                # Calculate the two points at the back of the arrow head (left and right corners)
                back_offset = head_size * 0.8
                side_offset = head_size * 0.5
                
                # Back center point
                back_center_lon = tip_lon - np.cos(angle_rad) * back_offset
                back_center_lat = tip_lat - np.sin(angle_rad) * back_offset
                
                # Left point of the arrow head
                left_lon = back_center_lon + np.cos(angle_rad + np.pi/2) * side_offset
                left_lat = back_center_lat + np.sin(angle_rad + np.pi/2) * side_offset
                
                # Right point of the arrow head
                right_lon = back_center_lon + np.cos(angle_rad - np.pi/2) * side_offset
                right_lat = back_center_lat + np.sin(angle_rad - np.pi/2) * side_offset
                
                # Create polygon data for the arrow head
                arrow_head_polygon = [{
                    "contour": [
                        [tip_lon, tip_lat],
                        [left_lon, left_lat],
                        [right_lon, right_lat]
                    ],
                    "color": [int(r_mod), int(g_mod), int(b_mod), int(250 * pulse_effect)]
                }]
                
                # Use PolygonLayer for the arrow head
                arrow_head_layer = pdk.Layer(
                    "PolygonLayer",
                    data=arrow_head_polygon,
                    get_polygon="contour",
                    get_fill_color="color",
                    get_line_color=[255, 255, 255, 100],
                    get_line_width=2,
                    pickable=False,
                    stroked=True,
                    filled=True,
                    extruded=False,
                    auto_highlight=False,
                )
                arrow_layers.append(arrow_head_layer)
                
                # Add glow effect for arrow head using PolygonLayer
                # Create a slightly larger polygon for the glow effect
                glow_scale = 1.5
                
                # Calculate the point at the front of the arrow (tip) - same as main triangle
                glow_tip_lon = tip_lon
                glow_tip_lat = tip_lat
                
                # Calculate the two points at the back of the arrow head with larger sizes
                glow_back_offset = head_size * 0.8 * glow_scale
                glow_side_offset = head_size * 0.5 * glow_scale
                
                # Back center point
                glow_back_center_lon = tip_lon - np.cos(angle_rad) * glow_back_offset
                glow_back_center_lat = tip_lat - np.sin(angle_rad) * glow_back_offset
                
                # Left point of the arrow head
                glow_left_lon = glow_back_center_lon + np.cos(angle_rad + np.pi/2) * glow_side_offset
                glow_left_lat = glow_back_center_lat + np.sin(angle_rad + np.pi/2) * glow_side_offset
                
                # Right point of the arrow head
                glow_right_lon = glow_back_center_lon + np.cos(angle_rad - np.pi/2) * glow_side_offset
                glow_right_lat = glow_back_center_lat + np.sin(angle_rad - np.pi/2) * glow_side_offset
                
                # Create polygon data for the arrow head glow
                arrow_glow_polygon = [{
                    "contour": [
                        [glow_tip_lon, glow_tip_lat],
                        [glow_left_lon, glow_left_lat],
                        [glow_right_lon, glow_right_lat]
                    ],
                    "color": [int(r_mod), int(g_mod), int(b_mod), int(100 * pulse_effect)]
                }]
                
                # Use PolygonLayer for the arrow head glow
                arrow_glow_layer = pdk.Layer(
                    "PolygonLayer",
                    data=arrow_glow_polygon,
                    get_polygon="contour",
                    get_fill_color="color",
                    get_line_color=[255, 255, 255, 60],
                    get_line_width=0,
                    pickable=False,
                    stroked=False,
                    filled=True,
                    extruded=False,
                    auto_highlight=False,
                )
                # Add glow effect beneath the main arrow head
                arrow_layers.insert(1, arrow_glow_layer)
        
        return arrow_layers
    
    def set_camera_view(current_quake, next_quake=None, animation_progress=0):
        """Set the camera view based on current earthquake and animation settings"""
        if current_quake is None:
            return None
            
        lat = current_quake['LATITUDE']
        lon = current_quake['LONGITUDE']
        magnitude = current_quake['MAGNITUDE']
        depth = current_quake['DEPTH (KM)']
        
        # Base zoom level depends on magnitude (larger earthquakes = zoom out more)
        zoom_level = max(5, 10 - magnitude * 0.5)
        
        # Adjust view based on selected camera style
        if selected_camera == "Top-down":
            # Simple top-down view with subtle movement
            return pdk.ViewState(
                latitude=lat,
                longitude=lon,
                zoom=zoom_level,
                pitch=0,
                bearing=animation_progress * 5  # Very subtle rotation
            )
        elif selected_camera == "Tilted View":
            # Enhanced tilted view with more dynamic angle
            pitch_angle = 45 + np.sin(animation_progress * np.pi) * 5
            bearing_angle = 45 + np.sin(animation_progress * np.pi * 2) * 15
            
            return pdk.ViewState(
                latitude=lat,
                longitude=lon,
                zoom=zoom_level,
                pitch=pitch_angle,
                bearing=bearing_angle
            )
        else:  # Dynamic Camera
            # Advanced camera with realistic movement and transitions
            
            # Calculate magnitude-based camera shake (stronger for larger earthquakes)
            mag_factor = min(1.0, magnitude / 8.0)
            shake_intensity = mag_factor * 0.5
            
            # Add subtle camera shake using noise-like functions
            shake_x = shake_intensity * np.sin(animation_progress * 20) * np.sin(animation_progress * 13)
            shake_y = shake_intensity * np.cos(animation_progress * 17) * np.sin(animation_progress * 11)
            
            if next_quake is not None and animation_progress > 0.5:
                # Start transitioning to next earthquake position at 50% of animation
                transition_factor = (animation_progress - 0.5) / 0.5  # Normalize to 0-1
                easing_factor = np.sin(transition_factor * np.pi/2)  # Apply easing for smoother transition
                
                target_lat = next_quake['LATITUDE']
                target_lon = next_quake['LONGITUDE']
                
                # Calculate direction vector between earthquakes
                dx = target_lon - lon
                dy = target_lat - lat
                distance = np.sqrt(dx**2 + dy**2)
                bearing_angle = np.degrees(np.arctan2(dy, dx))
                
                # Calculate optimal camera parameters based on distance and magnitude
                distance_factor = min(1.0, distance / 5.0)
                
                # For distant earthquakes, adjust transition speed
                transition_speed = 1.0 + distance_factor * 0.5
                easing_factor = easing_factor ** (1.0 / transition_speed)
                
                # Interpolate camera position with easing
                view_lat = lat + (target_lat - lat) * easing_factor + shake_y
                view_lon = lon + (target_lon - lon) * easing_factor + shake_x
                
                # Zoom out more for distant points but maintain detail for nearby points
                transition_zoom = zoom_level * (1.0 - distance_factor * 0.4)
                
                # Camera follows path direction with perspective enhancement
                # Point camera in direction of movement for more dramatic effect
                # Gradually adjust bearing to point toward next earthquake
                dynamic_bearing = 45 + (bearing_angle - 45) * easing_factor
                
                # Add dramatic tilt changes during transition
                dynamic_pitch = 45 + 20 * np.sin(animation_progress * 4)
                
                return pdk.ViewState(
                    latitude=view_lat,
                    longitude=view_lon,
                    zoom=transition_zoom,
                    pitch=dynamic_pitch,
                    bearing=dynamic_bearing
                )
            else:
                # Enhanced dynamic view around current earthquake with depth-aware camera
                
                # Deeper earthquakes get more elevated viewpoint
                depth_factor = min(1.0, depth / 100)
                pitch_base = 45 + depth_factor * 15  # Higher pitch for deeper quakes
                
                # Dynamic movement with multiple oscillation frequencies
                pitch_variation = pitch_base + 15 * np.sin(animation_progress * 3) * np.sin(animation_progress * 1.5)
                bearing_variation = 45 + 35 * np.sin(animation_progress * 2) * np.cos(animation_progress * 1.3)
                
                return pdk.ViewState(
                    latitude=lat + shake_y,
                    longitude=lon + shake_x,
                    zoom=zoom_level * (1.0 - animation_progress * 0.1),  # Subtle zoom change
                    pitch=pitch_variation,
                    bearing=bearing_variation
                )
    
    # ------------------------------------------------
    # Enhanced Sequential Animation Controller
    # ------------------------------------------------
    if st.session_state.animation_running and len(sorted_quakes) > 0:
        # Set up animation progress placeholder
        animation_status = st.empty()
        animation_controls = st.empty()
        details_container = st.empty()
        
        # Get current earthquake
        current_idx = st.session_state.current_quake_index
        current_quake = sorted_quakes.iloc[current_idx]
        
        # Sequence control - get next event in sequence
        next_idx = current_idx + 1 if current_idx < len(sorted_quakes) - 1 else None
        next_quake = sorted_quakes.iloc[next_idx] if next_idx is not None else None
        
        # Animation parameters
        start_time = time.time()
        elapsed_time = 0
        animation_complete = False
        
        # Main animation loop for current earthquake
        # Important: Only exit this loop when we complete the current earthquake animation
        # or if the user explicitly stops the animation
        while st.session_state.animation_running and not animation_complete:
            # Update elapsed time
            current_time = time.time()
            elapsed_time = current_time - start_time
            
            # Calculate animation progress (0 to 1)
            animation_progress = min(1.0, elapsed_time / event_duration)
            
            # Create wave layers for current earthquake
            quake_params = calculate_shockwave_parameters(
                current_quake['MAGNITUDE'],
                current_quake['DEPTH (KM)']
            )
            
            wave_layers = generate_ripple_layers(current_quake, elapsed_time, quake_params)
            
            # Add arrow layer pointing to next earthquake if available
            arrow_layer = []
            if next_quake is not None:
                arrow_layer = create_arrow_layer(current_quake, next_quake, animation_progress)
                if arrow_layer:
                    wave_layers.extend(arrow_layer)
            
            # Set camera view
            view_state = set_camera_view(current_quake, next_quake, animation_progress)
            
            # Create the map with all layers
            deck = pdk.Deck(
                map_style="mapbox://styles/mapbox/dark-v10",
                initial_view_state=view_state,
                layers=wave_layers,
                tooltip={"text": "{MAGNITUDE} - {LOCATION}"}
            )
            
            # Update the map
            map_container.pydeck_chart(deck)
            
            # Enhanced event information display with more details
            # Format depth with one decimal place to maintain precision
            depth_str = f"{current_quake['DEPTH (KM)']:.1f} km"
            province_str = f"({current_quake['PROVINCE']})" if 'PROVINCE' in current_quake and current_quake['PROVINCE'] else ""
            
            animation_status.info(
                f"Visualizing earthquake {current_idx + 1} of {len(sorted_quakes)}: "
                f"M{current_quake['MAGNITUDE']:.1f} at {depth_str} {current_quake['LOCATION']} {province_str} • "
                f"Event Progress: {int(animation_progress * 100)}% • "
                f"Sequence Progress: {int((current_idx + animation_progress) / len(sorted_quakes) * 100)}%"
            )
            
            # Detailed information in expandable section
            with details_container.expander("Current Earthquake Details", expanded=False):
                st.write(f"**Magnitude:** {current_quake['MAGNITUDE']:.1f}")
                st.write(f"**Depth:** {current_quake['DEPTH (KM)']:.1f} km")
                st.write(f"**Location:** {current_quake['LOCATION']}")
                if 'PROVINCE' in current_quake and current_quake['PROVINCE']:
                    st.write(f"**Province:** {current_quake['PROVINCE']}")
                if 'DATETIME' in current_quake:
                    st.write(f"**Time:** {current_quake['DATETIME']}")
                if 'INTENSITY' in current_quake:
                    st.write(f"**Intensity:** {current_quake['INTENSITY']}")
            
            # Check if animation for current earthquake is complete
            if elapsed_time >= event_duration:
                # Proceed to next earthquake in sequence or end animation
                if next_idx is not None:
                    # Move to next earthquake in sequence
                    st.session_state.current_quake_index = next_idx
                    # Don't set animation_complete = True here, as we want to continue to the next earthquake
                    st.rerun()  # Force a rerun to immediately start the next earthquake
                else:
                    # We've reached the end of the sequence
                    animation_complete = True
                    st.session_state.animation_running = False
                    
                    # Display comprehensive end-of-sequence summary
                    animation_status.success(f"Animation sequence complete! All {len(sorted_quakes)} earthquake events have been visualized successfully.")
                    
                    # Create summary statistics
                    total_events = len(sorted_quakes)
                    avg_magnitude = sorted_quakes['MAGNITUDE'].mean()
                    max_magnitude = sorted_quakes['MAGNITUDE'].max()
                    min_magnitude = sorted_quakes['MAGNITUDE'].min()
                    max_depth = sorted_quakes['DEPTH (KM)'].max()
                    min_depth = sorted_quakes['DEPTH (KM)'].min()
                    
                    # Show summary in expandable section
                    with details_container.expander("Animation Sequence Summary", expanded=True):
                        st.markdown("### Earthquake Sequence Statistics")
                        st.write(f"**Total Events:** {total_events}")
                        st.write(f"**Date Range:** {sorted_quakes['DATETIME'].min()} to {sorted_quakes['DATETIME'].max()}")
                        st.write(f"**Magnitude Range:** {min_magnitude:.1f} to {max_magnitude:.1f} (Average: {avg_magnitude:.1f})")
                        st.write(f"**Depth Range:** {min_depth:.1f} km to {max_depth:.1f} km")
                        
                        # Show provinces if available
                        if 'PROVINCE' in sorted_quakes.columns:
                            provinces = sorted_quakes['PROVINCE'].value_counts()
                            st.markdown("### Distribution by Province")
                            for province, count in provinces.items():
                                if province:  # Skip empty province names
                                    st.write(f"**{province}:** {count} events")
                        
                        # Show categories if available
                        if 'CATEGORY' in sorted_quakes.columns:
                            categories = sorted_quakes['CATEGORY'].value_counts()
                            st.markdown("### Distribution by Category")
                            for category, count in categories.items():
                                if category:  # Skip empty category names
                                    st.write(f"**{category}:** {count} events")
                    
                    # Offer replay option with prominent button per end-of-sequence handling requirement
                    col1, col2, col3 = animation_controls.columns([1, 2, 1])
                    with col2:
                        if st.button("▶️ Replay Entire Animation Sequence", use_container_width=True, type="primary"):
                            st.session_state.current_quake_index = 0
                            st.session_state.animation_running = True
                            st.rerun()
            
            # Small pause to reduce CPU usage
            time.sleep(0.05)
        
        # If animation was manually stopped or completed
        if not st.session_state.animation_running and not animation_complete:
            animation_status.warning("Animation stopped")
            if animation_controls.button("Resume Animation"):
                st.session_state.animation_running = True
                st.rerun()
    else:
        # Display static map with all earthquakes when not animating
        static_layers = []
        
        # Add markers for all earthquakes
        markers_data = sorted_quakes.apply(
            lambda row: {
                "position": [row['LONGITUDE'], row['LATITUDE']],
                "radius": row['MAGNITUDE'] * 2000,
                "color": intensity_base_colors.get(row['CATEGORY'], (128, 128, 128, 200)),
                "magnitude": row['MAGNITUDE'],
                "depth": row['DEPTH (KM)'],
                "location": row['LOCATION'],
                "province": row['PROVINCE'],
                "datetime": row['DATETIME'],
                "category": row['CATEGORY']
            }, 
            axis=1
        ).tolist()
        
        markers_layer = pdk.Layer(
            "ScatterplotLayer",
            data=markers_data,
            get_position="position",
            get_radius="radius",
            get_fill_color="color",
            get_elevation=0,
            elevation_scale=1,
            pickable=True,
            opacity=0.6,
            stroked=True,
            filled=True,
        )
        
        static_layers.append(markers_layer)
        
        # Calculate center of all earthquakes for initial view
        center_lat = sorted_quakes['LATITUDE'].mean()
        center_lon = sorted_quakes['LONGITUDE'].mean()
        
        # Create static map
        initial_view = pdk.ViewState(
            latitude=center_lat,
            longitude=center_lon,
            zoom=6,
            pitch=30,
            bearing=0
        )
        
        # Create the map
        static_map = pdk.Deck(
            map_style="mapbox://styles/mapbox/dark-v10",
            initial_view_state=initial_view,
            layers=static_layers,
            tooltip={
                "html": "<b>Magnitude:</b> {magnitude}<br/>"
                        "<b>Location:</b> {location}<br/>"
                        "<b>Province:</b> {province}<br/>"
                        "<b>Depth:</b> {depth} km<br/>"
                        "<b>Category:</b> {category}<br/>"
                        "<b>Time:</b> {datetime}",
                "style": {
                    "backgroundColor": "steelblue",
                    "color": "white"
                }
            }
        )
        
        # Display the static map
        map_container.pydeck_chart(static_map)
        
        if not st.session_state.animation_running:
            st.info("Click 'Start Animation' to begin the sequential earthquake visualization. Each event will animate for 5 seconds with directional indicators showing paths to the next earthquake.")

except Exception as e:
    st.error(f"An error occurred: {e}")
    st.exception(e)
