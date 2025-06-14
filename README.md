# 🌍 Earthquake Time Series Analysis

## Project Overview
This application provides interactive visualizations and analytical tools for exploring earthquake data over time. The dashboard allows users to analyze patterns, trends, and geographical distributions of earthquake events through multiple visualization methods.

## Features

### 📈 Plotting Time-Series
- Interactive time series charts for earthquake frequency analysis
- Multiple time resolution options (daily, weekly, monthly, quarterly, yearly)
- Moving average trend lines with customizable windows
- Calendar heatmap for visualizing daily earthquake activity patterns
- Hourly and day-of-week frequency analysis

### 🌍 Mapping & Animation
- Geographical visualization of earthquake locations
- Animation controls for viewing earthquake progression over time
- Interactive map with customizable layers and filters
- Regional clustering and hotspot identification

### 📊 DataFrame Time-Series
- Raw data exploration and filtering capabilities
- Custom query builder for advanced data filtering
- Temporal pattern analysis across different time scales
- Statistical summaries and aggregations

### Advanced Analysis Tools
- Cumulative event tracking and visualization
- Rate of change analysis for detecting patterns in earthquake frequency
- Moving window analysis for trend identification
- Seasonal pattern detection and comparative visualization

## Installation

### Prerequisites
- Python 3.7+
- pip (Python package installer)

### Setup Instructions

1. Clone this repository:
   ```
   git clone <repository-url>
   cd Project_IS-107-Jessel_Cagmat
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```
   
   Main dependencies include:
   - streamlit
   - pandas
   - plotly
   - numpy

3. Prepare your data:
   - Ensure your earthquake data is in CSV format
   - Required columns: 'DATE & TIME', 'DATE', 'LATITUDE', 'LONGITUDE', 'MAGNITUDE', 'DEPTH (KM)'
   - Place your CSV file as "Earthquake_Data.csv" in the project root directory (or modify the file path in the code)

## Usage

1. Launch the Streamlit application:
   ```
   streamlit run Home.py
   ```

2. Navigate through the application:
   - Use the sidebar to access different visualization pages
   - Apply global filters to customize your analysis
   - Interact with the timeline controls to sync visualizations across pages
   - Use playback controls for animated visualizations

## Data Structure

The application expects a CSV file with the following structure:
- `DATE & TIME`: Datetime of earthquake event (YYYY-MM-DD HH:MM:SS)
- `DATE`: Date of earthquake event (YYYY-MM-DD)
- `LATITUDE`: Geographical latitude of epicenter
- `LONGITUDE`: Geographical longitude of epicenter
- `MAGNITUDE`: Earthquake magnitude
- `DEPTH (KM)`: Depth of earthquake in kilometers

## Technical Details

### Architecture
The application is built using Streamlit's multi-page app framework:
- `Earthquake-Time_Series_Analysis.py`: Main entry point and global controls
- `pages/1_📈_Plotting_Time-Series.py`: Time series visualization tools
- `pages/2_🌍_Mapping_Animate.py`: Geographical mapping and animation
- `pages/3_📊_DataFrame_Time-Series.py`: Data exploration and manipulation

### State Management
- Uses Streamlit's session_state to maintain consistent state across pages
- Global timeline control synchronizes visualizations across different views

### Performance Considerations
- Data caching using @st.cache_data for improved responsiveness
- Progressive loading for large datasets
- Optimized rendering for complex visualizations

## Contributing
Contributions to enhance the application are welcome. Please follow these steps:
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Developed by Jessel Cagmat
- Earthquake data source: [Include your data source here]
- Built with Streamlit, Pandas, and Plotly

## Contact
For questions or feedback, please contact: [Include contact information if appropriate]
