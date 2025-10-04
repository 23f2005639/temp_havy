"""
Heavy Metal Pollution Index (HMPI) Application
This application calculates heavy metal pollution indices for groundwater quality assessment.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import io
import base64
from datetime import datetime

# Import custom modules
from hmpi_calculator import HMPICalculator
from data_processor import DataProcessor
from visualization import Visualizer

def main():
    st.set_page_config(
        page_title="Heavy Metal Pollution Index (HMPI) Calculator",
        page_icon="🌊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        font-size: 2.5em;
        margin-bottom: 0.5em;
    }
    .sub-header {
        text-align: center;
        color: #A23B72;
        font-size: 1.2em;
        margin-bottom: 2em;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1em;
        border-radius: 0.5em;
        margin: 1em 0;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1em;
        border-radius: 0.5em;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown('<h1 class="main-header">🌊 Heavy Metal Pollution Index Calculator</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Automated Assessment of Groundwater Heavy Metal Contamination</p>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'results' not in st.session_state:
        st.session_state.results = None
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select a page:",
        ["Home", "Data Input", "HMPI Analysis", "Visualization", "Export Results", "About"]
    )
    
    # Page routing
    if page == "Home":
        show_home_page()
    elif page == "Data Input":
        show_data_input_page()
    elif page == "HMPI Analysis":
        show_analysis_page()
    elif page == "Visualization":
        show_visualization_page()
    elif page == "Export Results":
        show_export_page()
    elif page == "About":
        show_about_page()

def show_home_page():
    """Display the home page with application overview"""
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div class="info-box">
        <h3>🎯 Purpose</h3>
        <p>This application provides automated calculation of Heavy Metal Pollution Indices (HMPI) 
        for groundwater quality assessment. It helps researchers, environmental scientists, and 
        policymakers evaluate heavy metal contamination levels in groundwater samples.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Key Features
    st.markdown("### 🌟 Key Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **📊 Automated Calculations**
        - Heavy Metal Pollution Index (HMPI)
        - Heavy Metal Evaluation Index (HEI) 
        - Contamination Index (Cd)
        - Pollution Load Index (PLI)
        """)
    
    with col2:
        st.markdown("""
        **📈 Visualization**
        - Interactive maps with geo-coordinates
        - Statistical charts and graphs
        - Contamination level categorization
        - Trend analysis
        """)
    
    with col3:
        st.markdown("""
        **💾 Data Management**
        - CSV/Excel file upload
        - Manual data entry
        - Results export
        - Report generation
        """)
    
    # Quick Stats
    if st.session_state.data is not None:
        st.markdown("### 📋 Current Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Samples", len(st.session_state.data))
        
        with col2:
            heavy_metals = [col for col in st.session_state.data.columns 
                          if col.lower() in ['pb', 'cd', 'cr', 'ni', 'cu', 'zn', 'fe', 'mn', 'as', 'hg']]
            st.metric("Heavy Metals", len(heavy_metals))
        
        with col3:
            if st.session_state.results is not None:
                high_pollution = len(st.session_state.results[st.session_state.results['HMPI_Category'] == 'High'])
                st.metric("High Pollution Sites", high_pollution)
        
        with col4:
            if 'latitude' in st.session_state.data.columns and 'longitude' in st.session_state.data.columns:
                geo_samples = len(st.session_state.data.dropna(subset=['latitude', 'longitude']))
                st.metric("Geo-located Samples", geo_samples)
    
    # Getting Started
    st.markdown("### 🚀 Getting Started")
    st.markdown("""
    1. **Data Input**: Upload your groundwater heavy metal concentration data (CSV/Excel)
    2. **Analysis**: Configure parameters and calculate HMPI values
    3. **Visualization**: Explore results through interactive charts and maps
    4. **Export**: Download results and generate reports
    """)

def show_data_input_page():
    """Display the data input page"""
    st.header("📥 Data Input")
    
    # Data input options
    input_method = st.radio(
        "Choose input method:",
        ["Upload File", "Manual Entry", "Use Sample Data"]
    )
    
    if input_method == "Upload File":
        show_file_upload()
    elif input_method == "Manual Entry":
        show_manual_entry()
    elif input_method == "Use Sample Data":
        show_sample_data()
    
    # Display current data
    if st.session_state.data is not None:
        st.markdown("### 📊 Current Dataset")
        st.dataframe(st.session_state.data)
        
        # Data quality checks
        st.markdown("### 🔍 Data Quality Check")
        data_processor = DataProcessor()
        quality_report = data_processor.check_data_quality(st.session_state.data)
        
        for check, status in quality_report.items():
            if status:
                st.success(f"✅ {check}")
            else:
                st.warning(f"⚠️ {check}")

def show_file_upload():
    """Handle file upload functionality"""
    uploaded_file = st.file_uploader(
        "Upload your data file (CSV/Excel)",
        type=['csv', 'xlsx', 'xls'],
        help="File should contain heavy metal concentrations with optional latitude/longitude columns"
    )
    
    if uploaded_file is not None:
        try:
            # Read file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success("✅ File uploaded successfully!")
            
            # Show preview
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            # Column mapping
            st.subheader("Column Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Required Columns (Heavy Metals)**")
                metal_columns = {}
                available_cols = df.columns.tolist()
                
                standard_metals = ['Pb', 'Cd', 'Cr', 'Ni', 'Cu', 'Zn', 'Fe', 'Mn', 'As', 'Hg']
                
                for metal in standard_metals:
                    metal_columns[metal] = st.selectbox(
                        f"{metal} concentration:",
                        ["None"] + available_cols,
                        index=0 if metal not in available_cols else available_cols.index(metal) + 1
                    )
            
            with col2:
                st.markdown("**Optional Columns (Location)**")
                lat_col = st.selectbox("Latitude:", ["None"] + available_cols)
                lon_col = st.selectbox("Longitude:", ["None"] + available_cols)
                sample_id_col = st.selectbox("Sample ID:", ["None"] + available_cols)
            
            if st.button("Process Data"):
                # Process the uploaded data
                processed_data = process_uploaded_data(df, metal_columns, lat_col, lon_col, sample_id_col)
                st.session_state.data = processed_data
                st.success("Data processed and stored successfully!")
                st.rerun()
                
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

def show_manual_entry():
    """Handle manual data entry"""
    st.subheader("Manual Data Entry")
    
    # Number of samples
    num_samples = st.number_input("Number of samples:", min_value=1, max_value=100, value=1)
    
    # Heavy metals to include
    available_metals = ['Pb', 'Cd', 'Cr', 'Ni', 'Cu', 'Zn', 'Fe', 'Mn', 'As', 'Hg']
    selected_metals = st.multiselect(
        "Select heavy metals to include:",
        available_metals,
        default=['Pb', 'Cd', 'Cr', 'Ni']
    )
    
    if selected_metals:
        # Create input form
        with st.form("manual_data_form"):
            data_dict = {'Sample_ID': []}
            
            # Initialize columns
            for metal in selected_metals:
                data_dict[metal] = []
            
            data_dict['Latitude'] = []
            data_dict['Longitude'] = []
            
            # Input fields for each sample
            for i in range(num_samples):
                st.markdown(f"### Sample {i+1}")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    sample_id = st.text_input(f"Sample ID {i+1}:", value=f"Sample_{i+1}")
                    data_dict['Sample_ID'].append(sample_id)
                
                with col2:
                    lat = st.number_input(f"Latitude {i+1}:", value=0.0, format="%.6f")
                    data_dict['Latitude'].append(lat if lat != 0.0 else None)
                
                with col3:
                    lon = st.number_input(f"Longitude {i+1}:", value=0.0, format="%.6f")
                    data_dict['Longitude'].append(lon if lon != 0.0 else None)
                
                # Metal concentrations
                metal_cols = st.columns(len(selected_metals))
                for j, metal in enumerate(selected_metals):
                    with metal_cols[j]:
                        conc = st.number_input(
                            f"{metal} (mg/L) {i+1}:",
                            min_value=0.0,
                            value=0.0,
                            format="%.4f"
                        )
                        data_dict[metal].append(conc)
            
            if st.form_submit_button("Save Data"):
                df = pd.DataFrame(data_dict)
                st.session_state.data = df
                st.success("Manual data saved successfully!")
                st.rerun()

def show_sample_data():
    """Load and display sample data"""
    st.subheader("Sample Dataset")
    
    # Create sample data
    sample_data = create_sample_data()
    
    st.markdown("**Sample groundwater heavy metal concentration data:**")
    st.dataframe(sample_data)
    
    if st.button("Use Sample Data"):
        st.session_state.data = sample_data
        st.success("Sample data loaded successfully!")
        st.rerun()

def show_analysis_page():
    """Display the HMPI analysis page"""
    st.header("🧪 HMPI Analysis")
    
    if st.session_state.data is None:
        st.warning("Please upload or enter data first in the Data Input page.")
        return
    
    # Analysis configuration
    st.subheader("Analysis Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Standards Selection**")
        standard_type = st.selectbox(
            "Select water quality standard:",
            ["WHO Guidelines", "EPA Standards", "IS 10500 (Indian)", "Custom"]
        )
    
    with col2:
        st.markdown("**Calculation Options**")
        include_hei = st.checkbox("Calculate Heavy Metal Evaluation Index (HEI)", value=True)
        include_cd = st.checkbox("Calculate Contamination Index (Cd)", value=True)
        include_pli = st.checkbox("Calculate Pollution Load Index (PLI)", value=True)
    
    # Standards values
    standards = get_water_quality_standards(standard_type)
    
    if standard_type == "Custom":
        st.subheader("Custom Standards (mg/L)")
        
        # Get available metals from data
        heavy_metals = [col for col in st.session_state.data.columns 
                       if col.lower() in ['pb', 'cd', 'cr', 'ni', 'cu', 'zn', 'fe', 'mn', 'as', 'hg']]
        
        custom_standards = {}
        metal_cols = st.columns(3)
        
        for i, metal in enumerate(heavy_metals):
            with metal_cols[i % 3]:
                custom_standards[metal] = st.number_input(
                    f"{metal} Standard:",
                    min_value=0.0,
                    value=standards.get(metal, 0.01),
                    format="%.4f"
                )
        
        standards = custom_standards
    
    # Display standards being used
    st.subheader("Current Standards (mg/L)")
    standards_df = pd.DataFrame(list(standards.items()), columns=['Heavy Metal', 'Standard Value'])
    st.dataframe(standards_df)
    
    # Run analysis
    if st.button("Calculate HMPI", type="primary"):
        with st.spinner("Calculating heavy metal pollution indices..."):
            calculator = HMPICalculator()
            
            results = calculator.calculate_all_indices(
                st.session_state.data, 
                standards,
                include_hei=include_hei,
                include_cd=include_cd,
                include_pli=include_pli
            )
            
            st.session_state.results = results
            
        st.success("Analysis completed successfully!")
        
        # Display results
        show_analysis_results()

def show_analysis_results():
    """Display analysis results"""
    if st.session_state.results is None:
        return
    
    results = st.session_state.results
    
    st.subheader("📊 Analysis Results")
    
    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_hmpi = results['HMPI'].mean()
        st.metric("Average HMPI", f"{avg_hmpi:.2f}")
    
    with col2:
        high_pollution = len(results[results['HMPI_Category'] == 'High'])
        st.metric("High Pollution Sites", high_pollution)
    
    with col3:
        max_hmpi = results['HMPI'].max()
        st.metric("Maximum HMPI", f"{max_hmpi:.2f}")
    
    with col4:
        contaminated_sites = len(results[results['HMPI'] > 100])
        st.metric("Contaminated Sites", contaminated_sites)
    
    # Results table
    st.subheader("Detailed Results")
    st.dataframe(results)
    
    # Category distribution
    st.subheader("Pollution Level Distribution")
    
    category_counts = results['HMPI_Category'].value_counts()
    
    fig_pie = px.pie(
        values=category_counts.values,
        names=category_counts.index,
        title="Distribution of Pollution Levels"
    )
    
    st.plotly_chart(fig_pie, use_container_width=True)

def show_visualization_page():
    """Display visualization page"""
    st.header("📈 Data Visualization")
    
    if st.session_state.results is None:
        st.warning("Please run HMPI analysis first.")
        return
    
    visualizer = Visualizer()
    results = st.session_state.results
    
    # Visualization options
    viz_type = st.selectbox(
        "Select visualization type:",
        ["Geographic Map", "HMPI Distribution", "Heavy Metal Comparison", "Metal Risk Contributions", 
         "Health Risk Assessment", "Contamination Source Analysis", "Correlation Analysis", "Statistical Summary"]
    )
    
    if viz_type == "Geographic Map":
        if 'latitude' in results.columns and 'longitude' in results.columns:
            fig_map = visualizer.create_pollution_map(results)
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.warning("Geographic coordinates not available in the dataset.")
    
    elif viz_type == "HMPI Distribution":
        fig_hist = visualizer.create_hmpi_histogram(results)
        st.plotly_chart(fig_hist, use_container_width=True)
        
        fig_box = visualizer.create_hmpi_boxplot(results)
        st.plotly_chart(fig_box, use_container_width=True)
    
    elif viz_type == "Heavy Metal Comparison":
        fig_metal = visualizer.create_metal_comparison(st.session_state.data)
        st.plotly_chart(fig_metal, use_container_width=True)
    
    elif viz_type == "Metal Risk Contributions":
        fig_contrib = visualizer.create_metal_contribution_chart(results)
        st.plotly_chart(fig_contrib, use_container_width=True)
        
        st.markdown("### 📋 Interpretation")
        st.markdown("""
        This chart shows how much each heavy metal contributes to the overall HMPI value for each sample.
        - **Higher contributions** indicate metals that are the primary pollution concern
        - **Stacked bars** show the relative importance of different metals
        - Use this to prioritize treatment strategies for specific metals
        """)
    
    elif viz_type == "Health Risk Assessment":
        fig_health = visualizer.create_health_risk_assessment_chart(results)
        st.plotly_chart(fig_health, use_container_width=True)
        
        st.markdown("### 🏥 Health Risk Interpretation")
        st.markdown("""
        **Risk Categories:**
        - **Minimal Risk**: Safe for consumption, no immediate health concerns
        - **Low Risk**: Generally safe, regular monitoring recommended
        - **Moderate Risk**: Treatment recommended before consumption
        - **High Risk**: Significant health concerns, avoid consumption
        - **Very High Risk**: Serious health hazards, immediate action required
        - **Extreme Risk**: Severe contamination, unsuitable for any use
        """)
    
    elif viz_type == "Contamination Source Analysis":
        fig_source = visualizer.create_contamination_source_analysis(results)
        st.plotly_chart(fig_source, use_container_width=True)
        
        st.markdown("### 🏭 Source Analysis Insights")
        st.markdown("""
        This analysis helps identify contamination patterns by source type:
        - **Industrial discharge**: Often shows elevated Pb, Cd, Cr, Ni levels
        - **Agricultural runoff**: May show As contamination from pesticides
        - **Mining activities**: Typically elevated Fe, Mn, Zn, Cu levels
        - **Urban pollution**: Mixed contamination from multiple sources
        """)
    
    elif viz_type == "Correlation Analysis":
        fig_corr = visualizer.create_correlation_heatmap(st.session_state.data)
        st.plotly_chart(fig_corr, use_container_width=True)
    
    elif viz_type == "Statistical Summary":
        visualizer.show_statistical_summary(results)

def show_export_page():
    """Display export page"""
    st.header("💾 Export Results")
    
    if st.session_state.results is None:
        st.warning("No results available to export. Please run analysis first.")
        return
    
    results = st.session_state.results
    
    st.subheader("Download Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CSV export
        csv_data = results.to_csv(index=False)
        st.download_button(
            label="📄 Download as CSV",
            data=csv_data,
            file_name=f"hmpi_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # Excel export
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            results.to_excel(writer, sheet_name='HMPI_Results', index=False)
            if st.session_state.data is not None:
                st.session_state.data.to_excel(writer, sheet_name='Original_Data', index=False)
        
        excel_data = excel_buffer.getvalue()
        
        st.download_button(
            label="📊 Download as Excel",
            data=excel_data,
            file_name=f"hmpi_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    # Generate report
    st.subheader("Generate Report")
    
    if st.button("📝 Generate Summary Report"):
        report = generate_summary_report(results)
        
        st.download_button(
            label="📑 Download Report",
            data=report,
            file_name=f"hmpi_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

def show_about_page():
    """Display about page"""
    st.header("ℹ️ About HMPI Calculator")
    
    st.markdown("""
    ### Heavy Metal Pollution Index (HMPI)
    
    The Heavy Metal Pollution Index is a comprehensive method for assessing the degree of heavy metal 
    contamination in groundwater. It provides a single numerical value that represents the overall 
    pollution level based on multiple heavy metal concentrations.
    
    #### Calculation Formula:
    
    **HMPI = Σ(Wi × Qi) / Σ(Wi)**
    
    Where:
    - **Wi** = Unit weight of the ith parameter
    - **Qi** = Sub-index of the ith parameter
    - **Wi = K / Si** (K = constant of proportionality, Si = standard permissible value)
    - **Qi = (Ci / Si) × 100** (Ci = observed concentration)
    
    #### Classification:
    - **HMPI < 30**: Low pollution
    - **30 ≤ HMPI < 50**: Medium pollution  
    - **50 ≤ HMPI < 100**: High pollution
    - **HMPI ≥ 100**: Very high pollution (unsuitable for drinking)
    
    #### Other Indices:
    
    **Heavy Metal Evaluation Index (HEI):**
    HEI = Σ(Ci / Si) / n
    
    **Contamination Index (Cd):**
    Cd = Σ(CAF) (where CAF = Ci / Si)
    
    **Pollution Load Index (PLI):**
    PLI = (CF₁ × CF₂ × ... × CFₙ)^(1/n)
    
    #### Standards Supported:
    - WHO Guidelines for Drinking Water Quality
    - US EPA National Primary Drinking Water Standards
    - Indian Standard IS 10500:2012
    - Custom user-defined standards
    
    #### Data Requirements:
    - Heavy metal concentrations (mg/L or μg/L)
    - Optional: Geographic coordinates (latitude, longitude)
    - Optional: Sample identifiers and metadata
    
    #### References:
    1. Prasad, B., & Bose, J. M. (2001). Evaluation of the heavy metal pollution index for surface and spring water near a limestone mining area of the lower Himalayas.
    2. Edet, A. E., & Offiong, O. E. (2002). Evaluation of water quality pollution indices for heavy metal contamination monitoring.
    3. WHO. (2017). Guidelines for drinking-water quality: fourth edition incorporating the first addendum.
    """)
    
    st.markdown("---")
    st.markdown("**Version:** 1.0.0 | **Developer:** Environmental Data Analysis Team")

# Helper functions

def process_uploaded_data(df, metal_columns, lat_col, lon_col, sample_id_col):
    """Process uploaded data based on column mapping"""
    processed_data = pd.DataFrame()
    
    # Add sample ID
    if sample_id_col != "None":
        processed_data['Sample_ID'] = df[sample_id_col]
    else:
        processed_data['Sample_ID'] = [f"Sample_{i+1}" for i in range(len(df))]
    
    # Add metal concentrations
    for metal, col_name in metal_columns.items():
        if col_name != "None":
            processed_data[metal] = pd.to_numeric(df[col_name], errors='coerce')
    
    # Add coordinates if available
    if lat_col != "None":
        processed_data['latitude'] = pd.to_numeric(df[lat_col], errors='coerce')
    if lon_col != "None":
        processed_data['longitude'] = pd.to_numeric(df[lon_col], errors='coerce')
    
    return processed_data

def get_water_quality_standards(standard_type):
    """Get water quality standards for different organizations"""
    standards = {
        "WHO Guidelines": {
            'Pb': 0.01, 'Cd': 0.003, 'Cr': 0.05, 'Ni': 0.07,
            'Cu': 2.0, 'Zn': 3.0, 'Fe': 0.3, 'Mn': 0.4,
            'As': 0.01, 'Hg': 0.006
        },
        "EPA Standards": {
            'Pb': 0.015, 'Cd': 0.005, 'Cr': 0.1, 'Ni': 0.1,
            'Cu': 1.3, 'Zn': 5.0, 'Fe': 0.3, 'Mn': 0.05,
            'As': 0.01, 'Hg': 0.002
        },
        "IS 10500 (Indian)": {
            'Pb': 0.01, 'Cd': 0.003, 'Cr': 0.05, 'Ni': 0.02,
            'Cu': 0.05, 'Zn': 5.0, 'Fe': 0.3, 'Mn': 0.1,
            'As': 0.01, 'Hg': 0.001
        }
    }
    
    return standards.get(standard_type, standards["WHO Guidelines"])

def create_sample_data():
    """Create realistic sample dataset based on research from groundwater studies"""
    np.random.seed(42)
    
    # Generate 25 sample locations representing different contamination scenarios
    n_samples = 25
    
    # Create realistic geographical coordinates (covering major Indian groundwater regions)
    # Including areas known for groundwater contamination: Delhi, Punjab, Gujarat, West Bengal, Rajasthan
    locations = [
        # Delhi NCR region (known industrial contamination)
        (28.6139, 77.2090), (28.5355, 77.3910), (28.7041, 77.1025), (28.4595, 77.0266),
        # Punjab (agricultural contamination)
        (30.9010, 75.8573), (31.3260, 75.5762), (30.7333, 76.7794), (31.1471, 75.3412),
        # Gujarat (industrial belt)
        (22.2587, 70.7795), (21.1702, 72.8311), (22.3072, 70.8022), (23.0225, 72.5714),
        # West Bengal (arsenic contamination)
        (22.5726, 88.3639), (23.2324, 87.8615), (22.9868, 87.8550), (23.5041, 87.3119),
        # Rajasthan (fluoride contamination)
        (26.9124, 75.7873), (27.0238, 74.2179), (26.4499, 74.6399), (25.3176, 74.1258),
        # Tamil Nadu (industrial areas)
        (11.0168, 76.9558), (12.9716, 77.5946), (13.0827, 80.2707), (11.3410, 77.7172),
        # Maharashtra (mining areas)
        (19.0760, 72.8777)
    ]
    
    # Extend locations if needed
    while len(locations) < n_samples:
        locations.append(locations[len(locations) % len(locations)])
    
    locations = locations[:n_samples]
    
    # Create realistic contamination data based on literature values
    # Sources: Various studies on groundwater contamination in India
    
    # Define contamination scenarios
    scenarios = {
        'background': {'weight': 0.3, 'multiplier': (0.1, 0.5)},    # Natural background levels
        'moderate': {'weight': 0.4, 'multiplier': (0.5, 2.0)},      # Moderate anthropogenic influence
        'high': {'weight': 0.2, 'multiplier': (2.0, 10.0)},         # High contamination
        'extreme': {'weight': 0.1, 'multiplier': (10.0, 50.0)}      # Extreme contamination (mining/industrial)
    }
    
    # WHO guideline values as baseline
    baseline_concentrations = {
        'Pb': 0.01,   # WHO guideline
        'Cd': 0.003,  # WHO guideline
        'Cr': 0.05,   # WHO guideline
        'Ni': 0.07,   # WHO guideline (updated)
        'Cu': 2.0,    # WHO guideline
        'Zn': 3.0,    # WHO no guideline, using aesthetic threshold
        'Fe': 0.3,    # WHO no guideline, using aesthetic threshold
        'Mn': 0.4,    # WHO guideline
        'As': 0.01,   # WHO guideline
        'Hg': 0.006   # WHO guideline
    }
    
    # Realistic correlation patterns (some metals tend to co-occur)
    correlation_groups = {
        'industrial': ['Pb', 'Cd', 'Cr', 'Ni'],    # Industrial sources
        'mining': ['Fe', 'Mn', 'Zn', 'Cu'],        # Mining/geological sources
        'agricultural': ['As', 'Cr'],               # Agricultural runoff
        'urban': ['Pb', 'Cu', 'Zn']                # Urban contamination
    }
    
    data = {
        'Sample_ID': [f"GW_{i+1:03d}" for i in range(n_samples)],
        'latitude': [loc[0] for loc in locations],
        'longitude': [loc[1] for loc in locations],
        'Location_Type': [],
        'Contamination_Source': []
    }
    
    # Initialize metal concentrations
    for metal in baseline_concentrations.keys():
        data[metal] = []
    
    # Generate samples with different contamination scenarios
    scenario_names = list(scenarios.keys())
    scenario_weights = [scenarios[s]['weight'] for s in scenario_names]
    
    for i in range(n_samples):
        # Choose contamination scenario
        scenario = np.random.choice(scenario_names, p=scenario_weights)
        min_mult, max_mult = scenarios[scenario]['multiplier']
        
        # Assign location type and source
        if i < 8:
            location_type = 'Industrial'
            contamination_source = 'Industrial_discharge'
            dominant_group = 'industrial'
        elif i < 16:
            location_type = 'Agricultural'
            contamination_source = 'Agricultural_runoff'
            dominant_group = 'agricultural'
        elif i < 20:
            location_type = 'Mining'
            contamination_source = 'Mining_activities'
            dominant_group = 'mining'
        else:
            location_type = 'Urban'
            contamination_source = 'Urban_pollution'
            dominant_group = 'urban'
        
        data['Location_Type'].append(location_type)
        data['Contamination_Source'].append(contamination_source)
        
        # Generate metal concentrations with realistic correlations
        base_multiplier = np.random.uniform(min_mult, max_mult)
        
        for metal in baseline_concentrations.keys():
            # Apply correlation effects
            if metal in correlation_groups[dominant_group]:
                # Higher contamination for metals in dominant group
                metal_multiplier = base_multiplier * np.random.uniform(0.8, 1.5)
            else:
                # Lower contamination for other metals
                metal_multiplier = base_multiplier * np.random.uniform(0.1, 0.8)
            
            # Add some random variation
            random_factor = np.random.lognormal(0, 0.3)
            
            # Calculate final concentration
            concentration = baseline_concentrations[metal] * metal_multiplier * random_factor
            
            # Add realistic detection limits and rounding
            if concentration < 0.001:
                concentration = np.random.uniform(0.0005, 0.001)
            
            # Round to realistic precision (analytical precision)
            if concentration < 0.01:
                concentration = round(concentration, 4)
            elif concentration < 1.0:
                concentration = round(concentration, 3)
            else:
                concentration = round(concentration, 2)
            
            data[metal].append(concentration)
    
    # Add some specific high-contamination cases based on real studies
    # Case 1: Arsenic contamination (West Bengal style)
    data['As'][12] = 0.085   # Severe arsenic contamination
    data['As'][13] = 0.156   # Extreme arsenic contamination
    data['As'][14] = 0.203   # Very extreme arsenic contamination
    
    # Case 2: Lead contamination (Industrial area)
    data['Pb'][1] = 0.187    # High lead contamination
    data['Pb'][2] = 0.045    # Moderate lead contamination
    
    # Case 3: Chromium contamination (Tannery effluent)
    data['Cr'][8] = 0.285    # High chromium contamination
    data['Cr'][9] = 0.156    # Moderate chromium contamination
    
    # Case 4: Multi-metal contamination (Mining area)
    data['Fe'][16] = 2.45    # High iron
    data['Mn'][16] = 1.25    # High manganese
    data['Zn'][16] = 8.95    # High zinc
    data['Cu'][16] = 4.56    # High copper
    
    # Case 5: Cadmium contamination (Battery manufacturing)
    data['Cd'][5] = 0.0245   # High cadmium
    data['Cd'][6] = 0.0156   # Moderate cadmium
    
    return pd.DataFrame(data)

def generate_summary_report(results):
    """Generate a summary report of the analysis"""
    report = f"""
HEAVY METAL POLLUTION INDEX (HMPI) ANALYSIS REPORT
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

==============================================
SUMMARY STATISTICS
==============================================

Total Samples Analyzed: {len(results)}
Average HMPI: {results['HMPI'].mean():.2f}
Maximum HMPI: {results['HMPI'].max():.2f}
Minimum HMPI: {results['HMPI'].min():.2f}
Standard Deviation: {results['HMPI'].std():.2f}

==============================================
POLLUTION LEVEL CLASSIFICATION
==============================================

"""
    
    category_counts = results['HMPI_Category'].value_counts()
    for category, count in category_counts.items():
        percentage = (count / len(results)) * 100
        report += f"{category} Pollution: {count} samples ({percentage:.1f}%)\n"
    
    report += f"""

==============================================
CONTAMINATED SITES (HMPI > 100)
==============================================

Number of contaminated sites: {len(results[results['HMPI'] > 100])}
"""
    
    contaminated = results[results['HMPI'] > 100]
    if len(contaminated) > 0:
        report += "\nContaminated sample details:\n"
        for _, row in contaminated.iterrows():
            report += f"- {row['Sample_ID']}: HMPI = {row['HMPI']:.2f}\n"
    
    report += """

==============================================
RECOMMENDATIONS
==============================================

Based on the HMPI analysis results:

1. Samples with HMPI < 30: Suitable for drinking with minimal treatment
2. Samples with 30 ≤ HMPI < 50: Requires basic treatment before consumption
3. Samples with 50 ≤ HMPI < 100: Requires advanced treatment
4. Samples with HMPI ≥ 100: Unsuitable for drinking, requires immediate action

For sites with high pollution levels, consider:
- Source identification and control
- Advanced water treatment technologies
- Alternative water sources
- Regular monitoring and assessment

==============================================
METHODOLOGY
==============================================

HMPI Calculation: Σ(Wi × Qi) / Σ(Wi)
Where Wi = K/Si and Qi = (Ci/Si) × 100

Standards used: WHO Guidelines for Drinking Water Quality
Analysis performed using automated HMPI calculator v1.0

==============================================
"""
    
    return report

if __name__ == "__main__":
    main()