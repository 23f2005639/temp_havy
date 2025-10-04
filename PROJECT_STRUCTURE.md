# HMPI Calculator - Project Structure

This document provides an overview of the project structure and file organization.

## Directory Structure

```
hmpi_app/
├── app.py                  # Main Streamlit application
├── hmpi_calculator.py      # Core HMPI calculation engine
├── data_processor.py       # Data validation and processing
├── visualization.py        # Charts and mapping functions
├── config.py              # Configuration and constants
├── requirements.txt       # Python dependencies
├── setup.py              # Automated setup script
├── start.sh              # Quick start bash script
├── sample_data.csv       # Sample groundwater data
├── README.md             # Detailed documentation
└── PROJECT_STRUCTURE.md  # This file
```

## File Descriptions

### Core Application Files

**app.py** (2,500+ lines)
- Main Streamlit application serving as the user interface
- Contains 6 main pages: Home, Data Input, HMPI Analysis, Visualization, Export, About
- Handles file uploads, data processing workflows, and user interactions
- Integrates all other modules for complete functionality

**hmpi_calculator.py** (~400 lines)
- Core calculation engine implementing HMPI and related pollution indices
- Functions: calculate_hmpi(), calculate_hei(), calculate_pollution_load_index()
- Supports multiple water quality standards (WHO, EPA, IS 10500)
- Includes categorization functions for pollution levels

**data_processor.py** (~350 lines)
- Data validation, cleaning, and preprocessing utilities
- Functions: check_data_quality(), clean_data(), validate_coordinates()
- Handles missing values, outliers, and data type conversions
- Ensures data integrity before calculations

**visualization.py** (~500 lines)
- Interactive visualizations using Plotly and Folium
- Functions: create_pollution_map(), create_hmpi_histogram(), create_correlation_heatmap()
- Generates maps, charts, and statistical summaries
- Supports export of visualizations

### Configuration and Setup

**config.py** (~150 lines)
- Central configuration file with water quality standards
- Contains WHO, EPA, and IS 10500 permissible limits
- HMPI classification thresholds and color schemes
- Application metadata and settings

**requirements.txt** (8 lines)
- Python package dependencies with version specifications
- Includes streamlit, pandas, plotly, folium, numpy, openpyxl
- Ensures reproducible environment setup

**setup.py** (~200 lines)
- Automated setup script for environment preparation
- Checks Python version compatibility
- Installs dependencies and creates sample data
- Optional application launcher

**start.sh** (~150 lines)
- Bash script for quick application startup
- Interactive menu for different setup options
- Color-coded output for better user experience
- Handles dependency installation and cleanup

### Data and Documentation

**sample_data.csv** (5 records)
- Sample groundwater data for testing the application
- Contains coordinates and heavy metal concentrations
- Pre-formatted for immediate use in the application
- Covers various pollution levels for demonstration

**README.md** (~200 lines)
- Comprehensive documentation and user guide
- Installation instructions and usage examples
- HMPI formula explanations and calculation methodology
- Troubleshooting guide and frequently asked questions

## Module Dependencies

```
app.py
├── hmpi_calculator.py
├── data_processor.py
├── visualization.py
└── config.py

hmpi_calculator.py
├── pandas
├── numpy
└── config.py

data_processor.py
├── pandas
└── numpy

visualization.py
├── plotly.express
├── plotly.graph_objects
├── folium
└── streamlit

config.py
└── (no dependencies)
```

## Data Flow

1. **Input**: User uploads CSV/Excel files or uses sample data
2. **Validation**: data_processor.py validates and cleans the data
3. **Calculation**: hmpi_calculator.py computes HMPI and other indices
4. **Visualization**: visualization.py creates interactive charts and maps
5. **Export**: app.py handles result export in multiple formats

## Key Features Implemented

### Calculation Engine
- Heavy Metal Pollution Index (HMPI) using standard formula
- Heavy metal Evaluation Index (HEI)
- Pollution Load Index (PLI)
- Contamination Degree (Cd)

### Data Processing
- CSV/Excel file upload support
- Data quality validation
- Missing value handling
- Outlier detection and removal
- Unit conversion capabilities

### Visualization
- Interactive pollution maps with Folium
- HMPI distribution histograms
- Correlation heatmaps
- Radar charts for multi-parameter analysis
- Statistical summary tables

### User Interface
- Multi-page Streamlit application
- File upload/download functionality
- Real-time calculation updates
- Interactive parameter adjustment
- Comprehensive help documentation

## Quality Assurance

### Testing Data
- Sample dataset covers various pollution scenarios
- Coordinates use realistic geographic locations
- Metal concentrations span different risk categories

### Error Handling
- Input validation for all user data
- Graceful handling of missing or invalid data
- Clear error messages and user guidance
- Fallback options for failed operations

### Documentation
- Inline code comments throughout all modules
- Comprehensive README with examples
- Formula explanations and references
- Installation and troubleshooting guides

## Deployment Ready

The application is ready for deployment with:
- ✅ All dependencies specified in requirements.txt
- ✅ Automated setup scripts for easy installation
- ✅ Sample data for immediate testing
- ✅ Comprehensive documentation
- ✅ Error handling and validation
- ✅ Multiple export formats supported
- ✅ Professional UI/UX design

## Next Steps for Users

1. **Installation**: Run `./start.sh` or `python setup.py`
2. **Testing**: Use provided sample data to test all features
3. **Customization**: Modify config.py for specific requirements
4. **Deployment**: Follow README.md for production deployment
5. **Enhancement**: Add new features based on user feedback

This structure provides a solid foundation for groundwater quality assessment and can be easily extended for additional environmental monitoring applications.