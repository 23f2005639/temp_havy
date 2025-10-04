"""
Configuration file for HMPI Calculator Application
"""

# Application Settings
APP_TITLE = "Heavy Metal Pollution Index Calculator"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = "Automated Assessment of Groundwater Heavy Metal Contamination"

# Water Quality Standards (mg/L)
WATER_QUALITY_STANDARDS = {
    "WHO Guidelines": {
        'Pb': 0.01,   # Lead
        'Cd': 0.003,  # Cadmium  
        'Cr': 0.05,   # Chromium
        'Ni': 0.07,   # Nickel
        'Cu': 2.0,    # Copper
        'Zn': 3.0,    # Zinc
        'Fe': 0.3,    # Iron
        'Mn': 0.4,    # Manganese
        'As': 0.01,   # Arsenic
        'Hg': 0.006   # Mercury
    },
    "EPA Standards": {
        'Pb': 0.015,
        'Cd': 0.005,
        'Cr': 0.1,
        'Ni': 0.1,
        'Cu': 1.3,
        'Zn': 5.0,
        'Fe': 0.3,
        'Mn': 0.05,
        'As': 0.01,
        'Hg': 0.002
    },
    "IS 10500 (Indian)": {
        'Pb': 0.01,
        'Cd': 0.003,
        'Cr': 0.05,
        'Ni': 0.02,
        'Cu': 0.05,
        'Zn': 5.0,
        'Fe': 0.3,
        'Mn': 0.1,
        'As': 0.01,
        'Hg': 0.001
    }
}

# HMPI Classification Thresholds
HMPI_THRESHOLDS = {
    'low': 30,
    'medium': 50,
    'high': 100
}

# HEI Classification Thresholds
HEI_THRESHOLDS = {
    'low': 10,
    'medium': 20,
    'high': 40
}

# PLI Classification Thresholds
PLI_THRESHOLDS = {
    'no_pollution': 1,
    'low': 2,
    'medium': 3
}

# Contamination Index Thresholds
CD_THRESHOLDS = {
    'low': 3,
    'medium': 6,
    'high': 12
}

# Enrichment Factor Thresholds
EF_THRESHOLDS = {
    'minimal': 2,
    'moderate': 5,
    'significant': 20,
    'very_high': 40
}

# Heavy Metals Configuration
HEAVY_METALS = {
    'Pb': {'name': 'Lead', 'symbol': 'Pb', 'color': '#FF6B6B'},
    'Cd': {'name': 'Cadmium', 'symbol': 'Cd', 'color': '#4ECDC4'},
    'Cr': {'name': 'Chromium', 'symbol': 'Cr', 'color': '#45B7D1'},
    'Ni': {'name': 'Nickel', 'symbol': 'Ni', 'color': '#FFA07A'},
    'Cu': {'name': 'Copper', 'symbol': 'Cu', 'color': '#98D8C8'},
    'Zn': {'name': 'Zinc', 'symbol': 'Zn', 'color': '#F7DC6F'},
    'Fe': {'name': 'Iron', 'symbol': 'Fe', 'color': '#BB8FCE'},
    'Mn': {'name': 'Manganese', 'symbol': 'Mn', 'color': '#85C1E9'},
    'As': {'name': 'Arsenic', 'symbol': 'As', 'color': '#F8C471'},
    'Hg': {'name': 'Mercury', 'symbol': 'Hg', 'color': '#82E0AA'}
}

# Crustal Abundance Values (mg/kg) for Enrichment Factor calculation
CRUSTAL_ABUNDANCE = {
    'Pb': 20,
    'Cd': 0.2,
    'Cr': 100,
    'Ni': 75,
    'Cu': 55,
    'Zn': 70,
    'Fe': 56300,
    'Mn': 950,
    'As': 1.8,
    'Hg': 0.08
}

# Color Schemes for Visualization
COLOR_SCHEMES = {
    'pollution_levels': {
        'Low': '#2ECC71',      # Green
        'Medium': '#F1C40F',    # Yellow
        'High': '#E67E22',      # Orange
        'Very High': '#E74C3C'  # Red
    },
    'sequential': ['#2ECC71', '#F1C40F', '#E67E22', '#E74C3C'],
    'metal_colors': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', 
                    '#F7DC6F', '#BB8FCE', '#85C1E9', '#F8C471', '#82E0AA']
}

# File Upload Configuration
MAX_FILE_SIZE_MB = 50
SUPPORTED_FILE_TYPES = ['csv', 'xlsx', 'xls']

# Calculation Parameters
HMPI_K_CONSTANT = 1.0  # Constant of proportionality for HMPI calculation

# Default Sample Data Configuration
DEFAULT_SAMPLE_SIZE = 20
DEFAULT_COORDINATE_RANGE = {
    'latitude': {'min': 20.0, 'max': 25.0},
    'longitude': {'min': 75.0, 'max': 80.0}
}

# Data Validation Settings
DATA_VALIDATION = {
    'max_concentration': 1000.0,  # mg/L
    'min_concentration': 0.0,
    'max_missing_percentage': 10.0,  # %
    'coordinate_precision': 6  # decimal places
}

# Export Settings
EXPORT_FORMATS = ['CSV', 'Excel']
REPORT_TEMPLATE = {
    'include_summary': True,
    'include_methodology': True,
    'include_recommendations': True,
    'include_references': True
}

# UI Configuration
SIDEBAR_WIDTH = 300
CHART_HEIGHT = 500
MAP_HEIGHT = 600

# Help Text and Messages
HELP_TEXT = {
    'hmpi_explanation': """
    The Heavy Metal Pollution Index (HMPI) is calculated using the formula:
    HMPI = Σ(Wi × Qi) / Σ(Wi)
    
    Where:
    - Wi = Unit weight (K/Si)
    - Qi = Sub-index ((Ci/Si) × 100)
    - K = Constant of proportionality
    - Si = Standard permissible value
    - Ci = Observed concentration
    """,
    
    'data_format': """
    Required data format:
    - Heavy metal concentrations in mg/L or μg/L
    - Optional: Sample IDs, geographic coordinates
    - Supported metals: Pb, Cd, Cr, Ni, Cu, Zn, Fe, Mn, As, Hg
    """,
    
    'classification': """
    HMPI Classification:
    - < 30: Low pollution (suitable for drinking)
    - 30-50: Medium pollution (basic treatment needed)
    - 50-100: High pollution (advanced treatment required)
    - ≥ 100: Very high pollution (unsuitable for drinking)
    """
}

# Application Metadata
METADATA = {
    'author': 'Environmental Data Analysis Team',
    'email': 'contact@hmpi-calculator.org',
    'license': 'MIT',
    'repository': 'https://github.com/hmpi-calculator',
    'documentation': 'https://hmpi-calculator.readthedocs.io'
}