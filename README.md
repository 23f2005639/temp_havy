# Heavy Metal Pollution Index (HMPI) Calculator

A comprehensive Streamlit application for calculating and analyzing heavy metal pollution indices in groundwater samples.

## Features

- **Automated HMPI Calculation**: Calculate Heavy Metal Pollution Index using standard formulas
- **Multiple Indices**: Support for HEI, Cd, PLI, and EF calculations
- **Interactive Visualizations**: Maps, charts, and statistical summaries
- **Data Quality Checks**: Automated validation and cleaning
- **Export Functionality**: Download results in CSV/Excel format
- **Multiple Standards**: WHO, EPA, IS 10500, and custom standards

## Installation

1. **Clone or download this repository**
   ```bash
   cd hmpi_app
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser** and go to `http://localhost:8501`

## Usage

### 1. Data Input
- **Upload File**: CSV or Excel files with heavy metal concentrations
- **Manual Entry**: Enter data directly through the web interface
- **Sample Data**: Use provided sample dataset for testing

### 2. Data Format
Your data should include:
- **Required**: Heavy metal concentrations (mg/L or μg/L)
- **Optional**: Sample IDs, latitude, longitude coordinates

Supported heavy metals:
- Pb (Lead)
- Cd (Cadmium) 
- Cr (Chromium)
- Ni (Nickel)
- Cu (Copper)
- Zn (Zinc)
- Fe (Iron)
- Mn (Manganese)
- As (Arsenic)
- Hg (Mercury)

### 3. Analysis Configuration
- Select water quality standards (WHO, EPA, IS 10500, or custom)
- Choose which indices to calculate
- Configure analysis parameters

### 4. Results and Visualization
- Interactive maps showing pollution distribution
- Statistical charts and summaries
- Correlation analysis
- Export results and generate reports

## HMPI Calculation

The Heavy Metal Pollution Index is calculated using the formula:

**HMPI = Σ(Wi × Qi) / Σ(Wi)**

Where:
- **Wi** = Unit weight = K / Si
- **Qi** = Sub-index = (Ci / Si) × 100
- **K** = Constant of proportionality (1.0)
- **Si** = Standard permissible value
- **Ci** = Observed concentration

### Classification:
- **HMPI < 30**: Low pollution
- **30 ≤ HMPI < 50**: Medium pollution
- **50 ≤ HMPI < 100**: High pollution
- **HMPI ≥ 100**: Very high pollution (unsuitable for drinking)

## Other Supported Indices

### Heavy Metal Evaluation Index (HEI)
**HEI = Σ(Ci / Si) / n**

### Contamination Index (Cd)
**Cd = Σ(CAF)** where CAF = Ci / Si

### Pollution Load Index (PLI)
**PLI = (CF₁ × CF₂ × ... × CFₙ)^(1/n)** where CF = Ci / Si

## Sample Data

The application includes sample groundwater data from 20 monitoring wells with:
- Geographic coordinates (latitude/longitude)
- Concentrations for 8 heavy metals
- Realistic pollution scenarios for testing

## Water Quality Standards

### WHO Guidelines (default)
- Pb: 0.01 mg/L
- Cd: 0.003 mg/L
- Cr: 0.05 mg/L
- Ni: 0.07 mg/L
- Cu: 2.0 mg/L
- Zn: 3.0 mg/L
- Fe: 0.3 mg/L
- Mn: 0.4 mg/L

### EPA Standards
- Similar parameters with slight variations

### IS 10500 (Indian Standard)
- Adapted for Indian groundwater conditions

## File Structure

```
hmpi_app/
├── app.py                 # Main Streamlit application
├── hmpi_calculator.py     # HMPI calculation functions
├── data_processor.py      # Data validation and processing
├── visualization.py       # Charts and maps
├── requirements.txt       # Python dependencies
├── sample_data.csv       # Sample dataset
└── README.md             # This file
```

## Dependencies

- **streamlit**: Web application framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **plotly**: Interactive visualizations
- **folium**: Map visualizations
- **openpyxl**: Excel file support

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **Data format issues**: Check that your CSV/Excel file has:
   - Proper column headers
   - Numeric values for concentrations
   - No missing required data

3. **Map not showing**: Ensure latitude/longitude columns are present and contain valid coordinates (-90 to 90 for latitude, -180 to 180 for longitude)

### Data Requirements

- Concentrations should be positive numbers
- Geographic coordinates are optional but recommended for mapping
- Sample IDs should be unique
- At least 3-4 heavy metals recommended for meaningful analysis

## Contributing

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## References

1. Prasad, B., & Bose, J. M. (2001). Evaluation of the heavy metal pollution index for surface and spring water near a limestone mining area of the lower Himalayas.

2. Edet, A. E., & Offiong, O. E. (2002). Evaluation of water quality pollution indices for heavy metal contamination monitoring.

3. WHO. (2017). Guidelines for drinking-water quality: fourth edition incorporating the first addendum.

4. US EPA. (2018). National Primary Drinking Water Regulations.

## License

This project is open source and available under the MIT License.

## Support

For questions or issues, please:
1. Check this README file
2. Review the sample data format
3. Ensure all dependencies are properly installed
4. Check that your data meets the format requirements

## Version History

- **v1.0.0**: Initial release with basic HMPI calculation
- Features: Data input, HMPI calculation, basic visualization
- Support for multiple standards and export functionality