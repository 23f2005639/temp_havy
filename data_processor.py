"""
Data Processor Module
This module handles data validation, cleaning, and preprocessing for HMPI analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

class DataProcessor:
    """Data processing and validation for heavy metal analysis"""
    
    def __init__(self):
        self.heavy_metals = ['Pb', 'Cd', 'Cr', 'Ni', 'Cu', 'Zn', 'Fe', 'Mn', 'As', 'Hg']
        self.required_columns = ['Sample_ID']
        
    def check_data_quality(self, data: pd.DataFrame) -> Dict[str, bool]:
        """
        Perform comprehensive data quality checks
        
        Args:
            data: Input DataFrame
            
        Returns:
            Dictionary with quality check results
        """
        checks = {}
        
        # Check if data is not empty
        checks["Data not empty"] = len(data) > 0
        
        # Check for required columns
        metal_cols = self._get_metal_columns(data)
        checks["Heavy metals present"] = len(metal_cols) > 0
        
        # Check for missing values
        if len(metal_cols) > 0:
            missing_pct = data[metal_cols].isnull().sum().sum() / (len(data) * len(metal_cols))
            checks["Missing values < 10%"] = missing_pct < 0.1
        else:
            checks["Missing values < 10%"] = False
        
        # Check for negative values
        if len(metal_cols) > 0:
            has_negative = (data[metal_cols] < 0).any().any()
            checks["No negative concentrations"] = not has_negative
        else:
            checks["No negative concentrations"] = True
        
        # Check for unreasonably high values (> 1000 mg/L)
        if len(metal_cols) > 0:
            has_extreme = (data[metal_cols] > 1000).any().any()
            checks["No extreme values"] = not has_extreme
        else:
            checks["No extreme values"] = True
        
        # Check for duplicate samples
        if 'Sample_ID' in data.columns:
            has_duplicates = data['Sample_ID'].duplicated().any()
            checks["No duplicate sample IDs"] = not has_duplicates
        else:
            checks["No duplicate sample IDs"] = True
        
        # Check coordinate validity if present
        if 'latitude' in data.columns and 'longitude' in data.columns:
            lat_valid = data['latitude'].between(-90, 90).all()
            lon_valid = data['longitude'].between(-180, 180).all()
            checks["Valid coordinates"] = lat_valid and lon_valid
        else:
            checks["Valid coordinates"] = True
        
        return checks
    
    def clean_data(self, data: pd.DataFrame, 
                  remove_outliers: bool = True,
                  outlier_method: str = 'iqr') -> pd.DataFrame:
        """
        Clean and preprocess the data
        
        Args:
            data: Input DataFrame
            remove_outliers: Whether to remove outliers
            outlier_method: Method for outlier detection ('iqr' or 'zscore')
            
        Returns:
            Cleaned DataFrame
        """
        cleaned_data = data.copy()
        
        # Get metal columns
        metal_cols = self._get_metal_columns(cleaned_data)
        
        if not metal_cols:
            return cleaned_data
        
        # Handle missing values
        cleaned_data = self._handle_missing_values(cleaned_data, metal_cols)
        
        # Remove negative values (set to 0 or detection limit)
        for col in metal_cols:
            cleaned_data.loc[cleaned_data[col] < 0, col] = 0
        
        # Remove outliers if requested
        if remove_outliers:
            cleaned_data = self._remove_outliers(cleaned_data, metal_cols, method=outlier_method)
        
        # Standardize sample IDs if missing
        if 'Sample_ID' not in cleaned_data.columns:
            cleaned_data['Sample_ID'] = [f"Sample_{i+1:03d}" for i in range(len(cleaned_data))]
        
        return cleaned_data
    
    def _get_metal_columns(self, data: pd.DataFrame) -> List[str]:
        """Get list of heavy metal columns in the data"""
        metal_cols = []
        for col in data.columns:
            if col in self.heavy_metals or any(metal.lower() in col.lower() for metal in self.heavy_metals):
                metal_cols.append(col)
        return metal_cols
    
    def _handle_missing_values(self, data: pd.DataFrame, metal_cols: List[str]) -> pd.DataFrame:
        """Handle missing values in metal concentration columns"""
        cleaned_data = data.copy()
        
        for col in metal_cols:
            # Calculate detection limit (half of minimum non-zero value)
            non_zero_values = cleaned_data[col][cleaned_data[col] > 0]
            if len(non_zero_values) > 0:
                detection_limit = non_zero_values.min() / 2
                cleaned_data[col].fillna(detection_limit, inplace=True)
            else:
                # If all values are zero/missing, use a very small default
                cleaned_data[col].fillna(0.001, inplace=True)
        
        return cleaned_data
    
    def _remove_outliers(self, data: pd.DataFrame, 
                        metal_cols: List[str], 
                        method: str = 'iqr') -> pd.DataFrame:
        """Remove outliers from the dataset"""
        cleaned_data = data.copy()
        
        for col in metal_cols:
            if method == 'iqr':
                # Interquartile Range method
                Q1 = cleaned_data[col].quantile(0.25)
                Q3 = cleaned_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap values instead of removing rows
                cleaned_data.loc[cleaned_data[col] < lower_bound, col] = lower_bound
                cleaned_data.loc[cleaned_data[col] > upper_bound, col] = upper_bound
                
            elif method == 'zscore':
                # Z-score method (3 standard deviations)
                mean_val = cleaned_data[col].mean()
                std_val = cleaned_data[col].std()
                z_threshold = 3
                
                lower_bound = mean_val - z_threshold * std_val
                upper_bound = mean_val + z_threshold * std_val
                
                # Cap values instead of removing rows
                cleaned_data.loc[cleaned_data[col] < lower_bound, col] = max(0, lower_bound)
                cleaned_data.loc[cleaned_data[col] > upper_bound, col] = upper_bound
        
        return cleaned_data
    
    def validate_coordinates(self, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate geographic coordinates
        
        Args:
            data: DataFrame with potential coordinate columns
            
        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []
        is_valid = True
        
        if 'latitude' in data.columns:
            invalid_lat = ~data['latitude'].between(-90, 90)
            if invalid_lat.any():
                is_valid = False
                issues.append(f"Invalid latitude values found in {invalid_lat.sum()} rows")
        
        if 'longitude' in data.columns:
            invalid_lon = ~data['longitude'].between(-180, 180)
            if invalid_lon.any():
                is_valid = False
                issues.append(f"Invalid longitude values found in {invalid_lon.sum()} rows")
        
        return is_valid, issues
    
    def convert_units(self, data: pd.DataFrame, 
                     from_unit: str, 
                     to_unit: str = 'mg/L') -> pd.DataFrame:
        """
        Convert concentration units
        
        Args:
            data: DataFrame with concentration data
            from_unit: Source unit ('μg/L', 'ppb', 'ppm', etc.)
            to_unit: Target unit (default: 'mg/L')
            
        Returns:
            DataFrame with converted units
        """
        converted_data = data.copy()
        metal_cols = self._get_metal_columns(converted_data)
        
        # Conversion factors to mg/L
        conversion_factors = {
            'μg/L': 0.001,
            'ug/L': 0.001,
            'ppb': 0.001,
            'ng/L': 0.000001,
            'ppm': 1.0,
            'mg/L': 1.0,
            'g/L': 1000.0
        }
        
        if from_unit not in conversion_factors:
            raise ValueError(f"Unsupported unit: {from_unit}")
        
        if to_unit not in conversion_factors:
            raise ValueError(f"Unsupported unit: {to_unit}")
        
        # Calculate conversion factor
        factor = conversion_factors[from_unit] / conversion_factors[to_unit]
        
        # Apply conversion to metal columns
        for col in metal_cols:
            converted_data[col] = converted_data[col] * factor
        
        return converted_data
    
    def detect_metal_columns(self, data: pd.DataFrame) -> Dict[str, str]:
        """
        Automatically detect heavy metal columns in the dataset
        
        Args:
            data: Input DataFrame
            
        Returns:
            Dictionary mapping standard metal names to detected column names
        """
        column_mapping = {}
        
        for metal in self.heavy_metals:
            # Look for exact matches (case-insensitive)
            exact_matches = [col for col in data.columns if col.upper() == metal.upper()]
            if exact_matches:
                column_mapping[metal] = exact_matches[0]
                continue
            
            # Look for partial matches
            partial_matches = [col for col in data.columns 
                             if metal.lower() in col.lower() or col.lower() in metal.lower()]
            if partial_matches:
                # Choose the shortest match (likely the most specific)
                column_mapping[metal] = min(partial_matches, key=len)
        
        return column_mapping
    
    def get_data_summary(self, data: pd.DataFrame) -> Dict:
        """
        Generate a comprehensive summary of the dataset
        
        Args:
            data: Input DataFrame
            
        Returns:
            Dictionary with data summary statistics
        """
        metal_cols = self._get_metal_columns(data)
        
        summary = {
            'total_samples': len(data),
            'total_columns': len(data.columns),
            'metal_columns': len(metal_cols),
            'metal_names': metal_cols,
            'has_coordinates': 'latitude' in data.columns and 'longitude' in data.columns,
            'has_sample_ids': 'Sample_ID' in data.columns
        }
        
        if metal_cols:
            # Statistical summary for metals
            summary['concentration_stats'] = {}
            for metal in metal_cols:
                summary['concentration_stats'][metal] = {
                    'mean': float(data[metal].mean()),
                    'median': float(data[metal].median()),
                    'std': float(data[metal].std()),
                    'min': float(data[metal].min()),
                    'max': float(data[metal].max()),
                    'missing_count': int(data[metal].isnull().sum()),
                    'zero_count': int((data[metal] == 0).sum())
                }
        
        return summary
    
    def create_data_profile(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create a data profiling report
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with profiling information
        """
        profile_data = []
        
        for col in data.columns:
            if data[col].dtype in ['int64', 'float64']:
                profile_info = {
                    'Column': col,
                    'Type': 'Numeric',
                    'Count': len(data[col]),
                    'Missing': data[col].isnull().sum(),
                    'Missing_%': (data[col].isnull().sum() / len(data)) * 100,
                    'Mean': data[col].mean() if pd.api.types.is_numeric_dtype(data[col]) else None,
                    'Std': data[col].std() if pd.api.types.is_numeric_dtype(data[col]) else None,
                    'Min': data[col].min(),
                    'Max': data[col].max(),
                    'Zeros': (data[col] == 0).sum() if pd.api.types.is_numeric_dtype(data[col]) else 0
                }
            else:
                profile_info = {
                    'Column': col,
                    'Type': 'Text',
                    'Count': len(data[col]),
                    'Missing': data[col].isnull().sum(),
                    'Missing_%': (data[col].isnull().sum() / len(data)) * 100,
                    'Mean': None,
                    'Std': None,
                    'Min': None,
                    'Max': None,
                    'Zeros': 0
                }
            
            profile_data.append(profile_info)
        
        return pd.DataFrame(profile_data)
    
    def suggest_data_improvements(self, data: pd.DataFrame) -> List[str]:
        """
        Suggest improvements for data quality
        
        Args:
            data: Input DataFrame
            
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        quality_checks = self.check_data_quality(data)
        
        if not quality_checks["Heavy metals present"]:
            suggestions.append("Add heavy metal concentration columns (Pb, Cd, Cr, Ni, Cu, Zn, Fe, Mn, As, Hg)")
        
        if not quality_checks["Missing values < 10%"]:
            suggestions.append("Reduce missing values by improving sampling or using detection limit values")
        
        if not quality_checks["No negative concentrations"]:
            suggestions.append("Check and correct negative concentration values")
        
        if not quality_checks["No extreme values"]:
            suggestions.append("Review extremely high concentration values for data entry errors")
        
        if not quality_checks["No duplicate sample IDs"]:
            suggestions.append("Remove or rename duplicate sample IDs")
        
        if not quality_checks["Valid coordinates"]:
            suggestions.append("Verify and correct geographic coordinates")
        
        # Additional suggestions based on data characteristics
        metal_cols = self._get_metal_columns(data)
        
        if len(metal_cols) < 4:
            suggestions.append("Consider adding more heavy metals for comprehensive assessment")
        
        if 'latitude' not in data.columns or 'longitude' not in data.columns:
            suggestions.append("Add geographic coordinates for spatial analysis")
        
        if len(data) < 10:
            suggestions.append("Increase sample size for more reliable statistical analysis")
        
        return suggestions