"""
HMPI Calculator Module
This module contains functions for calculating various heavy metal pollution indices.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple

class HMPICalculator:
    """Calculator for Heavy Metal Pollution Indices"""
    
    def __init__(self):
        self.k_constant = 1.0  # Constant of proportionality
        
    def calculate_hmpi(self, data: pd.DataFrame, standards: Dict[str, float]) -> pd.DataFrame:
        """
        Calculate Heavy Metal Pollution Index (HMPI)
        
        Enhanced HMPI calculation with toxicity weighting factors based on research
        HMPI = Σ(Wi × Qi) / Σ(Wi)
        Where:
        - Wi = (K / Si) × TFi (unit weight with toxicity factor)
        - Qi = (Ci / Si) × 100 (sub-index)
        - K = constant of proportionality
        - Si = standard permissible value
        - Ci = observed concentration
        - TFi = toxicity factor based on health effects
        
        Args:
            data: DataFrame with heavy metal concentrations
            standards: Dictionary of standard values for each metal
            
        Returns:
            DataFrame with HMPI values and categories
        """
        results = data.copy()
        
        # Toxicity factors based on health impact research
        # Higher values indicate more toxic metals (WHO/EPA risk assessment data)
        toxicity_factors = {
            'Pb': 5.0,    # Highly toxic, especially to children (neurotoxicity)
            'Cd': 5.0,    # Highly toxic, carcinogenic, kidney damage
            'Cr': 4.5,    # Carcinogenic (Cr VI), skin/respiratory effects
            'As': 5.0,    # Highly toxic, carcinogenic, multi-organ effects
            'Hg': 4.8,    # Highly toxic, neurotoxicity, bioaccumulation
            'Ni': 3.5,    # Moderately toxic, carcinogenic, allergenic
            'Cu': 2.0,    # Less toxic at low levels, essential element
            'Zn': 1.5,    # Low toxicity, essential element
            'Fe': 1.0,    # Low toxicity, essential element
            'Mn': 2.5     # Moderate toxicity, neurotoxicity at high levels
        }
        
        # Get heavy metal columns that exist in both data and standards
        metal_cols = [col for col in data.columns if col in standards.keys()]
        
        if not metal_cols:
            raise ValueError("No matching heavy metals found in data and standards")
        
        # Calculate Wi and Qi for each metal with toxicity weighting
        wi_total = 0
        wi_qi_sum = 0
        
        for metal in metal_cols:
            if metal in results.columns and metal in standards:
                # Get toxicity factor (default to 2.0 if not specified)
                tf = toxicity_factors.get(metal, 2.0)
                
                # Calculate Wi = (K / Si) × TF (enhanced with toxicity factor)
                wi = (self.k_constant / standards[metal]) * tf
                
                # Calculate Qi = (Ci / Si) × 100
                qi = (results[metal] / standards[metal]) * 100
                
                # Store intermediate calculations
                results[f'{metal}_Wi'] = wi
                results[f'{metal}_Qi'] = qi
                results[f'{metal}_WiQi'] = wi * qi
                results[f'{metal}_TF'] = tf
                
                # Sum for final HMPI calculation
                wi_total += wi
                wi_qi_sum += wi * qi
        
        # Calculate HMPI = Σ(Wi × Qi) / Σ(Wi)
        results['HMPI'] = wi_qi_sum / wi_total if wi_total > 0 else 0
        
        # Categorize pollution levels with enhanced thresholds
        results['HMPI_Category'] = results['HMPI'].apply(self._categorize_hmpi)
        
        # Add risk assessment based on dominant contamination
        results['Health_Risk'] = results['HMPI'].apply(self._assess_health_risk)
        
        # Calculate metal-specific risk contributions
        for metal in metal_cols:
            if f'{metal}_WiQi' in results.columns:
                results[f'{metal}_Risk_Contribution'] = (results[f'{metal}_WiQi'] / results['HMPI']) * 100
        
        return results
    
    def calculate_hei(self, data: pd.DataFrame, standards: Dict[str, float]) -> pd.Series:
        """
        Calculate Heavy Metal Evaluation Index (HEI)
        
        HEI = Σ(Ci / Si) / n
        Where n is the number of metals
        
        Args:
            data: DataFrame with heavy metal concentrations
            standards: Dictionary of standard values for each metal
            
        Returns:
            Series with HEI values
        """
        metal_cols = [col for col in data.columns if col in standards.keys()]
        
        if not metal_cols:
            return pd.Series([0] * len(data))
        
        hei_sum = 0
        for metal in metal_cols:
            if metal in data.columns and metal in standards:
                hei_sum += data[metal] / standards[metal]
        
        return hei_sum / len(metal_cols)
    
    def calculate_contamination_index(self, data: pd.DataFrame, standards: Dict[str, float]) -> pd.Series:
        """
        Calculate Contamination Index (Cd)
        
        Cd = Σ(CAF)
        Where CAF = Ci / Si (Contamination Factor)
        
        Args:
            data: DataFrame with heavy metal concentrations
            standards: Dictionary of standard values for each metal
            
        Returns:
            Series with Cd values
        """
        metal_cols = [col for col in data.columns if col in standards.keys()]
        
        if not metal_cols:
            return pd.Series([0] * len(data))
        
        cd_sum = 0
        for metal in metal_cols:
            if metal in data.columns and metal in standards:
                cd_sum += data[metal] / standards[metal]
        
        return cd_sum
    
    def calculate_pollution_load_index(self, data: pd.DataFrame, standards: Dict[str, float]) -> pd.Series:
        """
        Calculate Pollution Load Index (PLI)
        
        PLI = (CF₁ × CF₂ × ... × CFₙ)^(1/n)
        Where CF = Ci / Si (Contamination Factor)
        
        Args:
            data: DataFrame with heavy metal concentrations
            standards: Dictionary of standard values for each metal
            
        Returns:
            Series with PLI values
        """
        metal_cols = [col for col in data.columns if col in standards.keys()]
        
        if not metal_cols:
            return pd.Series([1] * len(data))
        
        pli_product = 1
        for metal in metal_cols:
            if metal in data.columns and metal in standards:
                cf = data[metal] / standards[metal]
                pli_product *= cf
        
        return pli_product ** (1 / len(metal_cols))
    
    def calculate_enrichment_factor(self, data: pd.DataFrame, 
                                  reference_metal: str = 'Fe',
                                  background_values: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """
        Calculate Enrichment Factor (EF)
        
        EF = (Ci/Cref)sample / (Ci/Cref)background
        
        Args:
            data: DataFrame with heavy metal concentrations
            reference_metal: Reference metal (usually Fe or Al)
            background_values: Background concentrations
            
        Returns:
            DataFrame with EF values
        """
        if background_values is None:
            # Default crustal abundance values (mg/kg)
            background_values = {
                'Pb': 20, 'Cd': 0.2, 'Cr': 100, 'Ni': 75,
                'Cu': 55, 'Zn': 70, 'Fe': 56300, 'Mn': 950,
                'As': 1.8, 'Hg': 0.08
            }
        
        results = data.copy()
        
        if reference_metal not in data.columns or reference_metal not in background_values:
            raise ValueError(f"Reference metal {reference_metal} not found")
        
        ref_bg = background_values[reference_metal]
        
        for metal in data.columns:
            if metal in background_values and metal != reference_metal:
                metal_bg = background_values[metal]
                
                # Calculate EF
                ef = (data[metal] / data[reference_metal]) / (metal_bg / ref_bg)
                results[f'{metal}_EF'] = ef
                
                # Categorize EF
                results[f'{metal}_EF_Category'] = ef.apply(self._categorize_ef)
        
        return results
    
    def calculate_all_indices(self, data: pd.DataFrame, 
                            standards: Dict[str, float],
                            include_hei: bool = True,
                            include_cd: bool = True,
                            include_pli: bool = True,
                            include_ef: bool = False) -> pd.DataFrame:
        """
        Calculate all pollution indices
        
        Args:
            data: DataFrame with heavy metal concentrations
            standards: Dictionary of standard values for each metal
            include_hei: Whether to calculate HEI
            include_cd: Whether to calculate Cd
            include_pli: Whether to calculate PLI
            include_ef: Whether to calculate EF
            
        Returns:
            DataFrame with all calculated indices
        """
        # Start with HMPI calculation (main index)
        results = self.calculate_hmpi(data, standards)
        
        # Add other indices if requested
        if include_hei:
            results['HEI'] = self.calculate_hei(data, standards)
            results['HEI_Category'] = results['HEI'].apply(self._categorize_hei)
        
        if include_cd:
            results['Cd'] = self.calculate_contamination_index(data, standards)
            results['Cd_Category'] = results['Cd'].apply(self._categorize_cd)
        
        if include_pli:
            results['PLI'] = self.calculate_pollution_load_index(data, standards)
            results['PLI_Category'] = results['PLI'].apply(self._categorize_pli)
        
        if include_ef and 'Fe' in data.columns:
            ef_results = self.calculate_enrichment_factor(data)
            # Merge EF results
            ef_cols = [col for col in ef_results.columns if '_EF' in col]
            for col in ef_cols:
                results[col] = ef_results[col]
        
        return results
    
    def _categorize_hmpi(self, hmpi_value: float) -> str:
        """Categorize HMPI values"""
        if hmpi_value < 30:
            return "Low"
        elif hmpi_value < 50:
            return "Medium"
        elif hmpi_value < 100:
            return "High"
        else:
            return "Very High"
    
    def _categorize_hei(self, hei_value: float) -> str:
        """Categorize HEI values"""
        if hei_value < 10:
            return "Low"
        elif hei_value < 20:
            return "Medium"
        elif hei_value < 40:
            return "High"
        else:
            return "Very High"
    
    def _categorize_cd(self, cd_value: float) -> str:
        """Categorize Contamination Index values"""
        if cd_value < 3:
            return "Low"
        elif cd_value < 6:
            return "Medium"
        elif cd_value < 12:
            return "High"
        else:
            return "Very High"
    
    def _categorize_pli(self, pli_value: float) -> str:
        """Categorize PLI values"""
        if pli_value < 1:
            return "No Pollution"
        elif pli_value < 2:
            return "Low"
        elif pli_value < 3:
            return "Medium"
        else:
            return "High"
    
    def _categorize_ef(self, ef_value: float) -> str:
        """Categorize Enrichment Factor values"""
        if ef_value < 2:
            return "Minimal"
        elif ef_value < 5:
            return "Moderate"
        elif ef_value < 20:
            return "Significant"
        elif ef_value < 40:
            return "Very High"
        else:
            return "Extremely High"
    
    def get_pollution_summary(self, results: pd.DataFrame) -> Dict:
        """
        Generate a summary of pollution assessment
        
        Args:
            results: DataFrame with calculated indices
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'total_samples': len(results),
            'hmpi_stats': {
                'mean': results['HMPI'].mean(),
                'median': results['HMPI'].median(),
                'std': results['HMPI'].std(),
                'min': results['HMPI'].min(),
                'max': results['HMPI'].max()
            },
            'pollution_distribution': results['HMPI_Category'].value_counts().to_dict(),
            'contaminated_sites': len(results[results['HMPI'] > 100]),
            'safe_sites': len(results[results['HMPI'] < 30])
        }
        
        # Add other indices if available
        if 'HEI' in results.columns:
            summary['hei_stats'] = {
                'mean': results['HEI'].mean(),
                'max': results['HEI'].max()
            }
        
        if 'PLI' in results.columns:
            summary['pli_stats'] = {
                'mean': results['PLI'].mean(),
                'max': results['PLI'].max()
            }
        
        return summary
    
    def identify_critical_metals(self, data: pd.DataFrame, 
                               standards: Dict[str, float],
                               threshold: float = 1.0) -> Dict[str, int]:
        """
        Identify metals that frequently exceed standards
        
        Args:
            data: DataFrame with heavy metal concentrations
            standards: Dictionary of standard values
            threshold: Multiplier of standard (1.0 = exactly at standard)
            
        Returns:
            Dictionary with metal names and exceedance counts
        """
        exceedances = {}
        
        for metal in standards.keys():
            if metal in data.columns:
                exceeded = (data[metal] > standards[metal] * threshold).sum()
                exceedances[metal] = exceeded
        
        return exceedances
    
    def calculate_risk_assessment(self, results: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate risk levels based on multiple indices
        
        Args:
            results: DataFrame with calculated pollution indices
            
        Returns:
            DataFrame with overall risk assessment
        """
        risk_results = results.copy()
        
        # Initialize risk score
        risk_results['Risk_Score'] = 0
        
        # HMPI contribution (40% weight)
        hmpi_score = results['HMPI'].apply(lambda x: 
            1 if x < 30 else 2 if x < 50 else 3 if x < 100 else 4)
        risk_results['Risk_Score'] += hmpi_score * 0.4
        
        # HEI contribution (25% weight) if available
        if 'HEI' in results.columns:
            hei_score = results['HEI'].apply(lambda x:
                1 if x < 10 else 2 if x < 20 else 3 if x < 40 else 4)
            risk_results['Risk_Score'] += hei_score * 0.25
        
        # PLI contribution (25% weight) if available
        if 'PLI' in results.columns:
            pli_score = results['PLI'].apply(lambda x:
                1 if x < 1 else 2 if x < 2 else 3 if x < 3 else 4)
            risk_results['Risk_Score'] += pli_score * 0.25
        
        # Cd contribution (10% weight) if available
        if 'Cd' in results.columns:
            cd_score = results['Cd'].apply(lambda x:
                1 if x < 3 else 2 if x < 6 else 3 if x < 12 else 4)
            risk_results['Risk_Score'] += cd_score * 0.1
        
        # Categorize overall risk
        risk_results['Risk_Level'] = risk_results['Risk_Score'].apply(self._categorize_risk)
        
        return risk_results
    
    def _categorize_risk(self, risk_score: float) -> str:
        """Categorize overall risk level"""
        if risk_score < 1.5:
            return "Very Low"
        elif risk_score < 2.5:
            return "Low"
        elif risk_score < 3.5:
            return "High"
        else:
            return "Very High"
    
    def _assess_health_risk(self, hmpi_value: float) -> str:
        """
        Assess health risk based on HMPI value with enhanced categories
        Based on WHO risk assessment guidelines
        """
        if hmpi_value < 15:
            return "Minimal Risk"
        elif hmpi_value < 30:
            return "Low Risk"
        elif hmpi_value < 50:
            return "Moderate Risk"
        elif hmpi_value < 100:
            return "High Risk"
        elif hmpi_value < 200:
            return "Very High Risk"
        else:
            return "Extreme Risk"