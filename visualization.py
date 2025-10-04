"""
Visualization Module
This module contains functions for creating interactive charts and maps for HMPI analysis.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from typing import Dict, List, Optional
import streamlit as st

class Visualizer:
    """Visualization class for HMPI analysis results"""
    
    def __init__(self):
        self.color_scales = {
            'hmpi': ['green', 'yellow', 'orange', 'red'],
            'pollution': px.colors.sequential.Reds,
            'metal': px.colors.qualitative.Set3
        }
    
    def create_pollution_map(self, results: pd.DataFrame) -> go.Figure:
        """
        Create an enhanced interactive map showing pollution levels with research-grade visualization
        
        Args:
            results: DataFrame with HMPI results and coordinates
            
        Returns:
            Plotly Figure object
        """
        if 'latitude' not in results.columns or 'longitude' not in results.columns:
            return self._create_no_coordinates_figure()
        
        # Filter out rows with missing coordinates
        map_data = results.dropna(subset=['latitude', 'longitude'])
        
        if len(map_data) == 0:
            return self._create_no_coordinates_figure()
        
        # Enhanced color mapping with scientific color scheme
        color_map = {
            'Low': '#2E8B57',        # Sea Green
            'Medium': '#FFD700',     # Gold
            'High': '#FF6347',       # Tomato
            'Very High': '#DC143C'   # Crimson
        }
        
        # Create size mapping based on HMPI values (log scale for better visualization)
        map_data['size'] = np.log10(map_data['HMPI'] + 1) * 10 + 5
        
        # Enhanced hover information
        hover_data_list = ['HMPI', 'HMPI_Category']
        if 'Health_Risk' in map_data.columns:
            hover_data_list.append('Health_Risk')
        if 'Location_Type' in map_data.columns:
            hover_data_list.append('Location_Type')
        if 'Contamination_Source' in map_data.columns:
            hover_data_list.append('Contamination_Source')
        
        fig = px.scatter_mapbox(
            map_data,
            lat='latitude',
            lon='longitude',
            color='HMPI_Category',
            size='size',
            size_max=25,
            color_discrete_map=color_map,
            hover_name='Sample_ID',
            hover_data=hover_data_list,
            title='Groundwater Heavy Metal Contamination Assessment Map',
            mapbox_style='open-street-map',
            category_orders={'HMPI_Category': ['Low', 'Medium', 'High', 'Very High']}
        )
        
        # Enhanced layout with scientific styling
        fig.update_layout(
            height=650,
            margin=dict(l=0, r=0, t=70, b=0),
            title={
                'text': 'Groundwater Heavy Metal Contamination Assessment Map',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16, 'color': 'darkblue'}
            },
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
                title="Pollution Level"
            ),
            font=dict(family="Arial, sans-serif", size=12),
        )
        
        # Add annotations if needed
        if len(map_data) > 0:
            center_lat = map_data['latitude'].mean()
            center_lon = map_data['longitude'].mean()
            
            fig.update_layout(
                mapbox=dict(
                    center=dict(lat=center_lat, lon=center_lon),
                    zoom=6
                )
            )
        
        return fig
    
    def create_hmpi_histogram(self, results: pd.DataFrame) -> go.Figure:
        """
        Create histogram of HMPI values
        
        Args:
            results: DataFrame with HMPI results
            
        Returns:
            Plotly Figure object
        """
        fig = px.histogram(
            results,
            x='HMPI',
            nbins=20,
            title='Distribution of HMPI Values',
            color_discrete_sequence=['skyblue']
        )
        
        # Add vertical lines for category boundaries
        fig.add_vline(x=30, line_dash="dash", line_color="green", 
                     annotation_text="Low/Medium threshold")
        fig.add_vline(x=50, line_dash="dash", line_color="orange",
                     annotation_text="Medium/High threshold")
        fig.add_vline(x=100, line_dash="dash", line_color="red",
                     annotation_text="High/Very High threshold")
        
        fig.update_layout(
            xaxis_title="HMPI Value",
            yaxis_title="Frequency",
            showlegend=False
        )
        
        return fig
    
    def create_hmpi_boxplot(self, results: pd.DataFrame) -> go.Figure:
        """
        Create boxplot of HMPI values by category
        
        Args:
            results: DataFrame with HMPI results
            
        Returns:
            Plotly Figure object
        """
        fig = px.box(
            results,
            x='HMPI_Category',
            y='HMPI',
            title='HMPI Distribution by Pollution Category',
            color='HMPI_Category',
            color_discrete_map={
                'Low': 'green',
                'Medium': 'yellow',
                'High': 'orange', 
                'Very High': 'red'
            }
        )
        
        fig.update_layout(
            xaxis_title="Pollution Category",
            yaxis_title="HMPI Value",
            showlegend=False
        )
        
        return fig
    
    def create_metal_comparison(self, data: pd.DataFrame) -> go.Figure:
        """
        Create comparison chart for different heavy metals
        
        Args:
            data: DataFrame with heavy metal concentrations
            
        Returns:
            Plotly Figure object
        """
        # Get metal columns
        metal_cols = [col for col in data.columns 
                     if col in ['Pb', 'Cd', 'Cr', 'Ni', 'Cu', 'Zn', 'Fe', 'Mn', 'As', 'Hg']]
        
        if not metal_cols:
            return self._create_no_data_figure("No heavy metal data found")
        
        # Calculate mean concentrations
        mean_concentrations = data[metal_cols].mean()
        
        fig = px.bar(
            x=mean_concentrations.index,
            y=mean_concentrations.values,
            title='Average Heavy Metal Concentrations',
            labels={'x': 'Heavy Metals', 'y': 'Concentration (mg/L)'},
            color=mean_concentrations.values,
            color_continuous_scale='Reds'
        )
        
        fig.update_layout(
            xaxis_title="Heavy Metals",
            yaxis_title="Average Concentration (mg/L)",
            showlegend=False
        )
        
        return fig
    
    def create_correlation_heatmap(self, data: pd.DataFrame) -> go.Figure:
        """
        Create correlation heatmap for heavy metals
        
        Args:
            data: DataFrame with heavy metal concentrations
            
        Returns:
            Plotly Figure object
        """
        # Get numeric columns (heavy metals)
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove coordinate columns if present
        numeric_cols = [col for col in numeric_cols 
                       if col not in ['latitude', 'longitude']]
        
        if len(numeric_cols) < 2:
            return self._create_no_data_figure("Insufficient numeric data for correlation analysis")
        
        # Calculate correlation matrix
        correlation_matrix = data[numeric_cols].corr()
        
        fig = px.imshow(
            correlation_matrix,
            title='Heavy Metal Concentration Correlations',
            color_continuous_scale='RdBu',
            aspect='auto'
        )
        
        fig.update_layout(
            width=600,
            height=600
        )
        
        return fig
    
    def create_radar_chart(self, results: pd.DataFrame, 
                          sample_id: str) -> go.Figure:
        """
        Create radar chart for a specific sample
        
        Args:
            results: DataFrame with pollution indices
            sample_id: Sample ID to display
            
        Returns:
            Plotly Figure object
        """
        sample_data = results[results['Sample_ID'] == sample_id]
        
        if len(sample_data) == 0:
            return self._create_no_data_figure(f"Sample {sample_id} not found")
        
        sample = sample_data.iloc[0]
        
        # Prepare data for radar chart
        categories = []
        values = []
        
        if 'HMPI' in sample:
            categories.append('HMPI')
            values.append(min(sample['HMPI'], 200))  # Cap at 200 for visualization
        
        if 'HEI' in sample:
            categories.append('HEI')
            values.append(min(sample['HEI'], 100))  # Cap at 100
        
        if 'PLI' in sample:
            categories.append('PLI')
            values.append(min(sample['PLI'] * 50, 200))  # Scale and cap
        
        if 'Cd' in sample:
            categories.append('Cd')
            values.append(min(sample['Cd'] * 10, 200))  # Scale and cap
        
        if not categories:
            return self._create_no_data_figure("No pollution indices available")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=sample_id,
            line_color='red'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 200]
                )
            ),
            title=f'Pollution Indices Radar Chart - {sample_id}',
            showlegend=True
        )
        
        return fig
    
    def create_trend_analysis(self, results: pd.DataFrame,
                            date_column: Optional[str] = None) -> go.Figure:
        """
        Create trend analysis if temporal data is available
        
        Args:
            results: DataFrame with HMPI results
            date_column: Column name containing dates
            
        Returns:
            Plotly Figure object
        """
        if date_column is None or date_column not in results.columns:
            # Create sample-based trend (ordered by sample ID)
            results_sorted = results.sort_values('Sample_ID')
            
            fig = px.line(
                results_sorted,
                x='Sample_ID',
                y='HMPI',
                title='HMPI Values Across Samples',
                markers=True
            )
            
            # Add category threshold lines
            fig.add_hline(y=30, line_dash="dash", line_color="green",
                         annotation_text="Low/Medium threshold")
            fig.add_hline(y=50, line_dash="dash", line_color="orange",
                         annotation_text="Medium/High threshold")
            fig.add_hline(y=100, line_dash="dash", line_color="red",
                         annotation_text="High/Very High threshold")
            
        else:
            # Time-based trend analysis
            results_sorted = results.sort_values(date_column)
            
            fig = px.line(
                results_sorted,
                x=date_column,
                y='HMPI',
                title='HMPI Trend Over Time',
                markers=True
            )
        
        fig.update_layout(
            xaxis_title="Sample ID" if date_column is None else "Date",
            yaxis_title="HMPI Value"
        )
        
        return fig
    
    def create_multi_index_comparison(self, results: pd.DataFrame) -> go.Figure:
        """
        Create comparison chart for multiple pollution indices
        
        Args:
            results: DataFrame with multiple pollution indices
            
        Returns:
            Plotly Figure object
        """
        # Check which indices are available
        available_indices = []
        for index_col in ['HMPI', 'HEI', 'PLI', 'Cd']:
            if index_col in results.columns:
                available_indices.append(index_col)
        
        if len(available_indices) < 2:
            return self._create_no_data_figure("Need at least 2 pollution indices for comparison")
        
        # Normalize indices to 0-100 scale for comparison
        normalized_data = results[['Sample_ID'] + available_indices].copy()
        
        for col in available_indices:
            if col == 'HMPI':
                # HMPI is already on a suitable scale
                normalized_data[f'{col}_normalized'] = normalized_data[col]
            elif col == 'HEI':
                # Scale HEI to 0-100
                max_hei = normalized_data[col].max()
                normalized_data[f'{col}_normalized'] = (normalized_data[col] / max_hei) * 100
            elif col == 'PLI':
                # Scale PLI (typically 0-5) to 0-100
                normalized_data[f'{col}_normalized'] = normalized_data[col] * 20
            elif col == 'Cd':
                # Scale Cd to 0-100
                max_cd = normalized_data[col].max()
                normalized_data[f'{col}_normalized'] = (normalized_data[col] / max_cd) * 100
        
        # Create subplot
        fig = make_subplots(
            rows=1, cols=1,
            subplot_titles=['Pollution Indices Comparison (Normalized to 0-100 scale)']
        )
        
        # Add traces for each index
        colors = ['red', 'blue', 'green', 'orange']
        for i, col in enumerate(available_indices):
            fig.add_trace(
                go.Scatter(
                    x=normalized_data['Sample_ID'],
                    y=normalized_data[f'{col}_normalized'],
                    mode='lines+markers',
                    name=col,
                    line=dict(color=colors[i % len(colors)])
                )
            )
        
        fig.update_layout(
            title='Multi-Index Pollution Comparison',
            xaxis_title='Sample ID',
            yaxis_title='Normalized Index Value',
            height=500
        )
        
        return fig
    
    def show_statistical_summary(self, results: pd.DataFrame):
        """
        Display statistical summary using Streamlit
        
        Args:
            results: DataFrame with HMPI results
        """
        st.subheader("📊 Statistical Summary")
        
        # Basic statistics
        if 'HMPI' in results.columns:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Mean HMPI", f"{results['HMPI'].mean():.2f}")
            with col2:
                st.metric("Median HMPI", f"{results['HMPI'].median():.2f}")
            with col3:
                st.metric("Std Dev", f"{results['HMPI'].std():.2f}")
            with col4:
                st.metric("Range", f"{results['HMPI'].max() - results['HMPI'].min():.2f}")
        
        # Category distribution
        if 'HMPI_Category' in results.columns:
            st.subheader("Pollution Level Distribution")
            category_dist = results['HMPI_Category'].value_counts()
            
            for category, count in category_dist.items():
                percentage = (count / len(results)) * 100
                st.write(f"**{category}**: {count} samples ({percentage:.1f}%)")
        
        # Additional indices if available
        other_indices = [col for col in ['HEI', 'PLI', 'Cd'] if col in results.columns]
        
        if other_indices:
            st.subheader("Other Pollution Indices")
            
            summary_data = []
            for index in other_indices:
                summary_data.append({
                    'Index': index,
                    'Mean': results[index].mean(),
                    'Median': results[index].median(),
                    'Std Dev': results[index].std(),
                    'Min': results[index].min(),
                    'Max': results[index].max()
                })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df)
    
    def _create_no_data_figure(self, message: str) -> go.Figure:
        """Create a placeholder figure when no data is available"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            title="No Data Available",
            xaxis={'visible': False},
            yaxis={'visible': False},
            height=400
        )
        return fig
    
    def _create_no_coordinates_figure(self) -> go.Figure:
        """Create a placeholder figure when coordinates are not available"""
        fig = go.Figure()
        fig.add_annotation(
            text="Geographic coordinates not available in dataset.<br>Please add 'latitude' and 'longitude' columns for map visualization.",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=14)
        )
        fig.update_layout(
            title="Map Visualization Not Available",
            xaxis={'visible': False},
            yaxis={'visible': False},
            height=400
        )
        return fig
    
    def create_metal_contribution_chart(self, results: pd.DataFrame) -> go.Figure:
        """
        Create a stacked bar chart showing individual metal contributions to HMPI
        
        Args:
            results: DataFrame with HMPI results and metal risk contributions
            
        Returns:
            Plotly Figure object
        """
        # Find risk contribution columns
        contribution_cols = [col for col in results.columns if '_Risk_Contribution' in col]
        
        if not contribution_cols:
            return self._create_no_data_figure("No risk contribution data available")
        
        # Prepare data for stacked bar chart
        metals = [col.replace('_Risk_Contribution', '') for col in contribution_cols]
        
        fig = go.Figure()
        
        # Add bars for each metal
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', 
                 '#F7DC6F', '#BB8FCE', '#85C1E9', '#F8C471', '#82E0AA']
        
        for i, (metal, col) in enumerate(zip(metals, contribution_cols)):
            fig.add_trace(go.Bar(
                name=metal,
                x=results['Sample_ID'],
                y=results[col],
                marker_color=colors[i % len(colors)],
                hovertemplate=f'{metal}: %{{y:.1f}}%<extra></extra>'
            ))
        
        fig.update_layout(
            title='Heavy Metal Risk Contributions to HMPI by Sample',
            xaxis_title='Sample ID',
            yaxis_title='Risk Contribution (%)',
            barmode='stack',
            height=500,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            font=dict(family="Arial, sans-serif", size=11)
        )
        
        return fig
    
    def create_health_risk_assessment_chart(self, results: pd.DataFrame) -> go.Figure:
        """
        Create a comprehensive health risk assessment visualization
        
        Args:
            results: DataFrame with HMPI results and health risk data
            
        Returns:
            Plotly Figure object
        """
        if 'Health_Risk' not in results.columns:
            return self._create_no_data_figure("Health risk data not available")
        
        # Count samples by health risk category
        risk_counts = results['Health_Risk'].value_counts()
        
        # Define risk level colors
        risk_colors = {
            'Minimal Risk': '#2E8B57',     # Dark Sea Green
            'Low Risk': '#32CD32',         # Lime Green  
            'Moderate Risk': '#FFD700',    # Gold
            'High Risk': '#FF6347',        # Tomato
            'Very High Risk': '#DC143C',   # Crimson
            'Extreme Risk': '#800000'      # Maroon
        }
        
        # Create subplot with pie chart and bar chart
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Health Risk Distribution', 'Risk Level Analysis'),
            specs=[[{"type": "pie"}, {"type": "bar"}]]
        )
        
        # Pie chart
        fig.add_trace(
            go.Pie(
                labels=risk_counts.index,
                values=risk_counts.values,
                marker_colors=[risk_colors.get(risk, '#808080') for risk in risk_counts.index],
                name="Risk Distribution"
            ),
            row=1, col=1
        )
        
        # Bar chart
        fig.add_trace(
            go.Bar(
                x=risk_counts.index,
                y=risk_counts.values,
                marker_color=[risk_colors.get(risk, '#808080') for risk in risk_counts.index],
                name="Sample Count",
                showlegend=False
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title_text="Health Risk Assessment Summary",
            height=500,
            showlegend=False,
            font=dict(family="Arial, sans-serif", size=11)
        )
        
        return fig
    
    def create_contamination_source_analysis(self, results: pd.DataFrame) -> go.Figure:
        """
        Create analysis of contamination by source type
        
        Args:
            results: DataFrame with contamination source information
            
        Returns:
            Plotly Figure object
        """
        if 'Contamination_Source' not in results.columns:
            return self._create_no_data_figure("Contamination source data not available")
        
        # Group by contamination source and calculate average HMPI
        source_analysis = results.groupby('Contamination_Source').agg({
            'HMPI': ['mean', 'max', 'count'],
            'Health_Risk': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown'
        }).round(2)
        
        source_analysis.columns = ['Mean_HMPI', 'Max_HMPI', 'Sample_Count', 'Dominant_Risk']
        source_analysis = source_analysis.reset_index()
        
        # Create grouped bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Average HMPI',
            x=source_analysis['Contamination_Source'],
            y=source_analysis['Mean_HMPI'],
            yaxis='y',
            marker_color='lightblue',
            text=source_analysis['Mean_HMPI'],
            textposition='outside'
        ))
        
        fig.add_trace(go.Scatter(
            name='Sample Count',
            x=source_analysis['Contamination_Source'],
            y=source_analysis['Sample_Count'],
            yaxis='y2',
            mode='markers+lines',
            marker=dict(size=10, color='red'),
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title='Contamination Analysis by Source Type',
            xaxis_title='Contamination Source',
            yaxis=dict(
                title='Average HMPI',
                side='left'
            ),
            yaxis2=dict(
                title='Number of Samples',
                side='right',
                overlaying='y'
            ),
            height=500,
            font=dict(family="Arial, sans-serif", size=11)
        )
        
        return fig
    
    def create_sample_comparison_chart(self, results: pd.DataFrame,
                                     sample_ids: List[str]) -> go.Figure:
        """
        Create comparison chart for selected samples
        
        Args:
            results: DataFrame with HMPI results
            sample_ids: List of sample IDs to compare
            
        Returns:
            Plotly Figure object
        """
        # Filter data for selected samples
        selected_data = results[results['Sample_ID'].isin(sample_ids)]
        
        if len(selected_data) == 0:
            return self._create_no_data_figure("Selected samples not found")
        
        # Get available indices
        indices = [col for col in ['HMPI', 'HEI', 'PLI', 'Cd'] 
                  if col in selected_data.columns]
        
        if not indices:
            return self._create_no_data_figure("No pollution indices available")
        
        # Create grouped bar chart
        fig = go.Figure()
        
        colors = ['red', 'blue', 'green', 'orange']
        
        for i, index in enumerate(indices):
            fig.add_trace(go.Bar(
                name=index,
                x=selected_data['Sample_ID'],
                y=selected_data[index],
                marker_color=colors[i % len(colors)]
            ))
        
        fig.update_layout(
            title='Pollution Indices Comparison - Selected Samples',
            xaxis_title='Sample ID',
            yaxis_title='Index Value',
            barmode='group',
            height=500
        )
        
        return fig