"""
Wavelet Analysis Dashboard Layout
=================================

Defines the HTML layout for the wavelet analysis dashboard.
This file extracts the layout from the original Wavelet_Analysis.py file.
"""

import dash
from dash import dcc, html
import pywt
from ..constants import DEFAULT_WAVELET_TYPE, DEFAULT_WAVELET


def create_layout(well_options, data_type_options, wavelet_type_options, data):
    """
    Create the Dash application layout for the wavelet analysis dashboard.
    
    Parameters:
    -----------
    well_options : list
        List of well dropdown options
    data_type_options : list
        List of data type dropdown options
    wavelet_type_options : list
        List of wavelet type dropdown options
    data : pd.DataFrame
        The main dataset (for date range limits)
    
    Returns:
    --------
    html.Div
        The complete HTML layout for the dashboard.
    """
    # Extract layout from the monolithic source (lines 163-497).
    # Prefer the copy bundled in this package; fall back to env var / legacy paths.
    import os
    
    _here = os.path.dirname(os.path.abspath(__file__))
    _candidates = [
        os.path.normpath(os.path.join(_here, '..', '_wavelet_analysis_full.py')),
        os.environ.get('WAVELET_ANALYSIS_FILE', ''),
        r'c:\Users\oluwunmi\Downloads\SCox_update\groundwater_module\src\plotting\Wavelet_Analysis.py',
        r'c:\Users\oluwunmi\Downloads\groundwater_module\src\plotting\Wavelet_Analysis.py',
    ]
    _original_file = next((p for p in _candidates if p and os.path.exists(p)), None)
    
    if _original_file:
        # Read original file
        with open(_original_file, 'r', encoding='utf-8') as f:
            original_lines = f.readlines()
        
        # Extract layout code (lines 163-497, 0-based: 162-496)
        layout_lines = original_lines[162:497]
        layout_code = ''.join(layout_lines)
        
        # Extract just the html.Div expression (skip "app.layout = ")
        # Find the start of html.Div([
        div_start = layout_code.find('html.Div([')
        if div_start == -1:
            raise ValueError("Could not find html.Div in layout code")
        
        # Extract from html.Div([ to the matching closing ])
        # We need to find the matching closing bracket
        div_code = layout_code[div_start:]
        
        # Replace variable references
        div_code = div_code.replace('well_columns[0]', 'well_options[0]["value"] if well_options else None')
        div_code = div_code.replace('default_wavelet_type', 'DEFAULT_WAVELET_TYPE')
        div_code = div_code.replace('default_wavelet', 'DEFAULT_WAVELET')
        
        # Create namespace with necessary imports
        namespace = {
            'html': html,
            'dcc': dcc,
            'pywt': pywt,
            'data': data,
            'well_options': well_options,
            'data_type_options': data_type_options,
            'wavelet_type_options': wavelet_type_options,
            'DEFAULT_WAVELET_TYPE': DEFAULT_WAVELET_TYPE,
            'DEFAULT_WAVELET': DEFAULT_WAVELET,
        }
        
        # Import dash_table if needed
        try:
            from dash import dash_table
            namespace['dash_table'] = dash_table
        except:
            pass
        
        # Execute layout code
        try:
            layout_result = eval(div_code, namespace)
            return layout_result
        except Exception as e:
            import traceback
            print(f"Error extracting layout: {str(e)}")
            print(traceback.format_exc())
            # Fallback: return a simple layout
            return html.Div([
                html.H1("Wavelet Analysis Dashboard"),
                html.Div("Error loading layout. Please check the original file.")
            ])
    
    # Fallback layout if original file not found
    return html.Div([
        html.H1("Wavelet Analysis Dashboard"),
        html.Div("Layout file not found.")
    ])

