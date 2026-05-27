"""
Julian Plot Module
==================

This module provides Julian-Day plot visualization for groundwater data.

Main Functions:
--------------
- run_julian_plot: Main entry point for creating and displaying Julian-Day plots
"""

from .plotting import create_julian_plot


def run_julian_plot(data, well_columns):
    """
    Main function to run the Julian plot visualization.
    
    Parameters:
    -----------
    data : DataFrame
        Data containing DateTime and well columns
    well_columns : list
        List of well column names (not used directly, but kept for API consistency)
    """
    try:
        fig = create_julian_plot(data)
        if fig is not None:
            fig.show()
    except Exception as e:
        print(f"Error running Julian plot: {str(e)}")


__all__ = ['run_julian_plot', 'create_julian_plot']

