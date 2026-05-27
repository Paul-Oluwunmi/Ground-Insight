"""
Static Plots Module - Backward Compatibility Wrapper
=====================================================

This file provides backward compatibility for the original static_plots.py script.
All functionality has been moved to the modular static_plots package.

USAGE:
    from src.plotting.static_plots import plot_interactive_hydrograph
    fig = plot_interactive_hydrograph(data, well_columns)
"""

from .static_plots import plot_interactive_hydrograph

__all__ = ['plot_interactive_hydrograph']
