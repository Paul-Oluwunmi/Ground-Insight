"""
Julian Plot Module - Backward Compatibility Wrapper
====================================================

This file provides backward compatibility for the original julian_plot.py script.
All functionality has been moved to the modular julian_plot package.

USAGE:
    from src.plotting.julian_plot import run_julian_plot
    run_julian_plot(data, well_columns)
"""

# Import from the package directory
# Python gives precedence to packages over modules, so .julian_plot imports from julian_plot/ directory
from .julian_plot import run_julian_plot, create_julian_plot

__all__ = ['run_julian_plot', 'create_julian_plot']