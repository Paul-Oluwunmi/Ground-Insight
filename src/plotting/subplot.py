"""
Subplot Module - Backward Compatibility Wrapper
=================================================

This file provides backward compatibility for the original subplot.py script.
All functionality has been moved to the modular subplot package.

USAGE:
    from src.plotting.subplot import run_subplot
    run_subplot(data, well_columns)
"""

from .subplot import run_subplot, create_subplots

__all__ = ['run_subplot', 'create_subplots']
