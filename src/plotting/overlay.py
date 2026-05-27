"""
Overlay Module - Backward Compatibility Wrapper
================================================

This file provides backward compatibility for the original overlay.py script.
All functionality has been moved to the modular overlay package.

USAGE:
    from src.plotting.overlay import run_overlay
    run_overlay(data, well_columns, data_folder, rainfall_filename)
"""

from .overlay import run_overlay, create_interactive_overlay

__all__ = ['run_overlay', 'create_interactive_overlay']