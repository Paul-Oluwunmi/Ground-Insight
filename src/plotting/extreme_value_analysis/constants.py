"""
Constants for Extreme Value Analysis module
"""

# Season colors for visualization
SEASON_COLORS = {
    'Summer': 'rgba(255, 165, 0, 0.8)',  # Orange
    'Autumn': 'rgba(165, 42, 42, 0.8)',  # Brown
    'Winter': 'rgba(30, 144, 255, 0.8)',  # Blue
    'Spring': 'rgba(50, 205, 50, 0.8)'   # Green
}

# Color scheme for dashboard
COLORS = {
    'background': '#ffffff',
    'text': '#343a40',
    'primary': '#007bff',
    'secondary': '#6c757d',
    'success': '#28a745',
    'danger': '#dc3545',
    'warning': '#ffc107',
    'info': '#17a2b8',
    'light': '#f8f9fa',
    'dark': '#343a40',
    'hover': '#e9ecef', 
    'shadow': 'rgba(0, 0, 0, 0.1)',
    'grid': 'rgb(233, 236, 239)',
    'confidence': 'rgba(108, 117, 125, 0.2)'
}

# Global store for data (will be initialized in main module)
global_store: dict = {}

