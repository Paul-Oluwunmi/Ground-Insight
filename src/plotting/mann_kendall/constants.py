"""
Configuration parameters for Mann-Kendall analysis

These constants control the behavior of all Mann-Kendall analysis functions.
Configuration is loaded from config.txt file - DO NOT edit this Python file directly!
Edit config.txt instead to change settings.
"""

import os
import ast

# Default values (used if config.txt is missing or has errors)
_DEFAULTS = {
    'USE_ROLLING_MEAN': True,
    'ROLLING_WINDOW_SIZE': 7,
    'ALPHA_LEVEL': 0.05,
    'RESAMPLE_FREQUENCY': 'D',
    'CUSTOM_SAVE_LOCATION': None,
    'CREATE_FOLDERS': True,
    'SAVE_PLOTS': True,
    'RUN_ORIGINAL_MK': False,
    'RUN_SEASONAL_MK': False,
    'RUN_TIME_EVOLUTION': False,
    'RUN_HYDROLOGICAL_YEAR': False,
    'RUN_DURATION_LOW_FLOW': False,
    'RUN_YEARLY_STATISTICS': False,
    'TIME_EVOLUTION_WINDOW_SIZE': 1825,
    'TIME_EVOLUTION_STEP_SIZE': 1825,
    'TIME_EVOLUTION_OVERLAP': False,
    'TIME_EVOLUTION_MIN_POINTS': 1,
    'IGNORE_ZERO_VALUES_IN_TREND': False,
}

def _load_config():
    """Load configuration from config.txt file."""
    config = _DEFAULTS.copy()
    
    # Get the directory where this file is located
    config_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(config_dir, 'config.txt')
    
    if not os.path.exists(config_file):
        print(f"Warning: config.txt not found at {config_file}")
        print("Using default configuration values.")
        return config
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                # Skip comments and empty lines
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Parse key = value
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Try to evaluate the value (handles True, False, None, numbers, strings)
                    try:
                        # Handle None
                        if value.lower() == 'none':
                            config[key] = None
                        # Handle boolean
                        elif value.lower() == 'true':
                            config[key] = True
                        elif value.lower() == 'false':
                            config[key] = False
                        # Handle numbers
                        elif value.replace('.', '', 1).replace('-', '', 1).isdigit():
                            if '.' in value:
                                config[key] = float(value)
                            else:
                                config[key] = int(value)
                        # Handle strings (remove quotes if present)
                        else:
                            # Remove quotes if they exist
                            if (value.startswith('"') and value.endswith('"')) or \
                               (value.startswith("'") and value.endswith("'")):
                                config[key] = value[1:-1]
                            else:
                                config[key] = value
                    except Exception as e:
                        print(f"Warning: Could not parse value for {key} on line {line_num}: {value}")
                        print(f"  Error: {e}")
                        print(f"  Using default value: {_DEFAULTS.get(key, 'N/A')}")
    
    except Exception as e:
        print(f"Error reading config.txt: {e}")
        print("Using default configuration values.")
    
    return config

# Load configuration from file
_config = _load_config()

# ROLLING MEAN SETTINGS
USE_ROLLING_MEAN = _config.get('USE_ROLLING_MEAN', _DEFAULTS['USE_ROLLING_MEAN'])
ROLLING_WINDOW_SIZE = _config.get('ROLLING_WINDOW_SIZE', _DEFAULTS['ROLLING_WINDOW_SIZE'])

# ANALYSIS SETTINGS
ALPHA_LEVEL = _config.get('ALPHA_LEVEL', _DEFAULTS['ALPHA_LEVEL'])
RESAMPLE_FREQUENCY = _config.get('RESAMPLE_FREQUENCY', _DEFAULTS['RESAMPLE_FREQUENCY'])

# SAVE LOCATION SETTINGS
CUSTOM_SAVE_LOCATION = _config.get('CUSTOM_SAVE_LOCATION', _DEFAULTS['CUSTOM_SAVE_LOCATION'])
CREATE_FOLDERS = _config.get('CREATE_FOLDERS', _DEFAULTS['CREATE_FOLDERS'])
SAVE_PLOTS = _config.get('SAVE_PLOTS', _DEFAULTS['SAVE_PLOTS'])

# BATCH ANALYSIS SETTINGS
RUN_ORIGINAL_MK = _config.get('RUN_ORIGINAL_MK', _DEFAULTS['RUN_ORIGINAL_MK'])
RUN_SEASONAL_MK = _config.get('RUN_SEASONAL_MK', _DEFAULTS['RUN_SEASONAL_MK'])
RUN_TIME_EVOLUTION = _config.get('RUN_TIME_EVOLUTION', _DEFAULTS['RUN_TIME_EVOLUTION'])
RUN_HYDROLOGICAL_YEAR = _config.get('RUN_HYDROLOGICAL_YEAR', _DEFAULTS['RUN_HYDROLOGICAL_YEAR'])
RUN_DURATION_LOW_FLOW = _config.get('RUN_DURATION_LOW_FLOW', _DEFAULTS['RUN_DURATION_LOW_FLOW'])
RUN_YEARLY_STATISTICS = _config.get('RUN_YEARLY_STATISTICS', _DEFAULTS['RUN_YEARLY_STATISTICS'])

# TIME EVOLUTION SETTINGS
TIME_EVOLUTION_WINDOW_SIZE = _config.get('TIME_EVOLUTION_WINDOW_SIZE', _DEFAULTS['TIME_EVOLUTION_WINDOW_SIZE'])
TIME_EVOLUTION_STEP_SIZE = _config.get('TIME_EVOLUTION_STEP_SIZE', _DEFAULTS['TIME_EVOLUTION_STEP_SIZE'])
TIME_EVOLUTION_OVERLAP = _config.get('TIME_EVOLUTION_OVERLAP', _DEFAULTS['TIME_EVOLUTION_OVERLAP'])
TIME_EVOLUTION_MIN_POINTS = _config.get('TIME_EVOLUTION_MIN_POINTS', _DEFAULTS['TIME_EVOLUTION_MIN_POINTS'])

# TREND ANALYSIS SETTINGS
IGNORE_ZERO_VALUES_IN_TREND = _config.get('IGNORE_ZERO_VALUES_IN_TREND', _DEFAULTS['IGNORE_ZERO_VALUES_IN_TREND'])

__all__ = [
    'USE_ROLLING_MEAN',
    'ROLLING_WINDOW_SIZE',
    'ALPHA_LEVEL',
    'RESAMPLE_FREQUENCY',
    'CUSTOM_SAVE_LOCATION',
    'CREATE_FOLDERS',
    'SAVE_PLOTS',
    'RUN_ORIGINAL_MK',
    'RUN_SEASONAL_MK',
    'RUN_TIME_EVOLUTION',
    'RUN_HYDROLOGICAL_YEAR',
    'RUN_DURATION_LOW_FLOW',
    'RUN_YEARLY_STATISTICS',
    'TIME_EVOLUTION_WINDOW_SIZE',
    'TIME_EVOLUTION_STEP_SIZE',
    'TIME_EVOLUTION_OVERLAP',
    'TIME_EVOLUTION_MIN_POINTS',
    'IGNORE_ZERO_VALUES_IN_TREND'
]

