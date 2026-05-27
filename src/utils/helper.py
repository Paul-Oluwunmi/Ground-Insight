"""
Helper functions for the groundwater module.
"""

import gc

def cleanup():
    """Clean up resources and memory."""
    try:
        gc.collect()
    except Exception as e:
        print(f"Error during cleanup: {e}")