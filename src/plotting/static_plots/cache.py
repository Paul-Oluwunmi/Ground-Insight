"""
Static Plots Cache Management
=============================

Smart caching for processed well data to optimize performance.
"""

import gc
from typing import Dict, Tuple, Optional
import numpy as np

# Smart caching for processed data - avoids recalculating the same well data
_well_data_cache = {}


def clear_well_cache():
    """Clear the well data cache to free memory."""
    global _well_data_cache
    _well_data_cache.clear()
    gc.collect()


def get_cached_well_data(data_hash: str, well_column: str) -> Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """Get cached well data if available."""
    cache_key = f"{data_hash}_{well_column}"
    return _well_data_cache.get(cache_key)


def cache_well_data(data_hash: str, well_column: str, intervals: Dict[str, Tuple[np.ndarray, np.ndarray]]):
    """Cache processed well data for reuse."""
    cache_key = f"{data_hash}_{well_column}"
    _well_data_cache[cache_key] = intervals
    
    # Limit cache size to prevent memory issues
    if len(_well_data_cache) > 100:  # Keep only last 100 processed wells
        # Remove oldest entries
        oldest_keys = list(_well_data_cache.keys())[:20]
        for key in oldest_keys:
            del _well_data_cache[key]
        gc.collect()

