"""
Subplot Cache Management
=========================

Smart caching for resampled data to optimize performance.
"""

import gc
from typing import Optional
import pandas as pd

# Smart caching for processed data
_resampled_data_cache = {}


def clear_cache():
    """Clear the resampled data cache to free memory."""
    global _resampled_data_cache
    _resampled_data_cache.clear()
    gc.collect()


def get_cached_resampled_data(interval: str, data_hash: str) -> Optional[pd.DataFrame]:
    """Get cached resampled data if available."""
    cache_key = f"{data_hash}_{interval}"
    return _resampled_data_cache.get(cache_key)


def cache_resampled_data(interval: str, data_hash: str, resampled_data: pd.DataFrame):
    """Cache resampled data for reuse."""
    cache_key = f"{data_hash}_{interval}"
    _resampled_data_cache[cache_key] = resampled_data
    
    # Limit cache size to prevent memory issues
    if len(_resampled_data_cache) > 20:  # Keep only last 20 resampled datasets
        # Remove oldest entries
        oldest_keys = list(_resampled_data_cache.keys())[:5]
        for key in oldest_keys:
            del _resampled_data_cache[key]
        gc.collect()

