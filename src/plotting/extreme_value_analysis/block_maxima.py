"""
Block maxima extraction functions
"""

import logging
import traceback
import pandas as pd
import numpy as np
from typing import Optional
from .utils import get_block_size_days


def get_block_maxima(ts: pd.Series, block: str, data_type: str) -> Optional[pd.Series]:
    """Get block maxima using fixed-day blocks. Includes all blocks with data."""
    try:
        if ts is None or len(ts) < 10:
            logging.warning("Insufficient data points for block maxima calculation")
            return None
            
        # Ensure the time series is sorted by index
        ts = ts.sort_index()
        
        # Convert block period to days for consistent handling
        if block.endswith('M'):
            days = int(block[:-1]) * 30  # Convert months to 30-day periods
        elif block.endswith('Y'):
            days = int(block[:-1]) * 365  # Convert years to days
        elif block.endswith('W'):
            days = int(block[:-1]) * 7    # Convert weeks to days
        elif block.endswith('D'):
            days = int(block[:-1])        # Already in days
        else:
            block_days = get_block_size_days(block)
            days = int(round(block_days))
        
        # Create fixed-size blocks
        start_date = ts.index.min()
        end_date = ts.index.max()
        
        # Generate block boundaries
        block_boundaries = pd.date_range(
            start=start_date,
            end=end_date,
            freq=f'{days}D'
        )
        
        # Calculate maxima for each block
        extremes = []
        
        for start, end in zip(block_boundaries[:-1], block_boundaries[1:]):
            block_data = ts[start:end]
            if len(block_data) > 0:
                max_value = block_data.max()
                max_time = block_data.idxmax()
                
                # Check if maximum occurs exactly on the end boundary
                if max_time == end:
                    # Look for the actual peak before the boundary
                    block_data_before = block_data[:-1]  
                    if len(block_data_before) > 0:
                        max_value = block_data_before.max()
                        max_time = block_data_before.idxmax()
                
                extremes.append((max_time, max_value))
                block_length = (block_data.index.max() - block_data.index.min()).total_seconds() / (24*3600)
                logging.debug(f"Block {start} to {end}: length={block_length:.1f} days, max={max_value:.2f}")
        
        if extremes:
            # Create series from extreme values
            extremes = pd.Series(
                data=[v[1] for v in extremes],
                index=[v[0] for v in extremes],
                name='extremes'
            )
            
            extremes = extremes.sort_index()
            
            logging.info(f"Found {len(extremes)} maxima using {days}-day blocks")
            logging.info(f"First extreme: {extremes.index[0]}, Last extreme: {extremes.index[-1]}")
            logging.info(f"Value range: {extremes.min():.2f} to {extremes.max():.2f}")
            
            return extremes
        else:
            logging.warning("No extreme values found")
            return None
        
    except Exception as e:
        logging.error(f"Error in get_block_maxima: {str(e)}")
        traceback.print_exc()
        return None

