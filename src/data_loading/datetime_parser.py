"""
Helper functions for parsing datetime columns.
"""

import logging
import pandas as pd


# Common date formats to try
COMMON_DATE_FORMATS = [
    "%d/%m/%Y %H:%M",           # 06/03/2019 00:00 (most common)
    "%d/%m/%Y %H:%M:%S",       # 06/03/2019 00:00:00
    "%Y-%m-%d %H:%M:%S",       # 2019-03-06 00:00:00 (ISO)
    "%Y-%m-%d %H:%M",          # 2019-03-06 00:00
    "%m/%d/%Y %H:%M",          # 03/06/2019 00:00 (US format)
    "%m/%d/%Y %H:%M:%S",       # 03/06/2019 00:00:00 (US format)
    "%d-%m-%Y %H:%M",          # 06-03-2019 00:00
    "%d-%m-%Y %H:%M:%S",       # 06-03-2019 00:00:00
    "%Y/%m/%d %H:%M:%S",       # 2019/03/06 00:00:00
    "%Y/%m/%d %H:%M",          # 2019/03/06 00:00
    "%d/%m/%Y",                # 06/03/2019 (date only)
    "%Y-%m-%d",                # 2019-03-06 (date only)
    "%m/%d/%Y",                # 03/06/2019 (US date only)
]


def parse_datetime_column(data_pd, datetime_col='DateTime', min_success_rate=0.9):
    """
    Parse datetime column with multiple format attempts.
    
    Parameters:
    -----------
    data_pd : pandas.DataFrame
        Dataframe with datetime column to parse
    datetime_col : str, default 'DateTime'
        Name of datetime column
    min_success_rate : float, default 0.9
        Minimum success rate (0-1) to consider parsing successful
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with parsed datetime column
    """
    datetime_parsed = False
    
    for fmt in COMMON_DATE_FORMATS:
        try:
            parsed = pd.to_datetime(data_pd[datetime_col], format=fmt, errors='coerce')
            success_rate = parsed.notna().sum() / len(data_pd)
            if success_rate >= min_success_rate:
                data_pd[datetime_col] = parsed
                datetime_parsed = True
                logging.info(f"Successfully parsed DateTime using format: {fmt} ({parsed.notna().sum()}/{len(data_pd)} valid dates)")
                break
        except:
            continue
    
    # Fallback to auto-detection
    if not datetime_parsed:
        data_pd[datetime_col] = pd.to_datetime(data_pd[datetime_col], errors='coerce')
        valid_dates = data_pd[datetime_col].notna().sum()
        logging.info(f"Using auto-detection for DateTime parsing ({valid_dates}/{len(data_pd)} valid dates)")
    
    return data_pd

