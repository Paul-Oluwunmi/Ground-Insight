"""
Data loading module for groundwater analysis.

This module provides functions to load groundwater data from various formats:
- Indexed format (SiteName, Time, Value columns)
- Single file (wide or long format)
- Multiple CSV files from a folder
- Large multi-file long-format datasets
"""

from .indexed_loader import load_indexed_format
from .single_file_loader import load_and_process_data, load_mapping_file
from .multi_file_loader import load_multiple_csv_enhanced
from .large_dataset_loader import load_large_datasets
from .format_detection import detect_data_format
from .utils import clean_filename, convert_units, remove_duplicates
from .rainfall_loader import load_rainfall_data, merge_rainfall_data
from .datetime_parser import parse_datetime_column, COMMON_DATE_FORMATS

__all__ = [
    'load_indexed_format',
    'load_and_process_data',
    'load_multiple_csv_enhanced',
    'load_large_datasets',
    'detect_data_format',
    'clean_filename',
    'convert_units',
    'remove_duplicates',
    'load_mapping_file',
    'load_rainfall_data',
    'merge_rainfall_data',
    'parse_datetime_column',
    'COMMON_DATE_FORMATS',
]

