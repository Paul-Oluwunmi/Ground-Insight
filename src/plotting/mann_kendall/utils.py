"""
Utility functions for Mann-Kendall analysis

This module contains helper functions for date calculations, rolling windows,
season detection, data validation, and file naming.
"""

import pandas as pd
import numpy as np
from functools import lru_cache
from .constants import ROLLING_WINDOW_SIZE


# =============================================================================
# DATE AND TIME UTILITIES
# =============================================================================

@lru_cache(maxsize=256)
def safe_calculate_years_from_dates(start_date, end_date):
    """
    Safely calculate years between two dates, handling different date types.
    
    Args:
        start_date: Start date (datetime, Timestamp, or string)
        end_date: End date (datetime, Timestamp, or string)
    
    Returns:
        float: Years between dates
    """
    try:
        # Ensure we have proper datetime objects
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Calculate date range
        date_range = end_dt - start_dt
        
        # Convert to years
        if hasattr(date_range, 'days'):
            return date_range.days / 365.25
        elif hasattr(date_range, 'total_seconds'):
            return date_range.total_seconds() / (365.25 * 24 * 3600)
        else:
            # Fallback: calculate days manually
            return (end_dt - start_dt).days / 365.25
    except Exception as e:
        print(f"Warning: Error calculating years from dates: {e}")
        return 0.0


@lru_cache(maxsize=256)
def safe_calculate_duration_days(start_date, end_date):
    """
    Safely calculate duration in days between two dates, handling different date types.
    
    Args:
        start_date: Start date (datetime, Timestamp, or string)
        end_date: End date (datetime, Timestamp, or string)
    
    Returns:
        int: Duration in days
    """
    try:
        # Ensure we have proper datetime objects
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Calculate duration
        duration = end_dt - start_dt
        
        # Convert to days
        if hasattr(duration, 'days'):
            return duration.days
        elif hasattr(duration, 'total_seconds'):
            return int(duration.total_seconds() / (24 * 3600))
        else:
            # Fallback: calculate days manually
            return (end_dt - start_dt).days
    except Exception as e:
        print(f"Warning: Error calculating duration in days: {e}")
        return 0


@lru_cache(maxsize=1024)
def get_nz_season(date):
    """
    Get New Zealand season for a given date.
    
    Args:
        date: pandas Timestamp or datetime object
    
    Returns:
        str: Season name (Summer, Autumn, Winter, Spring)
    """
    if pd.isna(date):
        return 'N/A'
    
    month = date.month
    
    if month in [12, 1, 2]:
        return 'Summer'
    elif month in [3, 4, 5]:
        return 'Autumn'
    elif month in [6, 7, 8]:
        return 'Winter'
    elif month in [9, 10, 11]:
        return 'Spring'
    else:
        return 'Unknown'


# =============================================================================
# ROLLING WINDOW UTILITIES
# =============================================================================

def calculate_strict_rolling_window(data_series, target_years=None, rolling_window_size=None):
    """
    Calculate rolling window size - ALWAYS applies rolling mean regardless of data length.
    
    This function simply returns the rolling window configuration without restrictions.
    
    Args:
        data_series: pandas Series with datetime index
        target_years: Target years for rolling window (default: uses ROLLING_WINDOW_SIZE/365.25)
        rolling_window_size: Specific rolling window size to use
    
    Returns:
        dict: Rolling window configuration including:
            - window_size_days: Rolling window size in days
            - window_size_years: Window size in years
            - can_apply: Always True
            - reason: Always applies rolling mean
    """
    try:
        # Always apply rolling mean - no restrictions
        if rolling_window_size is None:
            window_size_days = ROLLING_WINDOW_SIZE
        else:
            window_size_days = rolling_window_size
            
        window_size_years = window_size_days / 365.25
        
        result = {
            'window_size_days': window_size_days,
            'window_size_years': window_size_years,
            'can_apply': True,
            'reason': f'ALWAYS APPLIED - {window_size_days}-day rolling window'
        }
        
        return result
        
    except Exception as e:
        print(f"Warning: Error calculating rolling window: {e}")
        return {
            'window_size_days': 0,
            'window_size_years': 0,
            'can_apply': False,
            'reason': f'IGNORED - Error: {str(e)}'
        }


def check_rolling_mean_compatibility(data, well_columns, target_years=None):
    """
    Check rolling mean compatibility for all wells - always applies rolling mean.
    
    Args:
        data: DataFrame with datetime index and well columns
        well_columns: List of well column names
        target_years: Target years for rolling window (default: uses ROLLING_WINDOW_SIZE/365.25)
    
    Returns:
        dict: Compatibility information for each well
    """
    compatibility_info = {}
    
    for well in well_columns:
        if well in data.columns:
            well_data = data[well].dropna()
            if len(well_data) > 0:
        
                try:
                    # Calculate rolling window for this well - always applies
                    adaptive_config = calculate_strict_rolling_window(well_data, target_years)
                    
                    # Always apply rolling mean - no restrictions
                    can_apply_rolling = True
                    
                    compatibility_info[well] = {
                        'years_available': round(adaptive_config['window_size_years'], 1),
                        'can_apply_rolling_mean': True,
                        'data_points': len(well_data),
                        'date_range_start': pd.to_datetime(well_data.index.min()).strftime('%Y-%m-%d'),
                        'date_range_end': pd.to_datetime(well_data.index.max()).strftime('%Y-%m-%d'),
                        'rolling_mean_applied': True,
                        'rolling_window_days': adaptive_config['window_size_years'],
                        'rolling_window_years': round(adaptive_config['window_size_years'], 1),
                        'adaptive_reason': 'ALWAYS APPLIED - No restrictions'
                    }
                    
                except Exception as e:
                    print(f"Warning: Error calculating rolling window for {well}: {e}")
                    compatibility_info[well] = {
                        'years_available': 0,
                        'can_apply_rolling_mean': False,
                        'data_points': len(well_data),
                        'date_range_start': 'Error',
                        'date_range_end': 'Error',
                        'rolling_mean_applied': False,
                        'rolling_window_days': None,
                        'rolling_window_years': None,
                        'adaptive_reason': f'Error: {str(e)}'
                    }
            else:
                compatibility_info[well] = {
                    'years_available': 0,
                    'can_apply_rolling_mean': False,
                    'data_points': 0,
                    'date_range_start': 'N/A',
                    'date_range_end': 'N/A',
                    'rolling_mean_applied': False,
                    'rolling_window_days': None,
                    'rolling_window_years': None,
                    'adaptive_reason': 'No data available'
                }
    
    return compatibility_info


# =============================================================================
# SLOPE CONVERSION UTILITIES
# =============================================================================

def convert_slope_to_mm_per_year(slope, resample_freq):
    """Convert slope to mm per year."""
    if pd.isna(slope) or slope == 0:
        return 'N/A'
    
    # Convert based on resampling frequency with error handling
    try:
        if resample_freq == 'D':  # Daily
            return slope * 365.25 * 1000  # Convert to mm/year
        elif resample_freq == 'W':  # Weekly
            return slope * 52.18 * 1000
        elif resample_freq == 'M':  # Monthly
            return slope * 12 * 1000
        elif resample_freq == 'A':  # Yearly
            return slope * 1000
        else:
            return slope * 1000  # Default conversion
    except (TypeError, ValueError, ZeroDivisionError):
        return 'N/A'


def format_slope_mm(value):
    """Format slope in mm/year."""
    if pd.isna(value) or value == 'N/A':
        return 'N/A'
    try:
        return f"{float(value):.2f}"
    except:
        return str(value)


# =============================================================================
# DATA VALIDATION UTILITIES
# =============================================================================

def ensure_datetime_index(data):
    """
    Ensure data has a proper datetime index, handling various input formats.
    
    Args:
        data: pandas DataFrame or Series
    
    Returns:
        pandas DataFrame or Series with proper datetime index
    """
    try:
        if isinstance(data, pd.DataFrame):
            # Check if DataFrame has DateTime column
            if 'DateTime' in data.columns:
                # Convert DateTime column to datetime if it's not already
                data = data.copy()
                data['DateTime'] = pd.to_datetime(data['DateTime'], errors='coerce')
                data = data.set_index('DateTime')
            elif 'Date' in data.columns:
                # Try Date column as alternative
                data = data.copy()
                data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
                data = data.set_index('Date')
            elif 'date' in data.columns:
                # Try lowercase date column
                data = data.copy()
                data['date'] = pd.to_datetime(data['date'], errors='coerce')
                data = data.set_index('date')
            elif 'time' in data.columns:
                # Try time column
                data = data.copy()
                data['time'] = pd.to_datetime(data['time'], errors='coerce')
                data = data.set_index('time')
            elif 'TIME' in data.columns:
                # Try uppercase TIME column
                data = data.copy()
                data['TIME'] = pd.to_datetime(data['TIME'], errors='coerce')
                data = data.set_index('TIME')
            else:
                # Check if index is already datetime
                if not isinstance(data.index, pd.DatetimeIndex):
                    # Try to convert index to datetime
                    try:
                        data.index = pd.to_datetime(data.index, errors='coerce')
                    except:
                        print("Warning: Could not convert index to datetime")
        
        elif isinstance(data, pd.Series):
            # Check if Series index is datetime
            if not isinstance(data.index, pd.DatetimeIndex):
                try:
                    data.index = pd.to_datetime(data.index, errors='coerce')
                except:
                    print("Warning: Could not convert Series index to datetime")
        
        # Final validation - ensure we have a proper datetime index
        if hasattr(data, 'index') and not isinstance(data.index, pd.DatetimeIndex):
            print("Warning: Data still does not have proper datetime index after conversion")
            return data
            
        return data
        
    except Exception as e:
        print(f"Error ensuring datetime index: {e}")
        # Try to provide more helpful error information
        if 'DateTime' in str(e):
            print("This appears to be a DateTime column access error")
            print("Recommendation: Check if your data has a 'DateTime' column or proper datetime index")
        return data


def safe_check_data_structure(data, function_name="unknown"):
    """
    Safely check data structure before analysis to prevent timestamp interval errors.
    
    Args:
        data: DataFrame or Series to check
        function_name: Name of the calling function for error reporting
    
    Returns:
        bool: True if data structure is valid, False otherwise
    """
    try:
        if data is None:
            print(f"Error in {function_name}: Data is None")
            return False
            
        # Ensure data has proper datetime index
        data = ensure_datetime_index(data)
        
        if not isinstance(data.index, pd.DatetimeIndex):
            print(f"Error in {function_name}: Data does not have proper datetime index")
            print("Attempting to fix...")
            
            # Try to find datetime column
            if hasattr(data, 'columns'):
                datetime_cols = [col for col in data.columns if any(keyword in col.lower() for keyword in ['date', 'time', 'datetime'])]
                if datetime_cols:
                    print(f"Found potential datetime columns: {datetime_cols}")
                    # Try the first one
                    try:
                        data = data.set_index(datetime_cols[0])
                        data.index = pd.to_datetime(data.index, errors='coerce')
                        if isinstance(data.index, pd.DatetimeIndex):
                            print("Successfully fixed datetime index")
                            return True
                    except:
                        pass
            
            print("Could not fix datetime index automatically")
            return False
        
        return True
        
    except Exception as e:
        print(f"Error checking data structure in {function_name}: {e}")
        return False


def safe_analyze_timestamp_intervals(data):
    """
    Safely analyze timestamp intervals in data without causing errors.
    
    Args:
        data: pandas DataFrame or Series with datetime index
    
    Returns:
        dict: Analysis results or error information
    """
    try:
        # Ensure data has proper datetime index first
        data = ensure_datetime_index(data)
        
        if not isinstance(data.index, pd.DatetimeIndex):
            return {
                'status': 'error',
                'message': 'Data does not have proper datetime index',
                'recommendation': 'Use ensure_datetime_index() function first'
            }
        
        if len(data) < 2:
            return {
                'status': 'error',
                'message': 'Insufficient data for interval analysis',
                'recommendation': 'Need at least 2 data points'
            }
        
        # Safe timestamp interval analysis
        try:
            # Check date range
            start_date = data.index.min()
            end_date = data.index.max()
            date_range = (end_date - start_date).days
            
            # Check for gaps (safely)
            if len(data) > 1:
                time_diffs = data.index.to_series().diff().dropna()
                max_gap = time_diffs.max()
                min_gap = time_diffs.min()
                mean_gap = time_diffs.mean()
            else:
                max_gap = min_gap = mean_gap = None
            
            return {
                'status': 'success',
                'start_date': start_date.strftime('%Y-%m-%d') if start_date else 'N/A',
                'end_date': end_date.strftime('%Y-%m-%d') if end_date else 'N/A',
                'date_range_days': date_range,
                'total_points': len(data),
                'max_gap': str(max_gap) if max_gap else 'N/A',
                'min_gap': str(min_gap) if min_gap else 'N/A',
                'mean_gap': str(mean_gap) if mean_gap else 'N/A'
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error during timestamp interval analysis: {str(e)}',
                'recommendation': 'Check data format and datetime index'
            }
            
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Error in safe_analyze_timestamp_intervals: {str(e)}',
            'recommendation': 'Ensure data has proper structure'
        }


def detect_data_format(data):
    """
    Detect the format of input data and provide compatibility information.
    
    Args:
        data: pandas DataFrame or Series
    
    Returns:
        dict: Information about data format and compatibility
    """
    try:
        format_info = {
            'has_datetime_index': False,
            'has_datetime_column': False,
            'datetime_column_name': None,
            'index_type': str(type(data.index)),
            'data_type': str(type(data)),
            'columns': list(data.columns) if hasattr(data, 'columns') else None,
            'sample_index': str(data.index[0]) if len(data.index) > 0 else 'No index',
            'compatibility_notes': []
        }
        
        if isinstance(data, pd.DataFrame):
            # Check for datetime columns
            datetime_columns = []
            for col in data.columns:
                if any(keyword in col.lower() for keyword in ['date', 'time', 'datetime']):
                    datetime_columns.append(col)
            
            if datetime_columns:
                format_info['has_datetime_column'] = True
                format_info['datetime_column_name'] = datetime_columns[0]
                format_info['compatibility_notes'].append(f"Found datetime column: {datetime_columns[0]}")
            
            # Check index type
            if isinstance(data.index, pd.DatetimeIndex):
                format_info['has_datetime_index'] = True
                format_info['compatibility_notes'].append("Data has proper datetime index")
            else:
                format_info['compatibility_notes'].append(f"Index type: {type(data.index).__name__}")
        
        elif isinstance(data, pd.Series):
            if isinstance(data.index, pd.DatetimeIndex):
                format_info['has_datetime_index'] = True
                format_info['compatibility_notes'].append("Series has proper datetime index")
            else:
                format_info['compatibility_notes'].append(f"Series index type: {type(data.index).__name__}")
        
        return format_info
        
    except Exception as e:
        return {
            'error': f'Error detecting data format: {str(e)}',
            'compatibility_notes': ['Could not determine data format']
        }


def assess_data_quality_for_mk(series):
    """
    Assess data quality for Mann-Kendall analysis to prevent errors.
    
    Args:
        series: pandas Series with groundwater level data
    
    Returns:
        dict: Data quality assessment with recommendations
    """
    try:
        if series is None or len(series) == 0:
            return {
                'quality': 'No Data',
                'issues': ['Series is None or empty'],
                'recommendations': ['Check data source and loading'],
                'mk_possible': False
            }
        
        # Remove NaN values
        clean_series = series.dropna()
        
        if len(clean_series) == 0:
            return {
                'quality': 'No Valid Data',
                'issues': ['All values are NaN'],
                'recommendations': ['Check data cleaning process'],
                'mk_possible': False
            }
        
        if len(clean_series) < 2:
            return {
                'quality': 'Insufficient Data',
                'issues': [f'Only {len(clean_series)} data point(s)'],
                'recommendations': ['Need at least 2 points for basic analysis', 'Cannot perform Mann-Kendall trend analysis'],
                'mk_possible': False
            }
        
        if len(clean_series) < 3:
            return {
                'quality': 'Limited Data',
                'issues': [f'Only {len(clean_series)} data point(s)'],
                'recommendations': ['Need at least 3 points for variance calculation', 'Cannot perform Mann-Kendall trend analysis'],
                'mk_possible': False
            }
        
        # Check for identical values
        unique_values = clean_series.nunique()
        if unique_values == 1:
            return {
                'quality': 'No Variance',
                'issues': ['All values are identical'],
                'recommendations': ['No trend possible with constant values', 'Cannot perform Mann-Kendall trend analysis'],
                'mk_possible': False
            }
        
        # Check variance
        variance = clean_series.var()
        if variance < 1e-12:
            return {
                'quality': 'Very Low Variance',
                'issues': [f'Variance: {variance:.2e}'],
                'recommendations': ['Data may be too uniform for trend analysis', 'Cannot perform Mann-Kendall trend analysis'],
                'mk_possible': False
            }
        
        # Check data span
        if hasattr(clean_series.index, 'min') and hasattr(clean_series.index, 'max'):
            try:
                time_span = clean_series.index.max() - clean_series.index.min()
                if time_span.days < 30:
                    return {
                        'quality': 'Short Time Span',
                        'issues': [f'Time span: {time_span.days} days'],
                        'recommendations': ['Consider longer time series for trend analysis'],
                        'mk_possible': True,
                        'warning': 'Short time series may not show meaningful trends'
                    }
            except:
                pass
        
        # Good quality data
        return {
            'quality': 'Good',
            'issues': [],
            'recommendations': ['Data suitable for Mann-Kendall analysis'],
            'mk_possible': True,
            'data_points': len(clean_series),
            'variance': variance,
            'unique_values': unique_values
        }
        
    except Exception as e:
        return {
            'quality': 'Error in Assessment',
            'issues': [f'Assessment failed: {str(e)}'],
            'recommendations': ['Check data format and structure'],
            'mk_possible': False
        }


# =============================================================================
# FILE NAMING UTILITIES
# =============================================================================

def generate_well_filename(well_name, suffix, mapping_dict=None):
    """
    Generate a filename that includes both well name and mapping name when available.
    
    Args:
        well_name: Original well name
        mapping_dict: Dictionary mapping well names to display names
        suffix: File suffix (e.g., '_duration_analysis.csv')
    
    Returns:
        str: Proper filename format
    """
    if mapping_dict and well_name in mapping_dict:
        # Use format: well_name.mapping_name_suffix
        mapping_name = mapping_dict[well_name]
        # Replace dots with underscores in well names only, preserve file extension
        well_part = f"{well_name}.{mapping_name}".replace('.', '_')
        filename = f"{well_part}{suffix}"
        return filename
    else:
        # No mapping available, use well name directly (replace dots with underscores in well name only)
        well_part = well_name.replace('.', '_')
        return f"{well_part}{suffix}"


def generate_well_folder_name(well_name, mapping_dict=None):
    """
    Generate a folder name that includes both well name and mapping name when available.
    
    Args:
        well_name: Original well name
        mapping_dict: Dictionary mapping well names to display names
    
    Returns:
        str: Proper folder name format
    """
    if mapping_dict and well_name in mapping_dict:
        # Use format: well_name.mapping_name
        mapping_name = mapping_dict[well_name]
        # Replace dots with underscores to prevent nested directory issues
        folder_name = f"{well_name}.{mapping_name}".replace('.', '_')
        return folder_name
    else:
        # No mapping available, use well name directly (replace dots with underscores)
        return well_name.replace('.', '_')


def generate_well_display_name(well_name, mapping_dict=None):
    """
    Generate a display name for the Well column in CSV files.
    
    Args:
        well_name: Original well name
        mapping_dict: Dictionary mapping well names to display names
    
    Returns:
        str: Well column value format
    """
    if mapping_dict and well_name in mapping_dict:
        # Use format: well_name.mapping_name
        mapping_name = mapping_dict[well_name]
        return f"{well_name}.{mapping_name}"
    else:
        # No mapping available, use well name directly
        return well_name


def get_well_folder_name_consistent(well_name, mapping_dict=None):
    """
    Get consistent folder name for a well based on mapping_dict availability.
    This function ensures all analysis phases use the same naming convention.
    
    Args:
        well_name: Original well name
        mapping_dict: Dictionary mapping well names to display names
    
    Returns:
        str: Consistent folder name to be used across all analysis phases
    """
    if mapping_dict and well_name in mapping_dict:
        # Use format: well_name.mapping_name
        mapping_name = mapping_dict[well_name]
        # Replace dots with underscores to prevent nested directory issues
        folder_name = f"{well_name}.{mapping_name}".replace('.', '_')
        return folder_name
    else:
        # No mapping available, use well name directly (replace dots with underscores)
        return well_name.replace('.', '_')


# =============================================================================
# DECORATORS
# =============================================================================

def robust_callback(func):
    """Decorator to make callbacks more robust."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Error in callback: {e}")
            return None
    return wrapper


def handle_timestamp_interval_errors(func):
    """
    Decorator to safely handle timestamp interval analysis errors.
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = str(e)
            if 'DateTime' in error_msg or 'timestamp' in error_msg.lower():
                print(f"Error analyzing timestamp intervals: {error_msg}")
                print("Attempting to fix data structure...")
                try:
                    # Try to fix the data structure
                    if args and hasattr(args[0], 'index'):
                        fixed_data = ensure_datetime_index(args[0])
                        if isinstance(fixed_data.index, pd.DatetimeIndex):
                            print("Data structure fixed successfully")
                            # Retry the function with fixed data
                            args = (fixed_data,) + args[1:]
                            return func(*args, **kwargs)
                except:
                    pass
                print("Could not automatically fix data structure")
                return None
            else:
                # Re-raise non-timestamp related errors
                raise
    return wrapper


def check_well_data_availability(data, well_columns):
    """
    Check data availability for each well to identify potential issues before analysis.
    
    Args:
        data: DataFrame with datetime index and well columns
        well_columns: List of well column names
    
    Returns:
        dict: Dictionary with data availability information for each well
    """
    availability_info = {}
    
    # Ensure data is a DataFrame
    if not hasattr(data, 'columns'):
        print("Warning: data is not a DataFrame, cannot check well availability")
        return {}
    
    for well in well_columns:
        if well in data.columns:
            try:
                well_data = data[well].dropna()
                total_points = len(well_data)
                
                if total_points == 0:
                    status = "No data"
                    recommendation = "Skip analysis"
                elif total_points == 1:
                    status = "Single data point"
                    recommendation = "Skip analysis - insufficient for meaningful results"
                elif total_points < 10:
                    status = "Very limited data"
                    recommendation = "May produce limited results"
                elif total_points < 50:
                    status = "Limited data"
                    recommendation = "Should produce basic results"
                else:
                    status = "Sufficient data"
                    recommendation = "Should produce comprehensive results"
                
                # Check year span if we have data
                if total_points > 0:
                    try:
                        # Ensure we have a DatetimeIndex
                        if isinstance(well_data.index, pd.DatetimeIndex):
                            years = well_data.index.year.unique()
                            year_span = len(years)
                            if year_span == 1:
                                year_info = f"Single year ({years[0]})"
                            else:
                                year_info = f"{year_span} years ({min(years)}-{max(years)})"
                        else:
                            # Try to convert to datetime if possible
                            try:
                                datetime_index = pd.to_datetime(well_data.index)
                                years = datetime_index.year.unique()
                                year_span = len(years)
                                if year_span == 1:
                                    year_info = f"Single year ({years[0]})"
                                else:
                                    year_info = f"{year_span} years ({min(years)}-{max(years)})"
                            except:
                                year_info = "Year info unavailable (non-datetime index)"
                    except Exception as e:
                        year_info = f"Year info unavailable (error: {str(e)[:50]})"
                else:
                    year_info = "N/A"
                
                availability_info[well] = {
                    'total_points': total_points,
                    'status': status,
                    'recommendation': recommendation,
                    'year_span': year_info
                }
            except Exception as e:
                print(f"Warning: Error checking well {well}: {e}")
                availability_info[well] = {
                    'total_points': 0,
                    'status': "Error during check",
                    'recommendation': "Check data format",
                    'year_span': "Error"
                }
    
    return availability_info


def display_well_data_availability(availability_info, mapping_dict=None):
    """
    Display well data availability information in a user-friendly format.
    
    Args:
        availability_info: Dictionary from check_well_data_availability function
        mapping_dict: Dictionary mapping well names to display names
    """
    print("\n" + "=" * 100)
    print("WELL DATA AVAILABILITY CHECK")
    print("=" * 100)
    
    # Safety check for empty or invalid availability_info
    if not availability_info or not isinstance(availability_info, dict):
        print("⚠️  No availability information available or invalid format")
        print("=" * 100)
        return
    
    # Group wells by status
    status_groups = {}
    for well, info in availability_info.items():
        if not isinstance(info, dict):
            print(f"⚠️  Invalid info format for well {well}: {type(info)}")
            continue
            
        status = info.get('status', 'Unknown')
        if status not in status_groups:
            status_groups[status] = []
        status_groups[status].append(well)
    
    # Display summary
    total_wells = len(availability_info)
    print(f"Total wells: {total_wells}")
    
    if not status_groups:
        print("⚠️  No valid well information found")
        print("=" * 100)
        return
    
    for status, wells in status_groups.items():
        print(f"\n{status}: {len(wells)} wells")
        for well in wells:
            well_display = mapping_dict.get(well, well) if mapping_dict else well
            info = availability_info[well]
            if isinstance(info, dict):
                total_points = info.get('total_points', 'Unknown')
                year_span = info.get('year_span', 'Unknown')
                recommendation = info.get('recommendation', 'No recommendation')
                print(f"  • {well_display}: {total_points} points, {year_span}")
                print(f"    Recommendation: {recommendation}")
            else:
                print(f"  • {well_display}: Invalid info format")
    
    # Show warnings for problematic wells
    problematic_wells = [w for w, info in availability_info.items() 
                        if isinstance(info, dict) and info.get('status') in ['No data', 'Single data point']]
    
    if problematic_wells:
        print(f"\n⚠️  WARNING: {len(problematic_wells)} wells have insufficient data for analysis:")
        for well in problematic_wells:
            well_display = mapping_dict.get(well, well) if mapping_dict else well
            print(f"  • {well_display}")
        print("These wells will be skipped during analysis to prevent errors.")
    
    print("=" * 100)


__all__ = [
    # Date utilities
    'safe_calculate_years_from_dates',
    'safe_calculate_duration_days',
    'get_nz_season',
    # Rolling window utilities
    'calculate_strict_rolling_window',
    'check_rolling_mean_compatibility',
    # Slope conversion
    'convert_slope_to_mm_per_year',
    'format_slope_mm',
    # Data validation
    'ensure_datetime_index',
    'safe_check_data_structure',
    'safe_analyze_timestamp_intervals',
    'detect_data_format',
    'assess_data_quality_for_mk',
    # File naming
    'generate_well_filename',
    'generate_well_folder_name',
    'generate_well_display_name',
    'get_well_folder_name_consistent',
    # Decorators
    'robust_callback',
    'handle_timestamp_interval_errors',
    # Data availability
    'check_well_data_availability',
    'display_well_data_availability'
]

