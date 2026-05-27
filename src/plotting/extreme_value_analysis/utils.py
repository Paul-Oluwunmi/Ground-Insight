"""
Utility functions for Extreme Value Analysis
"""

import logging
from typing import Tuple


def get_period_hours(period: str) -> float:
    """Convert period string to hours"""
    try:
        if not period:
            return 24  # Default to 1 day
            
        value = float(period[:-1])
        unit = period[-1]
        
        conversions = {
            'T': 1/60,        # Minutes to hours
            'H': 1,           # Hours
            'D': 24,          # Days
            'W': 24 * 7,      # Weeks
            'M': 24 * 30.44,  # Average month
            'Y': 24 * 365.25  # Year with leap
        }
        
        if unit not in conversions:
            logging.warning(f"Unrecognized time unit: {unit}, defaulting to hours")
            return value
            
        return value * conversions[unit]
        
    except ValueError as e:
        logging.error(f"Invalid period format: {period} - {str(e)}")
        return 24
    except Exception as e:
        logging.error(f"Error in get_period_hours: {str(e)}")
        return 24


def get_block_size_days(block: str) -> float:
    """Convert block period to approximate number of days"""
    try:
        if not block:
            return 1
        
        # Extract numeric value and unit
        value = float(block[:-1])
        unit = block[-1]
        
        conversions = {
            'H': 1/24,        # Hours to days
            'D': 1,           # Days
            'W': 7,          # Weeks to days
            'M': 30.44,      # Average month length
            'Y': 365.25      # Account for leap years
        }
        
        if unit not in conversions:
            logging.warning(f"Unrecognized time unit: {unit}, defaulting to days")
            return value
            
        return value * conversions[unit]
        
    except ValueError as e:
        logging.error(f"Invalid block period format: {block} - {str(e)}")
        return 1
    except Exception as e:
        logging.error(f"Error in get_block_size_days: {str(e)}")
        return 1


def format_resample_period(resample: str) -> str:
    """Format resample period for display."""
    try:
        # Handle empty or invalid input
        if not resample or not isinstance(resample, str):
            return "Original data"
            
        # Extract numeric value and unit
        value = float(resample[:-1])
        unit = resample[-1]
        
        # Handle integer values
        if value.is_integer():
            value = int(value)
        
        formats = {
            'T': ('minute', 'minutes'),
            'H': ('hour', 'hours'),
            'D': ('day', 'days'),
            'W': ('week', 'weeks'),
            'M': ('month', 'months'),
            'Y': ('year', 'years')
        }
        
        if unit in formats:
            singular, plural = formats[unit]
            return f"{value} {singular if value == 1 else plural}"
            
        return resample
        
    except Exception as e:
        logging.error(f"Error in format_resample_period: {str(e)}")
        return str(resample)


def convert_time_unit(resample: str) -> str:
    """Convert pandas time units to polars time units"""
    # Extract numeric value and unit
    value = resample[:-1]
    unit = resample[-1]
    
    # Mapping of pandas units to polars units
    unit_map = {
        'T': 'm',    # minutes
        'H': 'h',    # hours
        'D': 'd',    # days
        'W': 'w',    # weeks
        'M': 'mo',   # months
        'Y': 'y'     # years
    }
    
    if unit not in unit_map:
        raise ValueError(f"Unsupported time unit: {unit}")
        
    return f"{value}{unit_map[unit]}"


def get_suitable_block_periods(data_length_years: float, data_type: str) -> Tuple[list, str]:
    """Return suitable block periods based on data length and data type"""
    try:
        suitable_periods = []
        min_blocks = 8 if data_type == 'gwl' else 10
        
        # Define periods with metadata for easier maintenance
        periods = [
            # (label, value, blocks_per_year, min_years, max_years)
            ('1 Day', '1D', 365, 0.1, None),
            ('2 Days', '2D', 182.5, 0.1, None),
            ('3 Days', '3D', 121.67, 0.1, None),
            ('4 Days', '4D', 91.25, 0.1, None),
            ('5 Days', '5D', 73, 0.1, None),
            ('6 Days', '6D', 60.83, 0.1, None),
            ('1 Week', '1W', 52, 0.25, None),    
            ('2 Weeks', '2W', 26, 0.5, None),    
            ('3 Weeks', '3W', 17.33, 0.75, None), 
            ('1 Month', '1M', 12, 1, None),      
            ('2 Months', '2M', 6, 1.5, None),    
            ('3 Months', '3M', 4, 2, None),      
            ('4 Months', '4M', 3, 2.7, None),    
            ('5 Months', '5M', 2.4, 3.3, None),  
            ('6 Months', '6M', 2, 4, None),      
            ('7 Months', '7M', 1.71, 4.7, None), 
            ('8 Months', '8M', 1.5, 5.3, None),  
            ('9 Months', '9M', 1.33, 6, None),   
            ('10 Months', '10M', 1.2, 6.7, None),
            ('11 Months', '11M', 1.09, 7.3, None),
            ('1 Year', '1Y', 1, 8, None),        
            ('2 Years', '2Y', 0.5, 16, None),    
            ('5 Years', '5Y', 0.2, 40, 200),     
            ('10 Years', '10Y', 0.1, 80, 400),
            ('20 Years', '20Y', 0.05, 160, 800),
            ('50 Years', '50Y', 0.02, 400, 2000),
            ('100 Years', '100Y', 0.01, 800, 4000),
            ('200 Years', '200Y', 0.005, 1600, 8000),
            ('500 Years', '500Y', 0.002, 4000, 20000),
        ]
        
        # Add periods that provide sufficient blocks and meet min/max requirements
        for label, value, blocks_per_year, min_years, max_years in periods:
            if data_length_years >= min_years:
                if max_years is None or data_length_years <= max_years:
                    num_blocks = data_length_years * blocks_per_year
                    if num_blocks >= min_blocks:
                        suitable_periods.append({'label': label, 'value': value})
        
        default_period = get_default_block_period(data_length_years, suitable_periods)
        return suitable_periods, default_period
        
    except Exception as e:
        logging.error(f"Error in get_suitable_block_periods: {str(e)}")
        return [{'label': '1 Month', 'value': '1M'}], '1M'


def get_default_block_period(data_length_years: float, suitable_periods: list) -> str:
    """Choose optimal default block period based on data length"""
    try:
        if not suitable_periods:
            return '1M'
            
        # Define optimal periods for different data lengths
        if data_length_years >= 80:
            preferred = ['2Y', '1Y', '6M', '4M', '3M', '2M', '1M']
        elif data_length_years >= 40:
            preferred = ['1Y', '6M', '4M', '3M', '2M', '1M']
        elif data_length_years >= 16:
            preferred = ['6M', '4M', '3M', '2M', '1M']
        elif data_length_years >= 8:
            preferred = ['4M', '3M', '2M', '1M']
        else:
            preferred = ['1M', '2M', '3M']
            
        # Find the first available preferred period
        available_periods = [p['value'] for p in suitable_periods]
        for period in preferred:
            if period in available_periods:
                return period
                
        # If no preferred period is available, return the largest available period
        return available_periods[0]
        
    except Exception as e:
        logging.error(f"Error in get_default_block_period: {str(e)}")
        return '1M'


def get_resample_options(block_period: str, data_type: str) -> list:
    """Get valid resample options based on block period and data type"""
    try:
        # Get block period in hours
        block_hours = get_period_hours(block_period)
        block_days = block_hours / 24  # Convert to days
        
        # Calculate dynamic points based on original measurement frequency
        if data_type == 'rain':
            # Rainfall: 10-minute measurements
            measurements_per_day = 144  # 24 * 60 / 10
            original_interval = '10T'
            original_label = '10 minutes'
        else:  # gwl
            # Groundwater: 15-minute measurements
            measurements_per_day = 96   # 24 * 60 / 15
            original_interval = '15T'
            original_label = '15 minutes'
            
        # Calculate total possible points for this block period
        total_possible_points = block_days * measurements_per_day
        
        # Use total possible points as maximum 
        max_points = total_possible_points
        
        # Ensure minimum points for statistical validity
        min_points = max(8, block_days / 30)  
        
        logging.info(f"""
        Block period: {block_period}
        Block days: {block_days:.1f}
        Original measurements per day: {measurements_per_day}
        Total possible points: {total_possible_points:.1f}
        Max points: {max_points:.1f}
        Min points: {min_points:.1f}
        """)

        # Always include original measurement interval with clear label
        valid_periods = [(original_label, original_interval)]
        
        # Add other valid periods
        all_periods = [
            # High frequency options (minutes)
            ('30 minutes', '30T'),
            # Hourly options
            *[(f'{h} hour{"s" if h > 1 else ""}', f'{h}H') for h in range(1, 24)],
            # Daily options
            *[(f'{d} day{"s" if d > 1 else ""}', f'{d}D') for d in range(1, 8)],
            # Weekly options
            ('1 week', '1W'),
            ('2 weeks', '2W'),
            # Monthly options (if block period is long enough)
            ('1 month', '1M')
        ]
        
        # Filter periods based on points per block
        for label, value in all_periods:
            
            if value == original_interval:
                continue
                
            period_hours = get_period_hours(value)
            points_in_block = block_hours / period_hours
            
            if min_points <= points_in_block <= max_points:
                valid_periods.append((label, value))
        
        # Convert to dropdown options
        options = [{'label': label, 'value': value} for label, value in valid_periods]
        
        return options
        
    except Exception as e:
        logging.error(f"Error in get_resample_options: {str(e)}")
        default_option = (
            {'label': '10 minutes', 'value': '10T'} if data_type == 'rain' 
            else {'label': '15 minutes', 'value': '15T'}
        )
        return [default_option]


def validate_block_period(block):
    """Validate and standardize block period format"""
    if not isinstance(block, str):
        raise ValueError("Block period must be a string")
    
    # Standard formats: nD, nM, nY where n is a number
    if not any(block.endswith(unit) for unit in ['D', 'M', 'Y']):
        raise ValueError("Invalid block period format")
    
    try:
        value = float(block[:-1])
        if value <= 0:
            raise ValueError("Block period must be positive")
    except ValueError:
        raise ValueError("Invalid block period value")
    
    return block

