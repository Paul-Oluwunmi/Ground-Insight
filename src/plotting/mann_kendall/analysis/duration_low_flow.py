"""
Duration and Low Flow Analysis Module

This module contains functions for analyzing duration metrics and low flow periods
in groundwater time series data.
"""

import pandas as pd
import numpy as np
from ..constants import ROLLING_WINDOW_SIZE


def calculate_yearly_max_duration_metrics(series, resampling_freq='D', rolling_window=None, min_duration_days=1):
    """
    Calculate duration metrics for each year using total day counting approach.
    Produces the exact same format as the user's backup duration analysis tables.
    
    Args:
        series: Time series data with datetime index
        resampling_freq: Resampling frequency used for the data
        rolling_window: Window size for smoothing (auto-calculated if None)
        min_duration_days: Minimum consecutive days to count as a significant trend (for period identification)
    
    Returns:
        dict: Yearly duration metrics including total days, percentages, and period breakdowns (in calendar days)
    """
    try:
        if len(series) < 10:  # Reduced minimum data requirement for duration analysis
            return {
                'total_days_declining_per_year': 'N/A',
                'total_days_rising_per_year': 'N/A',
                'yearly_declining_periods': {},
                'yearly_rising_periods': {},
                'yearly_significant_declining_periods': {},
                'yearly_significant_rising_periods': {},
                'analysis_method': 'Insufficient data (< 10 points)'
            }
        
        # Step 1: Set rolling window based on resampling frequency if not provided
        if rolling_window is None:
            rolling_window = ROLLING_WINDOW_SIZE
        
        # Step 2: Apply rolling average to smooth noise (adaptive to resampling frequency)
        if len(series) >= rolling_window and rolling_window > 1:
            smoothed_series = series.rolling(window=rolling_window, center=False).mean()
            # Don't fill NaN values - let them remain NaN at edges
            # This is the correct behavior for rolling means
        else:
            smoothed_series = series
        
        # Step 3: Calculate changes from smoothed data
        changes = smoothed_series.diff()
        
        # Step 4: Calculate adaptive threshold based on data variability
        changes_std = changes.std()
        if pd.isna(changes_std) or changes_std == 0:
            # Fallback to fixed threshold if no variability
            adaptive_threshold = 0.01  # 1cm default
        else:
            # Use half standard deviation as threshold (more robust)
            adaptive_threshold = 0.5 * changes_std
        
        # Step 5: Adjust minimum duration based on resampling frequency
        min_duration_periods = 1  # Set to 1 to detect all events including single-day
        
        # Step 6: Classify each period as declining or rising
        # IMPORTANT: For total day counting, we need to ensure we're counting actual calendar days
        # If the data is not daily, we need to convert the boolean mask to represent daily periods
        declining = changes < -adaptive_threshold
        rising = changes > adaptive_threshold
        
        # Get unique years from the declining/rising series (which may now be daily)
        years = declining.index.year.unique()
        
        yearly_max_declining = {}
        yearly_max_rising = {}
        
        def find_all_duration_periods(condition_series, year, min_duration, original_series, threshold_value, trend_type):
            """
            Find ALL duration periods for a specific year with precise start and end dates using hybrid approach.
            Uses smoothed data to identify periods, then daily data for precise dates.
            Ensures non-overlapping periods by merging overlapping ones.
            
            NEW APPROACH: Uses rolling mean dates as reference points, then calculates
            daily averages from original data without shifting dates.
            
            Args:
                condition_series: Boolean series indicating declining/rising days (from smoothed data)
                year: Year to analyse
                min_duration: Minimum consecutive days to count as a significant trend
                original_series: Original daily series for precise date calculation
                threshold_value: The adaptive threshold value for precise date calculation
                trend_type: 'declining' or 'rising' to determine the comparison operator
            
            Returns:
                tuple: (all_periods_list, significant_periods_list, total_days)
            """

            year_data = condition_series[condition_series.index.year == year]
            if len(year_data) == 0 or year_data.sum() == 0:
                return [], [], 0
            
            # Find groups of consecutive True values using smoothed data
            groups = (year_data != year_data.shift()).cumsum()
            consecutive_lengths = year_data.groupby(groups).sum()

            # Get only periods where condition was True
            true_periods = consecutive_lengths[consecutive_lengths > 0]
            
            if len(true_periods) == 0:
                return [], [], 0
            
            # Calculate total days for this year
            total_days = year_data.sum()
            
            # Find all periods with precise start and end dates
            raw_periods = []
            
            for group_idx, length in true_periods.items():
                # Get the boolean mask for this group
                group_mask = groups == group_idx
                
                # Get the date range where this group occurs (from smoothed data)
                group_true_dates = year_data[group_mask & year_data].index
                
                if len(group_true_dates) > 0:
                    smoothed_start_date = group_true_dates[0]
                    smoothed_end_date = group_true_dates[-1]
                    
                    # NEW APPROACH: Use rolling mean dates as reference points - go back to original data
                    # Start from the exact rolling mean detection date, not before
                    # This preserves the original detection dates without shifting them
                    search_start = smoothed_start_date
                    search_end = smoothed_end_date
                    
                    # Get the daily data in this range (from rolling mean dates)
                    daily_range = original_series[search_start:search_end]
                    
                    # For each day in the range, calculate daily average and check threshold
                    # This handles cases with multiple readings per day (e.g., 15-minute intervals)
                    daily_averages = []
                    daily_dates = []
                    
                    # Group by date and calculate daily averages from original data
                    for date in pd.date_range(search_start, search_end, freq='D'):
                        if date in original_series.index:
                            # Get all readings for this specific date
                            day_data = original_series[original_series.index.date == date.date()]
                            if len(day_data) > 0:
                                daily_avg = day_data.mean()
                                daily_averages.append(daily_avg)
                                daily_dates.append(date)
                    
                    # Calculate daily changes from daily averages
                    if len(daily_averages) > 1:
                        daily_changes = np.diff(daily_averages)
                        
                        # Apply threshold based on trend type
                        if trend_type == 'declining':
                            daily_condition = daily_changes < -threshold_value
                        else:  # rising
                            daily_condition = daily_changes > threshold_value
                        
                        # Find the first and last days meeting the condition
                        below_threshold_indices = [i for i, below in enumerate(daily_condition) if below]
                        
                        if below_threshold_indices:
                            first_index = below_threshold_indices[0]
                            last_index = below_threshold_indices[-1]
                            precise_start_date = daily_dates[first_index]
                            precise_end_date = daily_dates[last_index]
                            actual_duration = (precise_end_date - precise_start_date).days + 1
                            
                            # Calculate mean groundwater level using daily averages for this period
                            period_daily_averages = []
                            for date in pd.date_range(precise_start_date, precise_end_date, freq='D'):
                                if date in original_series.index:
                                    day_data = original_series[original_series.index.date == date.date()]
                                    if len(day_data) > 0:
                                        period_daily_averages.append(day_data.mean())
                            
                            if period_daily_averages:
                                mean_level = np.mean(period_daily_averages)
                            else:
                                mean_level = original_series[precise_start_date:precise_end_date].mean()
                            
                            # Add to raw periods
                            raw_periods.append({
                                'start_date': precise_start_date,
                                'end_date': precise_end_date,
                                'duration_days': actual_duration,
                                'duration_periods': length,
                                'is_significant': length >= min_duration,
                                'mean_level': mean_level,
                                'smoothed_start': smoothed_start_date.strftime('%d-%b-%Y'),
                                'smoothed_end': smoothed_end_date.strftime('%d-%b-%Y')
                            })
                        else:
                            # No days below threshold found - skip this period
                            continue
                    else:
                        # Insufficient daily data - skip this period
                        continue
            
            # Merge overlapping periods to ensure non-overlapping
            if raw_periods:
                # Sort by start date
                raw_periods.sort(key=lambda x: x['start_date'])
                
                merged_periods = []
                current_period = raw_periods[0].copy()
                
                for next_period in raw_periods[1:]:
                    # Check if periods overlap or are adjacent
                    if next_period['start_date'] <= current_period['end_date'] + pd.Timedelta(days=1):
                        # Merge periods
                        current_period['end_date'] = max(current_period['end_date'], next_period['end_date'])
                        current_period['duration_days'] = (current_period['end_date'] - current_period['start_date']).days + 1
                        current_period['duration_periods'] = max(current_period['duration_periods'], next_period['duration_periods'])
                        current_period['is_significant'] = current_period['is_significant'] or next_period['is_significant']
                        # Recalculate mean for merged period
                        merged_period_data = original_series[current_period['start_date']:current_period['end_date']]
                        current_period['mean_level'] = merged_period_data.mean()
                    else:
                        # No overlap, add current period and start new one
                        merged_periods.append(current_period)
                        current_period = next_period.copy()
                
                # Add the last period
                merged_periods.append(current_period)
                
                # Convert to final format
                all_periods = []
                significant_periods = []
                
                for period in merged_periods:
                    # Add to all periods
                    all_periods.append({
                        'start_date': period['start_date'].strftime('%d-%b-%Y'),
                        'end_date': period['end_date'].strftime('%d-%b-%Y'),
                        'duration_days': period['duration_days'],
                        'duration_periods': period['duration_periods'],
                        'is_significant': period['is_significant'],
                        'mean_level': period['mean_level'],
                        'smoothed_start': period['smoothed_start'],
                        'smoothed_end': period['smoothed_end']
                    })
                    
                    # Add to significant periods if meets minimum duration
                    if period['is_significant']:
                        significant_periods.append({
                            'start_date': period['start_date'].strftime('%d-%b-%Y'),
                            'end_date': period['end_date'].strftime('%d-%b-%Y'),
                            'duration_days': period['duration_days'],
                            'mean_level': period['mean_level']
                        })
            else:
                all_periods = []
                significant_periods = []
            
            return all_periods, significant_periods, total_days
        
        # Calculate for each year using total day counting approach
        yearly_declining_periods = {}
        yearly_rising_periods = {}
        yearly_significant_declining_periods = {}
        yearly_significant_rising_periods = {}
        
        # Track total days for each year
        yearly_declining_days = {}
        yearly_rising_days = {}
        
        for year in years:
            declining_all_periods, declining_significant_periods, declining_total_days = find_all_duration_periods(declining, year, min_duration_periods, series, adaptive_threshold, 'declining')
            rising_all_periods, rising_significant_periods, rising_total_days = find_all_duration_periods(rising, year, min_duration_periods, series, adaptive_threshold, 'rising')
            
            # Total days are now already in calendar days since we expanded to daily resolution
            yearly_declining_days[year] = declining_total_days
            yearly_rising_days[year] = rising_total_days
            
            # Store the period information
            yearly_declining_periods[year] = declining_all_periods
            yearly_rising_periods[year] = rising_all_periods
            yearly_significant_declining_periods[year] = declining_significant_periods
            yearly_significant_rising_periods[year] = rising_significant_periods
        
        # Calculate annual averages
        if len(years) > 0:
            # Calculate average days per year
            annual_declining_days = np.mean(list(yearly_declining_days.values()))
            annual_rising_days = np.mean(list(yearly_rising_days.values()))
            
            # Range check - ensure values are reasonable
            max_possible_days = 366  # Maximum days per year
            if annual_declining_days > max_possible_days:
                annual_declining_days = min(annual_declining_days, max_possible_days)
            
            if annual_rising_days > max_possible_days:
                annual_rising_days = min(annual_rising_days, max_possible_days)
        else:
            annual_declining_days = 0
            annual_rising_days = 0
        
        # Calculate maximum and minimum durations and their years
        max_declining_duration = 0
        max_declining_year = None
        min_declining_duration = float('inf')
        min_declining_year = None
        
        max_rising_duration = 0
        max_rising_year = None
        min_rising_duration = float('inf')
        min_rising_year = None
        
        # Find maximum and minimum durations for each year (using ALL periods, not just significant ones)
        if yearly_declining_periods:
            for year, year_periods in yearly_declining_periods.items():
                for period in year_periods:
                    dur = period.get('duration_days', 0)
                    if dur > max_declining_duration:
                        max_declining_duration = dur
                        max_declining_year = year
                    if dur < min_declining_duration and dur > 0:  # Only consider non-zero durations
                        min_declining_duration = dur
                        min_declining_year = year
        
        if yearly_rising_periods:
            for year, year_periods in yearly_rising_periods.items():
                for period in year_periods:
                    dur = period.get('duration_days', 0)
                    if dur > max_rising_duration:
                        max_rising_duration = dur
                        max_rising_year = year
                    if dur < min_rising_duration and dur > 0:  # Only consider non-zero durations
                        min_rising_duration = dur
                        min_rising_year = year
        
        # Handle cases where no minimum was found (set to infinity)
        if min_declining_duration == float('inf'):
            min_declining_duration = 0
            min_declining_year = None
        if min_rising_duration == float('inf'):
            min_rising_duration = 0
            min_rising_year = None
        
        # Analysis method description
        smoothing_description = f"Rolling mean smoothing (window={rolling_window})"
        analysis_method = f"{smoothing_description}, total day counting approach, hybrid date calculation (smoothed identification + daily precision), adaptive threshold ({adaptive_threshold:.4f}m), max/min durations from ALL periods"
        
        # Format the maximum and minimum durations with their years
        max_declining_str = f"{max_declining_duration} days ({max_declining_year})" if max_declining_year is not None else "0 days"
        max_rising_str = f"{max_rising_duration} days ({max_rising_year})" if max_rising_year is not None else "0 days"
        min_declining_str = f"{min_declining_duration} days ({min_declining_year})" if min_declining_year is not None else "0 days"
        min_rising_str = f"{min_rising_duration} days ({min_rising_year})" if min_rising_year is not None else "0 days"
        
        return {
            'avg_max_declining_duration': max_declining_str,
            'avg_min_declining_duration': min_declining_str,
            'avg_max_rising_duration': max_rising_str,
            'avg_min_rising_duration': min_rising_str,
            'total_days_declining_per_year': f"{annual_declining_days:.1f} days/year",
            'total_days_rising_per_year': f"{annual_rising_days:.1f} days/year",
            'yearly_declining_periods': yearly_declining_periods,  # All periods
            'yearly_rising_periods': yearly_rising_periods,        # All periods
            'yearly_significant_declining_periods': yearly_significant_declining_periods,  # Periods meeting minimum duration
            'yearly_significant_rising_periods': yearly_significant_rising_periods,        # Periods meeting minimum duration
            'yearly_declining_days': yearly_declining_days,  # Total days per year
            'yearly_rising_days': yearly_rising_days,        # Total days per year
            'years_analysed': len(years),
            'analysis_method': analysis_method,
            'adaptive_threshold_used': f"{adaptive_threshold:.4f} metres",
            'original_data': series  # Add original data for overall statistics
        }
        
    except Exception as e:
        print(f"Duration calculation error: {e}")
        return {
            'total_days_declining_per_year': 'Error',
            'total_days_rising_per_year': 'Error',
            'yearly_declining_periods': {},
            'yearly_rising_periods': {},
            'yearly_significant_declining_periods': {},
            'yearly_significant_rising_periods': {},
            'analysis_method': f'Error: {str(e)}'
        }


def calculate_percentile_low_flow_metrics(series, resampling_freq='D', rolling_window=None, min_duration_days=1):
    """
    Calculate frequency and duration of groundwater levels below critical percentiles using total day counting approach.
    Produces the exact same format as the user's backup low flow analysis tables.
    
    Args:
        series: Time series data with datetime index
        resampling_freq: Resampling frequency used for the data
        rolling_window: Window size for smoothing (auto-calculated if None)
        min_duration_days: Minimum consecutive days to count as a low flow event (now used for period identification only)
    
    Returns:
        dict: Low flow metrics including total days, percentages, and yearly start/end dates below 10th and 5th percentiles
    """
    try:
        if len(series) < 2:  # Need at least 2 points for trend calculation
            return {
                'percentile_10th': 'N/A',
                'percentile_5th': 'N/A',
                'total_days_below_10th_per_year': 'N/A',
                'total_days_below_5th_per_year': 'N/A',
                'yearly_low_flow_periods_10th': {},
                'yearly_low_flow_periods_5th': {},
                'yearly_significant_periods_10th': {},
                'yearly_significant_periods_5th': {},
                'analysis_method': 'Insufficient data'
            }
        
        # Step 1: Set rolling window based on resampling frequency if not provided
        if rolling_window is None:
            rolling_window = ROLLING_WINDOW_SIZE
        
        # Step 2: Apply rolling average to smooth noise (same as duration analysis)
        if len(series) >= rolling_window and rolling_window > 1:
            smoothed_series = series.rolling(window=rolling_window, center=False).mean()
            # Don't fill NaN values - let them remain NaN at edges
            # This is the correct behavior for rolling means
        else:
            smoothed_series = series
        
        # Step 3: Calculate adaptive threshold based on data variability (same as duration analysis)
        changes = smoothed_series.diff()
        changes_std = changes.std()
        if pd.isna(changes_std) or changes_std == 0:
            # Fallback to fixed threshold if no variability
            adaptive_threshold = 0.01  # 1cm default
        else:
            # Use half standard deviation as threshold (more robust)
            adaptive_threshold = 0.5 * changes_std
        
        # Step 4: Calculate percentile thresholds from ORIGINAL data (not smoothed)
        # This ensures we detect single-day events like December 18th
        clean_original_data = series.dropna()
        percentile_10th = np.percentile(clean_original_data, 10)
        percentile_5th = np.percentile(clean_original_data, 5)
        
        # Step 5: Create hybrid detection - start with original data, then use rolling mean for extension
        # This ensures we detect single-day events like December 18th that get smoothed out
        
        # First, identify ALL potential periods from original data (including single-day events)
        below_10th_original = series < percentile_10th
        below_5th_original = series < percentile_5th
        
        # Then, identify periods from rolling mean for potential extension
        below_10th_smoothed = smoothed_series < percentile_10th
        below_5th_smoothed = smoothed_series < percentile_5th
        
        # Combine both approaches: start with original data periods, then extend using rolling mean
        def create_hybrid_condition(original_condition, smoothed_condition, original_series, percentile_threshold):
            """
            Create hybrid condition that starts with original data periods and extends them using rolling mean.
            This ensures we don't miss single-day events like December 18th.
            """
            # Start with original data condition
            hybrid_condition = original_condition.copy()
            
            # Find groups of consecutive True values in original condition
            groups = (original_condition != original_condition.shift()).cumsum()
            consecutive_lengths = original_condition.groupby(groups).sum()
            
            # Get only periods where condition was True
            true_periods = consecutive_lengths[consecutive_lengths > 0]
            
            for group_idx in true_periods.index:
                group_mask = groups == group_idx
                group_dates = original_condition[group_mask].index
                
                if len(group_dates) > 0:
                    start_date = group_dates[0]
                    end_date = group_dates[-1]
                    
                    # Look for adjacent rolling mean periods that can extend this period
                    # Check a few days before and after the original period
                    search_start = start_date - pd.Timedelta(days=3)
                    search_end = end_date + pd.Timedelta(days=3)
                    
                    # Get rolling mean data in this extended range
                    extended_smoothed = smoothed_condition[search_start:search_end]
                    
                    # Find consecutive True values in rolling mean that overlap with our search range
                    extended_groups = (extended_smoothed != extended_smoothed.shift()).cumsum()
                    extended_lengths = extended_smoothed.groupby(extended_groups).sum()
                    
                    for ext_group_idx in extended_lengths.index:
                        if extended_lengths[ext_group_idx] > 0:
                            ext_group_mask = extended_groups == ext_group_idx
                            ext_group_dates = extended_smoothed[ext_group_mask].index
                            
                            if len(ext_group_dates) > 0:
                                ext_start = ext_group_dates[0]
                                ext_end = ext_group_dates[-1]
                                
                                # Check if this rolling mean period overlaps with or is adjacent to our original period
                                if (ext_start <= end_date + pd.Timedelta(days=7) and 
                                    ext_end >= start_date - pd.Timedelta(days=7)):
                                    
                                    # Extend the hybrid condition to include this rolling mean period
                                    hybrid_condition[ext_start:ext_end] = True
                                    print(f"   🔗 Extending period: {start_date.strftime('%d-%b-%Y')} to {end_date.strftime('%d-%b-%Y')} with rolling mean: {ext_start.strftime('%d-%b-%Y')} to {ext_end.strftime('%d-%b-%Y')}")
            
            return hybrid_condition
        
        # Apply hybrid approach
        below_10th = create_hybrid_condition(below_10th_original, below_10th_smoothed, series, percentile_10th)
        below_5th = create_hybrid_condition(below_5th_original, below_5th_smoothed, series, percentile_5th)
        
        # Step 5: Convert minimum duration to periods based on resampling frequency
        min_duration_periods = 1  # Set to 1 to detect all events including single-day
        
        def find_all_low_flow_periods(condition_series, min_duration, original_series, percentile_threshold, adaptive_threshold):
            """
            Find ALL low flow periods with precise start and end dates using hybrid approach.
            Uses smoothed data to identify periods, then daily data for precise dates.
            Ensures non-overlapping periods by merging overlapping ones.
            
            NEW APPROACH: Uses rolling mean dates as reference points, then calculates
            daily averages from original data without shifting dates.
            
            Args:
                condition_series: Boolean series indicating low flow periods (from smoothed data)
                min_duration: Minimum consecutive periods to count as a significant low flow event
                original_series: Original daily series for precise date calculation
                percentile_threshold: The percentile threshold value for precise date calculation
            
            Returns:
                tuple: (all_periods_dict, significant_periods_dict)
            """

            all_periods = {}
            significant_periods = {}
            

            
            if condition_series.sum() == 0:
                return all_periods, significant_periods
            
            # Find consecutive periods using smoothed data
            groups = (condition_series != condition_series.shift()).cumsum()
            consecutive_lengths = condition_series.groupby(groups).sum()
            
            # Get only periods where condition was True
            true_periods = consecutive_lengths[consecutive_lengths > 0]
            
            if len(true_periods) == 0:
                return all_periods, significant_periods
            
            # Group periods by year and find precise start and end dates
            yearly_raw_periods = {}
            
            for group_idx in true_periods.index:
                # Get the date range where this group occurs (from smoothed data)
                group_mask = groups == group_idx
                group_dates = condition_series[group_mask].index
                
                if len(group_dates) > 0:
                    smoothed_start_date = group_dates[0]
                    smoothed_end_date = group_dates[-1]
                    year = smoothed_start_date.year
                    duration_periods = len(group_dates)
                    search_start = smoothed_start_date
                    search_end = smoothed_end_date
                    
                    # Get the daily data in this range (from rolling mean dates)
                    daily_range = original_series[search_start:search_end]
                    
                    # For each day in the range, calculate daily average and check threshold
                    # This handles cases with multiple readings per day (e.g., 15-minute intervals)
                    daily_averages = []
                    daily_dates = []
                    
                    # Group by date and calculate daily averages from original data
                    for date in pd.date_range(search_start, search_end, freq='D'):
                        if date in original_series.index:
                            # Get all readings for this specific date
                            day_data = original_series[original_series.index.date == date.date()]
                            if len(day_data) > 0:
                                daily_avg = day_data.mean()
                                daily_averages.append(daily_avg)
                                daily_dates.append(date)
                

                    
                    # Calculate duration in days using rolling mean dates (not shifted)
                    duration_days = (smoothed_end_date - smoothed_start_date).days + 1
                    
                    # Calculate mean groundwater level using daily averages for this period
                    period_daily_averages = []
                    for date in pd.date_range(smoothed_start_date, smoothed_end_date, freq='D'):
                        if date in original_series.index:
                            day_data = original_series[original_series.index.date == date.date()]
                            if len(day_data) > 0:
                                period_daily_averages.append(day_data.mean())
                    
                    if period_daily_averages:
                        mean_level = np.mean(period_daily_averages)
                    else:
                        mean_level = original_series[smoothed_start_date:smoothed_end_date].mean()
                    
                    # Initialize year if not exists
                    if year not in yearly_raw_periods:
                        yearly_raw_periods[year] = []
                    
                    # Add to raw periods
                    yearly_raw_periods[year].append({
                        'start_date': smoothed_start_date,  
                        'end_date': smoothed_end_date,      
                        'duration_days': duration_days,
                        'duration_periods': duration_periods,
                        'is_significant': duration_periods >= min_duration,
                        'mean_level': mean_level,
                        'smoothed_start': smoothed_start_date.strftime('%d-%b-%Y'),
                        'smoothed_end': smoothed_end_date.strftime('%d-%b-%Y')
                    })
            
            # Merge overlapping periods for each year
            for year, raw_periods in yearly_raw_periods.items():
                if raw_periods:
                    # Sort by start date
                    raw_periods.sort(key=lambda x: x['start_date'])
                    
                    merged_periods = []
                    current_period = raw_periods[0].copy()
                    
                    for next_period in raw_periods[1:]:
                        # Check if periods overlap or are adjacent  
                        gap_days = (next_period['start_date'] - current_period['end_date']).days
                        # Allow merging if gap is 7 days or less (hydrological continuity)
                        if next_period['start_date'] <= current_period['end_date'] + pd.Timedelta(days=7):
                            # Merge periods
                            current_period['end_date'] = max(current_period['end_date'], next_period['end_date'])
                            current_period['duration_days'] = (current_period['end_date'] - current_period['start_date']).days + 1
                            current_period['duration_periods'] = max(current_period['duration_periods'], next_period['duration_periods'])
                            current_period['is_significant'] = current_period['is_significant'] or next_period['is_significant']
                            # Recalculate mean for merged period
                            merged_period_data = original_series[current_period['start_date']:current_period['end_date']]
                            current_period['mean_level'] = merged_period_data.mean()
                        else:
                            # No overlap, add current period and start new one
                            merged_periods.append(current_period)
                            current_period = next_period.copy()
                    
                    # Add the last period
                    merged_periods.append(current_period)
                    
                    # Convert to final format
                    all_periods[year] = []
                    significant_periods[year] = []
                    
                    for period in merged_periods:
                        # Add to all periods
                        all_periods[year].append({
                            'start_date': period['start_date'].strftime('%d-%b-%Y'),
                            'end_date': period['end_date'].strftime('%d-%b-%Y'),
                            'duration_days': period['duration_days'],
                            'duration_periods': period['duration_periods'],
                            'is_significant': period['is_significant'],
                            'mean_level': period['mean_level'],
                            'smoothed_start': period['smoothed_start'],
                            'smoothed_end': period['smoothed_end']
                        })
                        
                        # Add to significant periods if meets minimum duration
                        if period['is_significant']:
                            significant_periods[year].append({
                                'start_date': period['start_date'].strftime('%d-%b-%Y'),
                                'end_date': period['end_date'].strftime('%d-%b-%Y'),
                                'duration_days': period['duration_days'],
                                'mean_level': period['mean_level']
                            })
            

            return all_periods, significant_periods
                
        # Step 6: Find all low flow periods and significant periods using hybrid approach
        all_periods_10th, significant_periods_10th = find_all_low_flow_periods(below_10th, min_duration_periods, series, percentile_10th, adaptive_threshold)
        all_periods_5th, significant_periods_5th = find_all_low_flow_periods(below_5th, min_duration_periods, series, percentile_5th, adaptive_threshold)
        
        # Step 7: Calculate annual averages in calendar days
        years_in_data = below_10th.index.year.unique()
        total_years = len(years_in_data)
               
        # Calculate average days per year properly
        if total_years > 0:
            # Count periods below threshold for each year and average
            annual_below_10th_by_year = []
            annual_below_5th_by_year = []
            
            for year in years_in_data:
                year_below_10th = below_10th[below_10th.index.year == year].sum()
                year_below_5th = below_5th[below_5th.index.year == year].sum()
                
                # Total days are now already in calendar days since we expanded to daily resolution
                year_days_10th = year_below_10th
                year_days_5th = year_below_5th
                
                # Sanity check - no year should exceed 366 days
                max_days_in_year = 366
                if year_days_10th > max_days_in_year:
                    year_days_10th = max_days_in_year
                if year_days_5th > max_days_in_year:
                    year_days_5th = max_days_in_year
                
                annual_below_10th_by_year.append(year_days_10th)
                annual_below_5th_by_year.append(year_days_5th)
            
            # Calculate averages
            annual_days_10th = np.mean(annual_below_10th_by_year)
            annual_days_5th = np.mean(annual_below_5th_by_year)
            
            # Range check - ensure values are reasonable
            max_possible_days = 366  # Maximum days per year
            if annual_days_10th > max_possible_days:
                print(f"Warning: Annual days below 10th percentile ({annual_days_10th:.1f}) exceeds maximum possible days per year")
                annual_days_10th = min(annual_days_10th, max_possible_days)
            
            if annual_days_5th > max_possible_days:
                print(f"Warning: Annual days below 5th percentile ({annual_days_5th:.1f}) exceeds maximum possible days per year")
                annual_days_5th = min(annual_days_5th, max_possible_days)
        else:
            annual_days_10th = 0
            annual_days_5th = 0
        
        # Step 8: Generate analysis method description
        smoothing_description = f"Rolling mean smoothing (window={rolling_window})"
        analysis_method = f"{smoothing_description}, total day counting approach, hybrid date calculation (smoothed identification + daily precision)"
        
        return {
            'percentile_10th': f"{percentile_10th:.3f} m",
            'percentile_5th': f"{percentile_5th:.3f} m",
            'total_days_below_10th_per_year': f"{annual_days_10th:.1f} days/year",
            'total_days_below_5th_per_year': f"{annual_days_5th:.1f} days/year",
            'yearly_low_flow_periods_10th': all_periods_10th,  # All periods, not just significant ones
            'yearly_low_flow_periods_5th': all_periods_5th,    # All periods, not just significant ones
            'yearly_significant_periods_10th': significant_periods_10th,  # Periods meeting minimum duration
            'yearly_significant_periods_5th': significant_periods_5th,    # Periods meeting minimum duration
            'analysis_method': analysis_method,
            'rolling_window_periods': rolling_window,
            'min_duration_periods': min_duration_periods,
            'resampling_frequency': resampling_freq,
            'percentile_calculation': 'Based on smoothed data for noise reduction',
            'counting_method': 'Total days below threshold (all periods included), hybrid date calculation (smoothed identification + daily precision)',
            'original_data': series  # Add original data for overall statistics
        }
        
    except Exception as e:
        print(f"Percentile low flow calculation error: {e}")
        return {
            'percentile_10th': 'Error',
            'percentile_5th': 'Error', 
            'total_days_below_10th_per_year': 'Error',
            'total_days_below_5th_per_year': 'Error',
            'yearly_low_flow_periods_10th': {},
            'yearly_low_flow_periods_5th': {},
            'yearly_significant_periods_10th': {},
            'yearly_significant_periods_5th': {},
            'analysis_method': f'Error: {str(e)}'
        }


__all__ = [
    'calculate_yearly_max_duration_metrics',
    'calculate_percentile_low_flow_metrics',
]
