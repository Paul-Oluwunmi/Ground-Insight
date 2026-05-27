"""
POT Dashboard Helper Functions
==============================

Helper functions for dashboard callbacks, including time series plotting
and threshold calculation.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import find_peaks as scipy_find_peaks
import logging

from ..constants import SEASON_COLORS


def create_time_series_plot(data_type, locations, threshold_method, threshold_value, sampling_interval, input_data, is_drought, calculate_threshold_func):
    """
    Create time series plot with threshold and peaks.
    
    Parameters:
    -----------
    data_type : str
        'gwl' or 'rain'
    locations : str or list
        Location(s) to plot
    threshold_method : str
        Method for calculating threshold
    threshold_value : float
        Threshold parameter value
    sampling_interval : str
        Time interval for resampling
    input_data : pd.DataFrame
        Input data with DateTime column
    is_drought : bool
        Whether analyzing drought conditions
    calculate_threshold_func : callable
        Function to calculate threshold
        
    Returns:
    --------
    go.Figure
        Plotly figure with time series and peaks
    """
    if (data_type == 'gwl' and not locations) or threshold_value is None:
        return go.Figure()

    fig = go.Figure()
    unit = 'm' if data_type == 'gwl' else 'mm'  

    try:
        if data_type == 'rain':
            column_to_analyse = 'mm_Rain'
            if 'mm_Rain' not in input_data.columns:
                raise ValueError("Rainfall data column 'mm_Rain' not found")

            if sampling_interval != 'original':
                resampled_data = (input_data[['DateTime', 'mm_Rain']]
                    .set_index('DateTime')
                    .resample(sampling_interval)
                    .sum()
                    .reset_index())
            else:
                resampled_data = input_data[['DateTime', 'mm_Rain']].copy()

            # Calculate threshold (same signature as EVA so time series and POT Analysis stay in sync)
            threshold = calculate_threshold_func(
                resampled_data['mm_Rain'],
                threshold_method,
                threshold_value,
                sampling_interval,
                data_type,
                is_drought
            )
            
            # Plot rainfall data
            fig.add_trace(
                go.Scattergl(
                    x=resampled_data['DateTime'],
                    y=resampled_data['mm_Rain'],
                    mode='lines',
                    name=f'Rainfall ({sampling_interval})',
                    line=dict(color='blue', width=1),
                    hovertemplate='Date: %{x}<br>Rainfall: %{y:.2f} mm<extra></extra>'
                )
            )

            # Add threshold line
            if isinstance(threshold, pd.Series):
                fig.add_trace(
                    go.Scattergl(
                        x=resampled_data['DateTime'],
                        y=threshold,
                        mode='lines',
                        name=f'Threshold',
                        line=dict(color='red', width=1, dash='dash'),
                        hovertemplate='Date: %{x}<br>Threshold: %{y:.2f} mm<extra></extra>'
                    )
                )
            else:
                fig.add_trace(
                    go.Scattergl(
                        x=[resampled_data['DateTime'].iloc[0], resampled_data['DateTime'].iloc[-1]],
                        y=[threshold, threshold],
                        mode='lines',
                        name=f'Threshold ({threshold:.2f} mm)',
                        line=dict(color='red', width=1, dash='dash'),
                        hovertemplate=f'Threshold: {threshold:.2f} mm<extra></extra>'
                    )
                )

            # Find and plot peaks
            peak_indices = scipy_find_peaks(resampled_data['mm_Rain'].values, height=threshold)[0]
            peaks_data = resampled_data.iloc[peak_indices].copy()
            peaks_data['Season'] = peaks_data['DateTime'].dt.month.map({
                12: 'Summer', 1: 'Summer', 2: 'Summer',  # NZ Summer: Dec-Feb
                3: 'Autumn', 4: 'Autumn', 5: 'Autumn',   # NZ Autumn: Mar-May
                6: 'Winter', 7: 'Winter', 8: 'Winter',   # NZ Winter: Jun-Aug
                9: 'Spring', 10: 'Spring', 11: 'Spring'  # NZ Spring: Sep-Nov
            })

            # Plot peaks by season
            for season, color in SEASON_COLORS.items():
                season_mask = peaks_data['Season'] == season
                if any(season_mask):
                    season_peaks = peaks_data[season_mask]
                    fig.add_trace(
                        go.Scattergl(
                            x=season_peaks['DateTime'],
                            y=season_peaks['mm_Rain'],
                            mode='markers',
                            name=f'{season} Peaks',
                            marker=dict(color=color, size=8),
                            hovertemplate='Date: %{x}<br>Rainfall: %{y:.2f} mm<extra></extra>'
                        )
                    )
                            
        else:  # GWL data
            if isinstance(locations, str):
                locations = [locations]
            elif not locations:
                return go.Figure()

            for location in locations:
                column_to_analyse = location  
                if location not in input_data.columns:
                    continue

                # Resample data first
                if sampling_interval != 'original':
                    resampled_data = (input_data[['DateTime', location]]
                        .set_index('DateTime')
                        .resample(sampling_interval)
                        .mean()
                        .reset_index())
                else:
                    resampled_data = input_data[['DateTime', location]].copy()

                # Calculate threshold using resampled data
                threshold = calculate_threshold_func(
                    resampled_data.set_index('DateTime')[location],  
                    threshold_method,
                    threshold_value,
                    sampling_interval,
                    data_type,
                    is_drought
                )

                # Plot GWL data
                fig.add_trace(
                    go.Scattergl(
                        x=resampled_data['DateTime'],
                        y=resampled_data[location],
                        mode='lines',
                        name=location,
                        line=dict(color='blue', width=1),
                        hovertemplate=f'Date: %{{x}}<br>{location}: %{{y:.2f}} m<extra></extra>'
                    )
                )

                # Find peaks based on drought/non-drought analysis
                peak_indices = np.array([], dtype=int)  
                if isinstance(threshold, pd.Series):
                    if is_drought:
                        # For drought analysis with moving average
                        peak_mask = resampled_data[location].values < threshold.values
                        if any(peak_mask):
                            masked_peaks = scipy_find_peaks(
                                -resampled_data[location].values[peak_mask],
                                height=-threshold.values[peak_mask]
                            )[0]
                            if len(masked_peaks) > 0:
                                peak_indices = np.where(peak_mask)[0][masked_peaks]
                        threshold_name = 'Drought Threshold'
                    else:
                        # For high extremes with moving average
                        peak_mask = resampled_data[location].values > threshold.values
                        if any(peak_mask):
                            masked_peaks = scipy_find_peaks(
                                resampled_data[location].values[peak_mask],
                                height=threshold.values[peak_mask]
                            )[0]
                            if len(masked_peaks) > 0:
                                peak_indices = np.where(peak_mask)[0][masked_peaks]
                        threshold_name = 'Threshold'
                else:
                    if is_drought:
                        peaks_result = scipy_find_peaks(-resampled_data[location].values, height=-threshold)
                        if len(peaks_result[0]) > 0:
                            peak_indices = peaks_result[0]
                        threshold_name = f'Drought Threshold ({threshold:.2f} m)'
                    else:
                        peaks_result = scipy_find_peaks(resampled_data[location].values, height=threshold)
                        if len(peaks_result[0]) > 0:
                            peak_indices = peaks_result[0]
                        threshold_name = f'Threshold ({threshold:.2f} m)'
                
                # Get peaks data
                if len(peak_indices) > 0:
                    peaks_data = resampled_data.iloc[peak_indices].copy()
                else:
                    # Create empty DataFrame with correct columns and types
                    peaks_data = pd.DataFrame({
                        'DateTime': pd.Series(dtype='datetime64[ns]'),
                        location: pd.Series(dtype='float64'),
                        'Season': pd.Series(dtype='object')
                    })

                # Add seasonal information only if we have peaks
                if len(peaks_data) > 0:
                    peaks_data['Season'] = peaks_data['DateTime'].dt.month.map({
                        12: 'Summer', 1: 'Summer', 2: 'Summer',  # NZ Summer: Dec-Feb
                        3: 'Autumn', 4: 'Autumn', 5: 'Autumn',   # NZ Autumn: Mar-May
                        6: 'Winter', 7: 'Winter', 8: 'Winter',   # NZ Winter: Jun-Aug
                        9: 'Spring', 10: 'Spring', 11: 'Spring'  # NZ Spring: Sep-Nov
                    })

                # Add threshold line
                if isinstance(threshold, pd.Series):
                    fig.add_trace(
                        go.Scattergl(
                            x=resampled_data['DateTime'],
                            y=threshold,
                            mode='lines',
                            name='Threshold',
                            line=dict(color='red', width=1, dash='dash'),
                            hovertemplate=f'Date: %{{x}}<br>Threshold: %{{y:.2f}} m<extra></extra>'
                        )
                    )

                else:
                    fig.add_trace(
                        go.Scattergl(
                            x=[resampled_data['DateTime'].iloc[0], resampled_data['DateTime'].iloc[-1]],
                            y=[threshold, threshold],
                            mode='lines',
                            name=threshold_name,
                            line=dict(color='red', width=1, dash='dash'),
                            hovertemplate=f'Threshold: {threshold:.2f} m<extra></extra>'
                        )
                    )

                # Plot peaks by season
                for season in SEASON_COLORS:
                    season_mask = peaks_data['Season'] == season
                    if any(season_mask):
                        season_data = peaks_data[season_mask]
                        n_events = len(season_data)  
                        
                        fig.add_trace(
                            go.Scattergl(
                                x=season_data['DateTime'],
                                y=season_data[column_to_analyse],
                                mode='markers',
                                name=f'{season} (n={n_events})',  
                                marker=dict(
                                    color=SEASON_COLORS[season],
                                    size=10,
                                    symbol='circle'
                                ),
                                hovertemplate=(
                                    f"<b>{season}</b><br>" +
                                    "Date: %{x}<br>" +
                                    f"{column_to_analyse}: %{{y:.2f}} {unit}<br>" +  
                                    f"Type: {'Drought' if is_drought else 'Peak'}<br>" +
                                    "<extra></extra>"
                                )
                            )
                        )

        # Update layout
        fig.update_layout(
            title=f"Time Series Analysis ({sampling_interval if sampling_interval != 'original' else 'Original'} Resolution)",
            xaxis=dict(
                title=dict(
                    text='Date',
                    font=dict(size=14, weight='bold')
                ),
                tickfont=dict(weight='bold'),
                rangeslider=dict(visible=True, autorange=True),
                autorange=True
            ),
            yaxis=dict(
                title=dict(
                    text='Groundwater Level (m)' if data_type == 'gwl' else 'Rainfall (mm)',
                    font=dict(size=14, weight='bold')
                ),
                tickfont=dict(weight='bold'),
                autorange=True
            ),
            template='plotly_white',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.02
            ),
            margin=dict(r=150)
        )

        if data_type == 'rain':
            fig.update_yaxes(rangemode='nonnegative')

        return fig
    except Exception as e:
        logging.error(f"Error in time series plot: {str(e)}")
        return go.Figure()


def calculate_threshold(data, method, value, sampling_interval, data_type, is_drought=False):
    """
    Calculate threshold based on selected method with improved robustness and validation.
    
    Parameters:
    -----------
    data : pd.Series
        Series with values (should have DatetimeIndex if using time-based methods)
    method : str
        Threshold calculation method
    value : float
        Threshold parameter value
    sampling_interval : str
        Data sampling interval
    data_type : str
        Either 'gwl' or 'rain'
    is_drought : bool
        Whether to analyze drought conditions (default: False)
        
    Returns:
    --------
    float or pd.Series
        Calculated threshold(s)
    """
    try:
        # Input validation
        if not isinstance(data, pd.Series):
            data = pd.Series(data)
        
        # Remove NaN values
        clean_data = data.dropna()
        if len(clean_data) == 0:
            raise ValueError("No valid data points after removing NaN values")
            
        # Convert to NumPy array for faster operations
        values = clean_data.to_numpy()
        
        # Define unit based on data type
        unit = 'm' if data_type == 'gwl' else 'mm'
        
        # Get data range for validation
        data_min = np.min(values)
        data_max = np.max(values)
        data_range = data_max - data_min
        
        if method == 'percentile':
            if not 0 <= value <= 100:
                logging.warning(f"Percentile value {value} adjusted to be between 0 and 100")
                value = max(0, min(100, value))
            return np.percentile(values, value)
            
        elif method == 'fixed':
            # Get data range
            data_min = np.min(values)
            data_max = np.max(values)
            
            # Adjust value to be within data range
            if value < data_min or value > data_max:
                old_value = value
                value = max(data_min, min(data_max, value))
                logging.warning(
                    f"Fixed threshold {old_value:.2f} adjusted to {value:.2f} to be within "
                    f"data range [{data_min:.2f}, {data_max:.2f}] {unit}"
                )
            
            return float(value)
            
        elif method == 'mean_std':
            if value < 0:
                logging.warning("Negative standard deviation multiplier adjusted to absolute value")
                value = abs(value)
            mean = np.mean(values)
            std = np.std(values)
            # For drought/low flow analysis, subtract standard deviations
            if is_drought:
                return mean - value * std
            # For high extremes, add standard deviations
            return mean + value * std
    
        elif method == 'moving_avg':
            if value < 0:
                logging.warning("Negative standard deviation multiplier adjusted to absolute value")
                value = abs(value)
            
            # Ensure we're working with a pandas Series with DatetimeIndex
            if not isinstance(data, pd.Series):
                data = pd.Series(data)
            
            # Get the data frequency
            if sampling_interval == 'original':
                # Try to infer frequency from the data
                freq = pd.infer_freq(data.index)
                if freq is None:
                    freq = 'D'  # Default to daily if can't infer
            else:
                freq = sampling_interval

            # Calculate window size in number of observations
            window_mapping = {
                'H': 24*30,    # 30 days of hourly data
                'D': 30,       # 30 days
                'W': 4,        # 4 weeks
                'M': 3,        # 3 months
                '3M': 2,       # 2 quarters
                '6M': 2,       # 2 half-years
                'Y': 2,        # 2 years
            }
            
            # Get base window size and adjust if needed
            base_window = window_mapping.get(freq, 30)
            
            # Ensure window size doesn't exceed data length
            window = min(base_window, len(data))
            
            # Calculate rolling statistics
            rolling_mean = data.rolling(
                window=window,
                min_periods=1,
                center=True
            ).mean()
            
            rolling_std = data.rolling(
                window=window,
                min_periods=1,
                center=True
            ).std()
            
            # Fill NaN values at the edges
            rolling_mean = rolling_mean.fillna(method='bfill').fillna(method='ffill')
            rolling_std = rolling_std.fillna(method='bfill').fillna(method='ffill')
            
            # Calculate threshold
            if is_drought:
                threshold = rolling_mean - value * rolling_std
            else:
                threshold = rolling_mean + value * rolling_std
            
            # Ensure threshold has same index as input data
            threshold.index = data.index
            
            return threshold
                            
        elif method == 'seasonal_percentile':
            if not 0 <= value <= 100:
                logging.warning(f"Percentile value {value} adjusted to be between 0 and 100")
                value = max(0, min(100, value))
            
            # For drought analysis, invert the percentile value
            # e.g., if user inputs 90th percentile for drought, we use 10th percentile
            if is_drought:
                value = 100 - value
                logging.info(f"Drought analysis: using {value}th percentile for threshold")
            
            # Define seasons
            seasons = clean_data.index.month.map({
                12: 'Summer', 1: 'Summer', 2: 'Summer',  # Summer: Dec-Feb
                3: 'Autumn', 4: 'Autumn', 5: 'Autumn',   # Autumn: Mar-May
                6: 'Winter', 7: 'Winter', 8: 'Winter',   # Winter: Jun-Aug
                9: 'Spring', 10: 'Spring', 11: 'Spring'  # Spring: Sep-Nov
            })
            
            # Calculate seasonal thresholds
            threshold = clean_data.groupby(seasons).transform(
                lambda x: np.percentile(x.dropna(), value)
            )
            
            # If the original data didn't have a datetime index, return just the values
            if not isinstance(data.index, pd.DatetimeIndex):
                return threshold.values
            return threshold
            
        elif method == 'pot':
            if value <= 0:
                logging.warning("Number of peaks per year adjusted to be positive")
                value = max(1, abs(value))
                
            # Calculate years of data using datetime objects
            if isinstance(clean_data.index, pd.DatetimeIndex):
                time_span = clean_data.index.max() - clean_data.index.min()
                years = time_span.total_seconds() / (365.25 * 24 * 60 * 60)
            else:
                # Fallback if no datetime index
                years = len(clean_data) / 365.25
                
            target_peaks = max(2, int(value * years))  # Ensure at least 2 peaks
            
            if target_peaks >= len(clean_data):
                logging.warning("Requested number of peaks exceeds available data points")
                target_peaks = max(2, len(clean_data) // 2)  # Ensure at least 2 peaks
                
            # Sort based on drought/non-drought condition
            if is_drought:
                sorted_data = np.sort(values)  # Ascending for drought
            else:
                sorted_data = np.sort(values)[::-1]  # Descending for non-drought
                
            return sorted_data[min(target_peaks, len(sorted_data) - 1)]

        elif method == 'declustering':
            if value <= 0:
                logging.warning("Minimum separation time adjusted to be positive")
                value = max(1, abs(value))
                
            # Implement run-length declustering
            min_separation = pd.Timedelta(days=value)
            peaks = []
            last_peak_time = clean_data.index[0] - min_separation
            
            for time, val in clean_data.items():
                if time - last_peak_time >= min_separation:
                    # Define window boundaries
                    window_start = time - min_separation/2
                    window_end = time + min_separation/2
                    
                    # Get window data
                    window_data = clean_data[
                        (clean_data.index >= window_start) & 
                        (clean_data.index <= window_end)
                    ]
                    
                    if len(window_data) > 0:  # Ensure window contains data
                        # For drought, look for minimum; for floods, look for maximum
                        if is_drought:
                            window_extreme = window_data.min()
                            if val <= window_extreme:
                                peaks.append(val)
                                last_peak_time = time
                        else:
                            window_extreme = window_data.max()
                            if val >= window_extreme:
                                peaks.append(val)
                                last_peak_time = time
            
            if not peaks:
                logging.warning("No peaks found with current declustering parameters")
                # For drought use 10th percentile, for floods use 90th
                return np.percentile(values, 10 if is_drought else 90)
            
            # Use a percentile of peaks instead of mean for threshold
            threshold_percentile = 10 if is_drought else 90
            return np.percentile(peaks, threshold_percentile)
            
        elif method == 'min_annual_max':
            if value <= 0:
                logging.warning("Number of years adjusted to be positive")
                value = max(1, abs(value))
                
            # Calculate annual extremes based on drought/non-drought
            if is_drought:
                annual_extremes = clean_data.groupby(clean_data.index.year).min()
            else:
                annual_extremes = clean_data.groupby(clean_data.index.year).max()
            
            if len(annual_extremes) < value:
                logging.warning(f"Insufficient years of data ({len(annual_extremes)} < {value})")
                # Use appropriate percentile for drought/non-drought
                return np.percentile(values, 10 if is_drought else 90)
                
            # Use rolling calculation based on drought/non-drought
            if is_drought:
                # For droughts, use rolling maximum of annual minima
                return annual_extremes.rolling(
                    window=int(value),
                    min_periods=1
                ).max().max()
            else:
                # For floods, use rolling minimum of annual maxima
                return annual_extremes.rolling(
                    window=int(value),
                    min_periods=1
                ).min().min()
    
    except Exception as e:
        logging.error(f"Error in threshold calculation: {str(e)}")
        # Fallback to 90th percentile
        return np.percentile(values[~np.isnan(values)], 90)

