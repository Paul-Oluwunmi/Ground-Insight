"""
Statistical calculation functions for groundwater analysis
"""

import numpy as np
from scipy import stats


def calculate_statistics(data1, data2):
    """
    Calculate statistical comparison between two datasets.
    
    Parameters:
    -----------
    data1 : array-like
        First dataset
    data2 : array-like
        Second dataset
        
    Returns:
    --------
    dict : Statistical results including t-statistic, p-value, effect size, etc.
    """
    t_stat, p_value = stats.ttest_ind(data1, data2)
    well1_mean = np.mean(data1)
    well2_mean = np.mean(data2)
    mean_diff = well1_mean - well2_mean
    effect_size = abs(mean_diff) / np.sqrt(
        (np.var(data1) + np.var(data2)) / 2
    )
    
    is_significant = p_value <= 0.05
    
    return {
        't_stat': t_stat,
        'p_value': p_value,
        'mean1': well1_mean,
        'mean2': well2_mean,
        'mean_diff': mean_diff,
        'effect_size': effect_size,
        'is_significant': is_significant
    }


def get_difference_magnitude(t_stat, effect_size):
    """
    Determine the magnitude of difference based on t-statistic and effect size.
    
    Parameters:
    -----------
    t_stat : float
        t-statistic value
    effect_size : float
        Effect size (Cohen's d)
        
    Returns:
    --------
    str : Magnitude description
    """
    t_abs = abs(t_stat)
    if t_abs > 2 and abs(effect_size) >= 0.8:
        direction = "lower" if t_stat < 0 else "higher"
        return f"Major ({direction})"
    elif t_abs > 1 and abs(effect_size) >= 0.2:
        direction = "lower" if t_stat < 0 else "higher"
        return f"Moderate ({direction})"
    else:
        return "Minor"


def get_confidence_level(p_value):
    """
    Determine confidence level based on p-value.
    
    Parameters:
    -----------
    p_value : float
        p-value from statistical test
        
    Returns:
    --------
    str : Confidence level description
    """
    if p_value <= 0.01:
        return "Very confident"
    elif p_value <= 0.05:
        return "Confident"
    else:
        return "No significant difference"


def format_significance_text(well1_name, well1_date, well2_name, well2_date, 
                             mean_diff, stats_result):
    """
    Format the significance interpretation text.
    
    Parameters:
    -----------
    well1_name : str
        Name of first well
    well1_date : str
        Date string for first well
    well2_name : str
        Name of second well
    well2_date : str
        Date string for second well
    mean_diff : float
        Mean difference between wells
    stats_result : dict
        Dictionary from calculate_statistics()
        
    Returns:
    --------
    str : Formatted HTML text
    """
    magnitude = get_difference_magnitude(stats_result['t_stat'], stats_result['effect_size'])
    confidence = get_confidence_level(stats_result['p_value'])
    
    if abs(mean_diff) < 0.001:
        level_difference = (
            f"<b>{well1_name} ({well1_date}) and "
            f"{well2_name} ({well2_date}) are identical</b>"
        )
        
        significance_text = (
            f"{level_difference}<br>"
            f"<b>No difference detected</b><br>"
            f"<b>Measurements match exactly</b>"
        )
    else:
        level_difference = (
            f"<b>{well1_name} ({well1_date}) is {abs(mean_diff):.3f}m "
            f"{'higher' if mean_diff > 0 else 'lower'} than "
            f"{well2_name} ({well2_date})</b>"
        )
        
        significance_text = (
            f"{level_difference}<br>"
            f"<b>{magnitude} difference</b> (t={stats_result['t_stat']:.3f}, d={stats_result['effect_size']:.3f})<br>"
            f"<b>{confidence}</b> (p={stats_result['p_value']:.3f})"
        )
    
    return significance_text

