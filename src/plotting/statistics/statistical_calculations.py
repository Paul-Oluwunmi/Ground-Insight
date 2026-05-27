"""
Statistical calculation functions
"""

import numpy as np
from scipy import stats


def calculate_significance(data1, data2):
    """
    Calculate statistical significance and effect size between two datasets.
    
    Parameters:
    -----------
    data1 : array-like
        First dataset
    data2 : array-like
        Second dataset
        
    Returns:
    --------
    tuple : (p_value, effect_size, statistic) or (None, None, None) if error
    """
    try:
        # Convert to numpy arrays and handle NaN values
        data1 = np.array(data1, dtype=float)
        data2 = np.array(data2, dtype=float)
        
        # Remove NaN values
        data1 = data1[~np.isnan(data1)]
        data2 = data2[~np.isnan(data2)]
        
        # Check if we have enough data
        if len(data1) < 2 or len(data2) < 2:
            return None, None, None
            
        # Perform Mann-Whitney U test
        try:
            statistic, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
        except ValueError:
            return None, None, None
        
        # Calculate Cliff's Delta effect size
        n1, n2 = len(data1), len(data2)
        dominance = 0
        
        for i in data1:
            for j in data2:
                if i > j:
                    dominance += 1
                elif i < j:
                    dominance -= 1
                    
        effect_size = dominance / (n1 * n2)
        
        return p_value, effect_size, statistic
        
    except Exception as e:
        print(f"Error in statistical calculation: {str(e)}")
        return None, None, None


def calculate_boxplot_statistics(values, q1_percentile, q3_percentile):
    """
    Calculate statistics for boxplot visualization.
    
    Parameters:
    -----------
    values : array-like
        Data values
    q1_percentile : float
        Lower percentile (default 25)
    q3_percentile : float
        Upper percentile (default 75)
        
    Returns:
    --------
    dict : Dictionary containing mean, median, std, q1, q3, iqr, bounds
    """
    stats_dict = {
        'mean': np.nanmean(values),
        'median': np.nanmedian(values),
        'std': np.nanstd(values),
        'q1': np.nanpercentile(values, q1_percentile),
        'q3': np.nanpercentile(values, q3_percentile)
    }
    stats_dict['iqr'] = stats_dict['q3'] - stats_dict['q1']
    stats_dict['lower_bound'] = stats_dict['q1'] - 1.5 * stats_dict['iqr']
    stats_dict['upper_bound'] = stats_dict['q3'] + 1.5 * stats_dict['iqr']
    
    return stats_dict

