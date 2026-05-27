"""
Seasonal Analysis Module

This module contains functions for seasonal Mann-Kendall analysis and
segmented seasonal analysis.
"""

from itertools import combinations
from ..constants import ALPHA_LEVEL
from ..core_tests import mann_kendall_test


def perform_segmented_enhanced_seasonal_analysis(data, analysis_types, alpha=None, custom_months=None, chunk_size=2):
    """Temporary stub for enhanced seasonal analysis."""
    try:
        # Return dummy results for now
        return {
            'all_trends': [
                {
                    'analysis_type': 'temporal_pairwise_monthly',
                    'period': 'Full_Series',
                    'method': 'Enhanced_Seasonal_MK',
                    'trend': 'no_trend',
                    'tau': 0.0,
                    'p_value': 1.0,
                    'z_score': 0.0,
                    'S': 0,
                    'var_S': 0,
                    'n_points': len(data),
                    'slope': 0.0,
                    'slope_mm_year': 0.0,
                    'intercept': 0.0,
                    'seasonal_group': 'Monthly',
                    'year_range': 'All_Years',
                    'month_range': 'All_Months'
                }
            ]
        }
    except Exception as e:
        print(f"Error in enhanced seasonal analysis: {e}")
        return {}


def perform_comprehensive_seasonal_analysis(series, alpha=None):
    """
    Perform comprehensive seasonal analysis types including individual and pairwise analysis.
    
    Args:
        series: Time series data with datetime index
        alpha: Significance level
    
    Returns:
        dict: Complete results for all analysis types including:
        - temporal_pairwise_monthly: Monthly pairwise comparisons
        - temporal_pairwise_season: Seasonal pairwise comparisons  
        - monthly_individual: Individual month analysis
        - seasonal_individual: Individual season analysis
        - monthly_all_years: Monthly analysis across all years
        - monthly_by_year: Monthly analysis by year
        - nz_seasons_by_year: NZ seasons by year
        - nz_seasons_individual_years: Individual NZ seasons
        - individual_years: Individual year analysis
        - nz_seasons_all_years: NZ seasons across all years
        - decade_windows: Decade-based analysis
    """
    # Use global alpha if none specified
    if alpha is None:
        alpha = ALPHA_LEVEL
        print(f"Using global alpha level: {alpha}")
    
    results = {}
    
    # Get basic data info
    all_months = sorted(series.index.month.unique())
    all_years = sorted(series.index.year.unique())     
    
    print(f"      Data info: {len(all_months)} months, {len(all_years)} years")
    print(f"      Years: {all_years}")
    print(f"      Months: {all_months}")
    
    # NZ Season definitions
    season_months = {
        'Summer': [12, 1, 2],
        'Autumn': [3, 4, 5], 
        'Winter': [6, 7, 8],
        'Spring': [9, 10, 11]
    }
    
    # 1. TEMPORAL PAIRWISE MONTHLY
    monthly_pairwise = []
    print(f"      Monthly pairwise analysis:")
    for month in all_months:
        month_data = series[series.index.month == month]
        month_years = sorted(month_data.index.year.unique())
        
        # Get valid years with sufficient data - RESTORED TO 3 DATA POINTS MINIMUM
        valid_years = []
        for year in month_years:
            year_month_data = month_data[month_data.index.year == year]
            if len(year_month_data) >= 3:  # RESTORED: Minimum 3 data points required
                valid_years.append(year)
        
        print(f"        Month {month}: {len(valid_years)} valid years {valid_years}")
        
        # Generate all pairwise combinations
        for year1, year2 in combinations(valid_years, 2):
            data1 = month_data[month_data.index.year == year1]
            data2 = month_data[month_data.index.year == year2]
            
            if len(data1) >= 3 and len(data2) >= 3:  # RESTORED: Minimum 3 data points required
                mk1 = mann_kendall_test(data1, alpha=alpha, use_rolling_mean=False)
                mk2 = mann_kendall_test(data2, alpha=alpha, use_rolling_mean=False)
                
                # Check if MK tests returned errors
                if 'error' in mk1 or 'error' in mk2:
                    continue  # Skip this pair if there's an error
                
                # Determine overall trend
                if mk1['trend'] == mk2['trend']:
                    overall_trend = mk1['trend']
                elif mk1['trend'] == 'no trend' and mk2['trend'] != 'no trend':
                    overall_trend = mk2['trend']
                elif mk1['trend'] != 'no trend' and mk2['trend'] == 'no trend':
                    overall_trend = mk1['trend']
                else:
                    # Different trends - use stronger one (higher tau)
                    overall_trend = mk1['trend'] if abs(mk1['tau']) > abs(mk2['tau']) else mk2['trend']
                
                month_name = data1.index.month_name()[0] if hasattr(data1.index, 'month_name') else f"Month_{month}"
                
                monthly_pairwise.append({
                    'period': month_name,
                    'year1': year1,
                    'year2': year2,
                    'trend': overall_trend,
                    'year1_tau': mk1['tau'],
                    'year1_p_value': mk1['p_value'],
                    'year1_trend': mk1['trend'],
                    'year2_tau': mk2['tau'],
                    'year2_p_value': mk2['p_value'],
                    'year2_trend': mk2['trend'],
                    'mean': (data1.mean() + data2.mean()) / 2,
                    'n_points': len(data1) + len(data2)
                })
    
    results['temporal_pairwise_monthly'] = monthly_pairwise
    
    # NEW: Individual month analysis for single years (always available)
    monthly_individual = []
    for month in all_months:
        month_data = series[series.index.month == month]
        month_years = sorted(month_data.index.year.unique())
        
        for year in month_years:
            year_month_data = month_data[month_data.index.year == year]
            if len(year_month_data) >= 3:  # RESTORED: Minimum 3 data points required
                # Add Mann-Kendall trend analysis for individual month-year
                mk_result = mann_kendall_test(year_month_data, alpha=alpha, use_rolling_mean=False)
                
                # Check if MK test returned an error
                if 'error' in mk_result:
                    continue  # Skip this month-year if there's an error
                
                month_name = year_month_data.index.month_name()[0] if hasattr(year_month_data.index, 'month_name') else f"Month_{month}"
                monthly_individual.append({
                    'period': month_name,
                    'year': year,
                    'trend': mk_result['trend'],
                    'p_value': mk_result['p_value'],
                    'tau': mk_result['tau'],
                    'S': mk_result.get('S', 0),
                    'var_S': mk_result.get('var_S', None),
                    'min_level': year_month_data.min(),
                    'max_level': year_month_data.max(),
                    'mean_level': year_month_data.mean(),
                    'n_points': len(year_month_data),
                    'analysis_type': 'individual_month'
                })
    
    results['monthly_individual'] = monthly_individual
    
    # 2. TEMPORAL PAIRWISE SEASONAL
    seasonal_pairwise = []
    for season, months in season_months.items():
        season_data = series[series.index.month.isin(months)]
        season_years = sorted(season_data.index.year.unique())
        
        # Get valid years with sufficient data - RESTORED TO 3 DATA POINTS MINIMUM
        valid_years = []
        for year in season_years:
            year_season_data = season_data[season_data.index.year == year]
            if len(year_season_data) >= 3:  # RESTORED: Minimum 3 data points required
                valid_years.append(year)
        
        # Generate all pairwise combinations
        for year1, year2 in combinations(valid_years, 2):
            data1 = season_data[season_data.index.year == year1]
            data2 = season_data[season_data.index.year == year2]
            
            if len(data1) >= 3 and len(data2) >= 3:  # RESTORED: Minimum 3 data points required
                mk1 = mann_kendall_test(data1, alpha=alpha, use_rolling_mean=False)
                mk2 = mann_kendall_test(data2, alpha=alpha, use_rolling_mean=False)
                
                # Check if MK tests returned errors
                if 'error' in mk1 or 'error' in mk2:
                    continue  # Skip this pair if there's an error
                
                # Determine overall trend
                if mk1['trend'] == mk2['trend']:
                    overall_trend = mk1['trend']
                elif mk1['trend'] == 'no trend' and mk2['trend'] != 'no trend':
                    overall_trend = mk2['trend']
                elif mk1['trend'] != 'no trend' and mk2['trend'] == 'no trend':
                    overall_trend = mk1['trend']
                else:
                    # Different trends - use stronger one (higher tau)
                    overall_trend = mk1['trend'] if abs(mk1['tau']) > abs(mk2['tau']) else mk2['trend']
                
                seasonal_pairwise.append({
                    'period': season,
                    'year1': year1,
                    'year2': year2,
                    'trend': overall_trend,
                    'year1_tau': mk1['tau'],
                    'year1_p_value': mk1['p_value'],
                    'year1_trend': mk1['trend'],
                    'year2_tau': mk2['tau'],
                    'year2_p_value': mk2['p_value'],
                    'year2_trend': mk2['trend'],
                    'mean': (data1.mean() + data2.mean()) / 2,
                    'n_points': len(data1) + len(data2)
                })
    
    results['temporal_pairwise_season'] = seasonal_pairwise
    
    # NEW: Individual seasonal analysis for single years (always available)
    seasonal_individual = []
    for season, months in season_months.items():
        season_data = series[series.index.month.isin(months)]
        season_years = sorted(season_data.index.year.unique())
        
        for year in season_years:
            year_season_data = season_data[season_data.index.year == year]
            if len(year_season_data) >= 3:  # RESTORED: Minimum 3 data points required
                # Add Mann-Kendall trend analysis for individual season-year
                mk_result = mann_kendall_test(year_season_data, alpha=alpha, use_rolling_mean=False)
                
                # Check if MK test returned an error
                if 'error' in mk_result:
                    continue  # Skip this season-year if there's an error
                
                seasonal_individual.append({
                    'period': season,
                    'year': year,
                    'trend': mk_result['trend'],
                    'p_value': mk_result['p_value'],
                    'tau': mk_result['tau'],
                    'S': mk_result.get('S', 0),
                    'var_S': mk_result.get('var_S', None),
                    'min_level': year_season_data.min(),
                    'max_level': year_season_data.max(),
                    'mean_level': year_season_data.mean(),
                    'n_points': len(year_season_data),
                    'analysis_type': 'individual_season'
                })
    
    results['seasonal_individual'] = seasonal_individual
    
    # 3. MONTHLY ALL YEARS - Each month across all years combined
    monthly_all_years = []
    for month in all_months:
        month_data = series[series.index.month == month]
        if len(month_data) >= 3:  # RESTORED: Minimum 3 data points required
            mk_result = mann_kendall_test(month_data, alpha=alpha, use_rolling_mean=False)
            
            # Check if MK test returned an error
            if 'error' in mk_result:
                continue  # Skip this month if there's an error
                
            month_name = month_data.index.month_name()[0] if hasattr(month_data.index, 'month_name') else f"Month_{month}"
            
            monthly_all_years.append({
                'period': month_name,
                'year': f"{month_data.index.year.min()}-{month_data.index.year.max()}",
                'trend': mk_result['trend'],
                'p_value': mk_result['p_value'],
                'tau': mk_result['tau'],
                'S': mk_result.get('S', 0),
                'var_S': mk_result.get('var_S', None),
                'min_level': month_data.min(),
                'max_level': month_data.max(),
                'mean_level': month_data.mean(),
                'n_points': len(month_data)
            })
    
    results['monthly_all_years'] = monthly_all_years
    
    # 4. MONTHLY BY YEAR - Each month for each individual year
    monthly_by_year = []
    for month in all_months:
        for year in all_years:
            year_month_data = series[(series.index.month == month) & (series.index.year == year)]
            if len(year_month_data) >= 3:  # RESTORED: Minimum 3 data points required
                mk_result = mann_kendall_test(year_month_data, alpha=alpha, use_rolling_mean=False)
                
                # Check if MK test returned an error
                if 'error' in mk_result:
                    continue  # Skip this month-year if there's an error
                    
                month_name = year_month_data.index.month_name()[0] if hasattr(year_month_data.index, 'month_name') else f"Month_{month}"
                
                monthly_by_year.append({
                    'period': f'{month_name}-{year}',
                    'trend': mk_result['trend'],
                    'p_value': mk_result['p_value'],
                    'tau': mk_result['tau'],
                    'S': mk_result.get('S', 0),
                    'var_S': mk_result.get('var_S', None),
                    'min_level': year_month_data.min(),
                    'max_level': year_month_data.max(),
                    'mean_level': year_month_data.mean(),
                    'n_points': len(year_month_data)
                })
    
    results['monthly_by_year'] = monthly_by_year
    
    # 5. NZ SEASONS BY YEAR - NZ seasons combined by year
    nz_seasons_by_year = []
    for season, months in season_months.items():
        for year in all_years:
            year_season_data = series[(series.index.month.isin(months)) & (series.index.year == year)]
            if len(year_season_data) >= 3:  # RESTORED: Minimum 3 data points required
                mk_result = mann_kendall_test(year_season_data, alpha=alpha, use_rolling_mean=False)
                
                # Check if MK test returned an error
                if 'error' in mk_result:
                    continue  # Skip this season-year if there's an error
                
                nz_seasons_by_year.append({
                    'period': f'{season}-{year}',
                    'trend': mk_result['trend'],
                    'p_value': mk_result['p_value'],
                    'tau': mk_result['tau'],
                    'S': mk_result.get('S', 0),
                    'var_S': mk_result.get('var_S', None),
                    'min_level': year_season_data.min(),
                    'max_level': year_season_data.max(),
                    'mean_level': year_season_data.mean(),
                    'n_points': len(year_season_data)
                })
    
    results['nz_seasons_by_year'] = nz_seasons_by_year
    
    # 6. NZ SEASONS INDIVIDUAL YEARS - Same as above but separate
    results['nz_seasons_individual_years'] = nz_seasons_by_year.copy()
    
    # 7. INDIVIDUAL YEARS - Year-by-year analysis
    individual_years = []
    for year in all_years:
        year_data = series[series.index.year == year]
        if len(year_data) >= 3:  # RESTORED: Minimum 3 data points required
            mk_result = mann_kendall_test(year_data, alpha=alpha, use_rolling_mean=False)
            
            # Check if MK test returned an error
            if 'error' in mk_result:
                continue  # Skip this year if there's an error
            
            individual_years.append({
                'period': str(year),
                'trend': mk_result['trend'],
                'p_value': mk_result['p_value'],
                'tau': mk_result['tau'],
                'S': mk_result.get('S', 0),
                'var_S': mk_result.get('var_S', None),
                'min_level': year_data.min(),
                'max_level': year_data.max(),
                'mean_level': year_data.mean(),
                'n_points': len(year_data)
            })
    
    results['individual_years'] = individual_years
    
    # 8. NZ SEASONS ALL YEARS - NZ seasons across all years combined
    nz_seasons_all_years = []
    for season, months in season_months.items():
        season_data = series[series.index.month.isin(months)]
        if len(season_data) >= 3:  # RESTORED: Minimum 3 data points required
            mk_result = mann_kendall_test(season_data, alpha=alpha, use_rolling_mean=False)
            
            # Check if MK test returned an error
            if 'error' in mk_result:
                continue  # Skip this season if there's an error
            
            nz_seasons_all_years.append({
                'period': season,
                'year': f"{season_data.index.year.min()}-{season_data.index.year.max()}",
                'trend': mk_result['trend'],
                'p_value': mk_result['p_value'],
                'tau': mk_result['tau'],
                'S': mk_result.get('S', 0),
                'var_S': mk_result.get('var_S', None),
                'min_level': season_data.min(),
                'max_level': season_data.max(),
                'mean_level': season_data.mean(),
                'n_points': len(season_data)
            })
    
    results['nz_seasons_all_years'] = nz_seasons_all_years
    
    # 9. DECADE WINDOWS - Decade-based analysis
    decade_windows = []
    decades = {}
    
    for year in all_years:
        decade = (year // 10) * 10
        if decade not in decades:
            decades[decade] = []
        decades[decade].append(year)
    
    for decade_start, years in decades.items():
        decade_end = decade_start + 9
        decade_data = series[series.index.year.isin(years)]
        if len(decade_data) >= 3:  # RESTORED: Minimum 3 data points required
            mk_result = mann_kendall_test(decade_data, alpha=alpha, use_rolling_mean=False)
            
            # Check if MK test returned an error
            if 'error' in mk_result:
                continue  # Skip this decade if there's an error
            
            decade_windows.append({
                'period': f'{decade_start}-{decade_end}',
                'year': f"{decade_start}-{decade_end}",
                'trend': mk_result['trend'],
                'p_value': mk_result['p_value'],
                'tau': mk_result['tau'],
                'S': mk_result.get('S', 0),
                'var_S': mk_result.get('var_S', None),
                'min_level': decade_data.min(),
                'max_level': decade_data.max(),
                'mean_level': decade_data.mean(),
                'n_points': len(decade_data)
            })
    
    results['decade_windows'] = decade_windows
    
    return results


__all__ = [
    'perform_segmented_enhanced_seasonal_analysis',
    'perform_comprehensive_seasonal_analysis'
]
