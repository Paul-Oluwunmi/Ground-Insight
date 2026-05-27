"""
CSV Export Functions

This module contains functions for exporting Mann-Kendall analysis results to CSV files.
"""

import pandas as pd
import os
from ..utils import (
    generate_well_display_name,
    generate_well_filename,
    get_nz_season
)
from ..constants import TIME_EVOLUTION_WINDOW_SIZE

def save_basic_mk_analysis_csv(results, save_dir, mapping_dict=None):
    """
    Save basic Mann-Kendall analysis results to CSV.
    
    Args:
        results: MK analysis results dictionary
        save_dir: Directory to save the CSV
        mapping_dict: Dictionary mapping well names to display names
    
    Returns:
        str: Path to saved CSV file
    """
    try:
        if not results or 'error' in results:
            return None
        
        # Create filename
        filename = "basic_mk_analysis.csv"
        file_path = os.path.join(save_dir, filename)
        
        # Prepare data for CSV - include ALL wells, even those with errors
        csv_data = []
        for well_name, well_results in results.items():
            well_display = generate_well_display_name(well_name, mapping_dict)
            
            # Check if this well has an error
            has_error = 'error' in well_results
            
            row = {
                'Well': well_display,
                'Trend': well_results.get('trend', 'ERROR' if has_error else 'N/A'),
                'P-value': well_results.get('p_value', 'ERROR' if has_error else 'N/A'),
                'Tau': well_results.get('tau', 'ERROR' if has_error else 'N/A'),
                'Z-score': well_results.get('z_score', 'ERROR' if has_error else 'N/A'),
                'Slope (m/day)': well_results.get('slope', 'ERROR' if has_error else 'N/A'),
                'Interpretation': well_results.get('interpretation', 'ERROR' if has_error else 'N/A'),
                'N Points': well_results.get('n_points', 'ERROR' if has_error else 'N/A'),
                'Original N Points': well_results.get('original_n_points', 'ERROR' if has_error else 'N/A'),
                'Rolling Window': well_results.get('rolling_window_size', 'ERROR' if has_error else 'N/A'),
                'Error Message': well_results.get('error', 'None'),
                'Mean Level (m)': well_results.get('mean_level', 'ERROR' if has_error else 'N/A'),
                'Min Level (m)': well_results.get('min_level', 'ERROR' if has_error else 'N/A'),
                'Max Level (m)': well_results.get('max_level', 'ERROR' if has_error else 'N/A'),
                'Std Level (m)': well_results.get('std_level', 'ERROR' if has_error else 'N/A'),
                'Start Date': well_results.get('start_date', 'ERROR' if has_error else 'N/A'),
                'End Date': well_results.get('end_date', 'ERROR' if has_error else 'N/A')
            }
            csv_data.append(row)
        
        if csv_data:
            df = pd.DataFrame(csv_data)
            df.to_csv(file_path, index=False)
            print(f"Saved basic MK analysis CSV: {file_path}")
            return file_path
        else:
            print("No valid data to save for basic MK analysis")
            return None
            
    except Exception as e:
        print(f"Error saving basic MK analysis CSV: {e}")
        return None

def save_duration_analysis_csv(results, save_dir, well_name, mapping_dict=None):
    """
    Save duration analysis results to CSV files for a single well.
    Creates both individual well files and prepares data for all-wells combined file.
    
    Args:
        results: Duration analysis results dictionary
        save_dir: Directory to save the CSV
        mapping_dict: Dictionary mapping well names to display names
        well_name: Name of the well
    
    Returns:
        dict: Dictionary with file paths and data for all-wells combined
    """
    try:
        if 'error' in results:
            return None
        
        # Get well display name
        well_display = generate_well_display_name(well_name, mapping_dict)
        
        # Create filename prefix
        filename_prefix = generate_well_filename(well_name, "_duration_analysis", mapping_dict)
        
        saved_files = []
        all_wells_data = {}
        
        # 1. SUMMARY STATISTICS CSV
        summary_data = [{
            'Statistic': 'Total Days Declining Per Year',
            'Value': results.get('total_days_declining_per_year', 'N/A'),
            'Interpretation': ''
        }, {
            'Statistic': 'Total Days Rising Per Year',
            'Value': results.get('total_days_rising_per_year', 'N/A'),
            'Interpretation': ''
        }, {
            'Statistic': 'Maximum Declining Duration',
            'Value': results.get('avg_max_declining_duration', 'N/A'),
            'Interpretation': ''
        }, {
            'Statistic': 'Minimum Declining Duration',
            'Value': results.get('avg_min_declining_duration', 'N/A'),
            'Interpretation': ''
        }, {
            'Statistic': 'Maximum Rising Duration',
            'Value': results.get('avg_max_rising_duration', 'N/A'),
            'Interpretation': ''
        }, {
            'Statistic': 'Minimum Rising Duration',
            'Value': results.get('avg_min_rising_duration', 'N/A'),
            'Interpretation': ''
        }, {
            'Statistic': 'Years Analysed',
            'Value': results.get('years_analysed', 'N/A'),
            'Interpretation': ''
        }, {
            'Statistic': 'Analysis Method',
            'Value': results.get('analysis_method', 'N/A'),
            'Interpretation': ''
        }]
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = os.path.join(save_dir, f"{filename_prefix}_summary.csv")
        summary_df.to_csv(summary_file, index=False)
        saved_files.append(summary_file)
        
        # Store for all-wells combined
        all_wells_data['summary'] = summary_data
        
        # 2. YEARLY BREAKDOWN CSV
        if 'yearly_declining_periods' in results or 'yearly_rising_periods' in results:
            yearly_data = []
            
            # Get years from either declining or rising periods
            years_declining = list(results.get('yearly_declining_periods', {}).keys())
            years_rising = list(results.get('yearly_rising_periods', {}).keys())
            all_years = sorted(set(years_declining + years_rising))
            
            if all_years:
                # Get the original data series to calculate overall stats
                well_data = results.get('original_data', None)
                if well_data is not None:
                    # Calculate overall maximum and minimum with seasons
                    overall_max = well_data.max()
                    overall_max_date = well_data.idxmax()
                    overall_max_season = get_nz_season(overall_max_date)
                    
                    overall_min = well_data.min()
                    overall_min_date = well_data.idxmin()
                    overall_min_season = get_nz_season(overall_min_date)
                    
                    # Count occurrences of max and min by season from yearly data
                    max_season_counts = {}
                    min_season_counts = {}
                    
                    for year in all_years:
                        # Count by year for now - you can modify this logic as needed
                        max_season_counts[year] = max_season_counts.get(year, 0) + 1
                        min_season_counts[year] = min_season_counts.get(year, 0) + 1
                    
                    # Format season counts (simplified for now)
                    max_season_summary = ', '.join([f"Year {year} {count}" for year, count in max_season_counts.items()])
                    min_season_summary = ', '.join([f"Year {year} {count}" for year, count in min_season_counts.items()])
                else:
                    # Fallback values if original data not available
                    overall_max = overall_max_season = overall_max_date = "N/A"
                    overall_min = overall_min_season = overall_min_date = "N/A"
                    max_season_summary = min_season_summary = "N/A"
            else:
                overall_max = overall_max_season = overall_max_date = "N/A"
                overall_min = overall_min_season = overall_min_date = "N/A"
                max_season_summary = min_season_summary = "N/A"
            
            # Get date ranges for declining and rising periods
            if 'yearly_declining_periods' in results and results['yearly_declining_periods']:
                periods_declining = []
                for year_periods in results['yearly_declining_periods'].values():
                    periods_declining.extend(year_periods)
                if periods_declining:
                    first_declining = periods_declining[0]
                    last_declining = periods_declining[-1]
                    declining_date_range = f"{first_declining['start_date']} to {last_declining['end_date']}"
                else:
                    declining_date_range = "No declining periods"
            else:
                declining_date_range = "No declining periods"
            
            if 'yearly_rising_periods' in results and results['yearly_rising_periods']:
                periods_rising = []
                for year_periods in results['yearly_rising_periods'].values():
                    periods_rising.extend(year_periods)
                if periods_rising:
                    first_rising = periods_rising[0]
                    last_rising = periods_rising[-1]
                    rising_date_range = f"{first_rising['start_date']} to {last_rising['end_date']}"
                else:
                    rising_date_range = "No rising periods"
            else:
                rising_date_range = "No rising periods"
            
            # Create yearly data entries
            for year in all_years:
                periods_declining_count = len(results.get('yearly_declining_periods', {}).get(year, []))
                periods_rising_count = len(results.get('yearly_rising_periods', {}).get(year, []))
                
                # Calculate total days from periods
                total_days_declining = sum(p['duration_days'] for p in results.get('yearly_declining_periods', {}).get(year, []))
                total_days_rising = sum(p['duration_days'] for p in results.get('yearly_rising_periods', {}).get(year, []))
                
                yearly_data.append({
                    'Year': year,
                    'Total Declining (days)': total_days_declining,
                    'Total Rising (days)': total_days_rising,
                    'Declining Periods': periods_declining_count,
                    'Rising Periods': periods_rising_count,
                    'Declining Periods Date Range': declining_date_range if year == all_years[0] else '',
                    'Rising Periods Date Range': rising_date_range if year == all_years[0] else ''
                })
                
                # Add summary statistics to the first row only
                if year == all_years[0]:
                    
                    yearly_data[-1].update({
                        'Overall_Max_Level_m': overall_max,
                        'Overall_Max_Season': overall_max_season,
                        'Overall_Max_Date': overall_max_date.strftime('%b-%Y') if hasattr(overall_max_date, 'strftime') else overall_max_date,
                        'Overall_Max_Description': 'Highest water level recorded',
                        'Overall_Min_Level_m': overall_min,
                        'Overall_Min_Season': overall_min_season,
                        'Overall_Min_Date': overall_min_date.strftime('%b-%Y') if hasattr(overall_min_date, 'strftime') else overall_min_date,
                        'Overall_Min_Description': 'Lowest water level recorded',
                        'Max_Season_Count': max_season_summary,
                        'Min_Season_Count': min_season_summary,
                        'Declining Periods Date Range': declining_date_range,
                        'Rising Periods Date Range': rising_date_range
                    })
            
            yearly_df = pd.DataFrame(yearly_data)
            yearly_file = os.path.join(save_dir, f"{filename_prefix}_yearly_breakdown.csv")
            yearly_df.to_csv(yearly_file, index=False)
            saved_files.append(yearly_file)
            
            # Store for all-wells combined
            all_wells_data['yearly'] = yearly_data
        
        # 3. DECLINING PERIODS DETAILED BREAKDOWN CSV
        if 'yearly_declining_periods' in results and results['yearly_declining_periods']:
            declining_data = []
            for year, year_periods in results['yearly_declining_periods'].items():
                for period in year_periods:
                    declining_data.append({
                        'Year': year,
                        'Start Date': period['start_date'],
                        'End Date': period['end_date'],
                        'Duration (Days)': period['duration_days']
                    })
            
            declining_df = pd.DataFrame(declining_data)
            declining_file = os.path.join(save_dir, f"{filename_prefix}_declining_periods.csv")
            declining_df.to_csv(declining_file, index=False)
            saved_files.append(declining_file)
            
            # Store for all-wells combined
            all_wells_data['declining'] = declining_data
        
        # 4. RISING PERIODS DETAILED BREAKDOWN CSV
        if 'yearly_rising_periods' in results and results['yearly_rising_periods']:
            rising_data = []
            for year, year_periods in results['yearly_rising_periods'].items():
                for period in year_periods:
                    rising_data.append({
                        'Year': year,
                        'Start Date': period['start_date'],
                        'End Date': period['end_date'],
                        'Duration (Days)': period['duration_days']
                    })
            
            rising_df = pd.DataFrame(rising_data)
            rising_file = os.path.join(save_dir, f"{filename_prefix}_rising_periods.csv")
            rising_df.to_csv(rising_file, index=False)
            saved_files.append(rising_file)
            
            # Store for all-wells combined
            all_wells_data['rising'] = rising_data
        
        print(f"Saved duration analysis CSV files for {well_display}: {len(saved_files)} files")
        
        return {
            'saved_files': saved_files,
            'all_wells_data': all_wells_data,
            'well_name': well_display
        }
        
    except Exception as e:
        print(f"Error saving duration analysis CSV for {well_name}: {e}")
        return None


def save_yearly_water_level_statistics_csv(results, save_dir, well_name, mapping_dict=None):
    """
    Save yearly water level statistics results to CSV files for a single well.
    
    Args:
        results: Yearly water level statistics results dictionary
        save_dir: Directory to save the CSV
        mapping_dict: Dictionary mapping well names to display names
        well_name: Name of the well
    
    Returns:
        dict: Dictionary with file paths and data for all-wells combined
    """
    try:
        if 'error' in results:
            return None
        
        # Get well display name
        well_display = generate_well_display_name(well_name, mapping_dict)
        
        # Create filename prefix
        filename_prefix = generate_well_filename(well_name, "_yearly_water_level_statistics", mapping_dict)
        
        saved_files = []
        all_wells_data = {}
        
        # YEARLY WATER LEVEL STATISTICS CSV
        if 'yearly_stats' in results:
            yearly_data = []
            
            # Add summary statistics row first
            if results['yearly_stats']:
                first_stat = results['yearly_stats'][0]
                summary_row = {
                    'Well': well_display,
                    'Year': 'SUMMARY STATISTICS',
                    'Season': '---',
                    'Maximum_Level_m': '---',
                    'Minimum_Level_m': '---',
                    'Mean_Level_m': '---',
                    'Overall_Max_Level_m': first_stat.get('Overall_Max_Level_m', 'N/A'),
                    'Overall_Max_Season': first_stat.get('Overall_Max_Season', 'N/A'),
                    'Overall_Max_Date': first_stat.get('Overall_Max_Date', 'N/A'),
                    'Overall_Max_Description': first_stat.get('Overall_Max_Description', 'N/A'),
                    'Overall_Min_Level_m': first_stat.get('Overall_Min_Level_m', 'N/A'),
                    'Overall_Min_Season': first_stat.get('Overall_Min_Season', 'N/A'),
                    'Overall_Min_Date': first_stat.get('Overall_Min_Date', 'N/A'),
                    'Overall_Min_Description': first_stat.get('Overall_Min_Description', 'N/A'),
                    'Max_Season_Count': first_stat.get('Max_Season_Count', 'N/A'),
                    'Min_Season_Count': first_stat.get('Min_Season_Count', 'N/A')
                }
                yearly_data.append(summary_row)
            
            # Add regular yearly data
            for stat in results['yearly_stats']:
                yearly_data.append({
                    'Well': well_display,
                    'Year': stat['year'],
                    'Season': stat['season'],
                    'Maximum_Level_m': stat['max_level_m'],
                    'Minimum_Level_m': stat['min_level_m'],
                    'Mean_Level_m': stat['mean_level_m'],
                    'Overall_Max_Level_m': '',
                    'Overall_Max_Season': '',
                    'Overall_Max_Date': '',
                    'Overall_Max_Description': '',
                    'Overall_Min_Level_m': '',
                    'Overall_Min_Season': '',
                    'Overall_Min_Date': '',
                    'Overall_Min_Description': '',
                    'Max_Season_Count': '',
                    'Min_Season_Count': ''
                })
            
            yearly_df = pd.DataFrame(yearly_data)
            yearly_file = os.path.join(save_dir, f"{filename_prefix}.csv")
            yearly_df.to_csv(yearly_file, index=False)
            saved_files.append(yearly_file)
            
            # Store for all-wells combined
            all_wells_data['yearly'] = yearly_data
        
        print(f"Saved yearly water level statistics CSV files for {well_display}: {len(saved_files)} files")
        return {
            'saved_files': saved_files,
            'all_wells_data': all_wells_data,
            'well_name': well_display
        }
        
    except Exception as e:
        print(f"Error saving yearly water level statistics CSV for {well_name}: {e}")
        return None


def save_low_flow_analysis_csv(results, save_dir, well_name, mapping_dict=None):
    """
    Save low flow analysis results to CSV files for a single well.
    Creates both individual well files and prepares data for all-wells combined file.
    
    Args:
        results: Low flow analysis results dictionary
        save_dir: Directory to save the CSV
        mapping_dict: Dictionary mapping well names to display names
        well_name: Name of the well
    
    Returns:
        dict: Dictionary with file paths and data for all-wells combined
    """
    try:
        if 'error' in results:
            return None
        
        # Get well display name
        well_display = generate_well_display_name(well_name, mapping_dict)
        
        # Create filename prefix
        filename_prefix = generate_well_filename(well_name, "_low_flow_analysis", mapping_dict)
        
        saved_files = []
        all_wells_data = {}
        
        # 1. YEARLY LOW FLOW STATISTICS CSV
        if 'yearly_low_flow' in results:
            yearly_data = []
            
            # Calculate overall seasonal summary statistics
            if 'yearly_low_flow' in results and results['yearly_low_flow']:
                # Get the original data series to calculate overall stats
                well_data = results.get('original_data', None)
                if well_data is not None:
                    # Calculate overall maximum and minimum with seasons
                    overall_max = well_data.max()
                    overall_max_date = well_data.idxmax()
                    overall_max_season = get_nz_season(overall_max_date)
                    
                    overall_min = well_data.min()
                    overall_min_date = well_data.idxmin()
                    overall_min_season = get_nz_season(overall_min_date)
                    
                    # Count occurrences of max and min by season from yearly data
                    max_season_counts = {}
                    min_season_counts = {}
                    
                    for stat in results['yearly_low_flow']:
                        # For low flow analysis, we'll use the year to determine season
                        # This is a simplified approach - you may want to refine this
                        year = stat['year']
                        # Count by year for now - you can modify this logic as needed
                        max_season_counts[year] = max_season_counts.get(year, 0) + 1
                        min_season_counts[year] = min_season_counts.get(year, 0) + 1
                    
                    # Format season counts (simplified for now)
                    max_season_summary = ', '.join([f"Year {year} {count}" for year, count in max_season_counts.items()])
                    min_season_summary = ', '.join([f"Year {year} {count}" for year, count in min_season_counts.items()])
                else:
                    # Fallback values if original data not available
                    overall_max = overall_max_season = overall_max_date = "N/A"
                    overall_min = overall_min_season = overall_min_date = "N/A"
                    max_season_summary = min_season_summary = "N/A"
            else:
                overall_max = overall_max_season = overall_max_date = "N/A"
                overall_min = overall_min_season = overall_min_date = "N/A"
                max_season_summary = min_season_summary = "N/A"
            
            # Get date ranges for yearly statistics
            if 'continuous_periods_10th' in results and results['continuous_periods_10th']:
                first_10th = results['continuous_periods_10th'][0]
                last_10th = results['continuous_periods_10th'][-1]
                date_range_10th = f"{first_10th['start_date'].strftime('%d-%b-%Y')} to {last_10th['end_date'].strftime('%d-%b-%Y')}"
            else:
                date_range_10th = "No continuous periods"
            
            if 'continuous_periods_5th' in results and results['continuous_periods_5th']:
                first_5th = results['continuous_periods_5th'][0]
                last_5th = results['continuous_periods_5th'][-1]
                date_range_5th = f"{first_5th['start_date'].strftime('%d-%b-%Y')} to {last_5th['end_date'].strftime('%d-%b-%Y')}"
            else:
                date_range_5th = "No continuous periods"
            
            # Add summary statistics to the first yearly entry
            for i, stat in enumerate(results['yearly_low_flow']):
                yearly_data.append({
                    'Well': well_display,
                    'Year': stat['year'],
                    'Days Below 10th Percentile': stat['days_below_10th_percentile'],
                    'Days Below 5th Percentile': stat['days_below_5th_percentile'],
                    '10th Percentile Threshold': stat['percentile_10th_threshold'],
                    '5th Percentile Threshold': stat['percentile_5th_threshold'],
                    'Continuous Periods Below 10th Percentile': stat.get('continuous_periods_10th', 0),
                    'Continuous Periods Below 5th Percentile': stat.get('continuous_periods_5th', 0)
                })
                
                # Add summary statistics to the first row only
                if i == 0:
                    yearly_data[-1].update({
                        'Overall_Max_Level_m': overall_max,
                        'Overall_Max_Season': overall_max_season,
                        'Overall_Max_Date': overall_max_date.strftime('%b-%Y') if hasattr(overall_max_date, 'strftime') else overall_max_date,
                        'Overall_Max_Description': 'Highest water level recorded',
                        'Overall_Min_Level_m': overall_min,
                        'Overall_Min_Season': overall_min_season,
                        'Overall_Min_Date': overall_min_date.strftime('%b-%Y') if hasattr(overall_min_date, 'strftime') else overall_min_date,
                        'Overall_Min_Description': 'Lowest water level recorded',
                        'Max_Season_Count': max_season_summary,
                        'Min_Season_Count': min_season_summary
                    })
            
            yearly_df = pd.DataFrame(yearly_data)
            yearly_file = os.path.join(save_dir, f"{filename_prefix}_yearly_statistics.csv")
            yearly_df.to_csv(yearly_file, index=False)
            saved_files.append(yearly_file)
            
            # Store for all-wells combined
            all_wells_data['yearly'] = yearly_data
        
        # 2. 10TH PERCENTILE DETAILED CSV
        if 'yearly_low_flow_periods_10th' in results and results['yearly_low_flow_periods_10th']:
            percentile_10th_data = []
            for year, year_periods in results['yearly_low_flow_periods_10th'].items():
                for period in year_periods:
                    percentile_10th_data.append({
                        'Well': well_display,
                        'Year': year,
                        'Start Date': period['start_date'],
                        'End Date': period['end_date'],
                        'Duration (Days)': period['duration_days'],
                        'Percentile Threshold': '10th',
                        'Threshold Value (m)': results['percentile_10th']
                    })
            
            percentile_10th_df = pd.DataFrame(percentile_10th_data)
            percentile_10th_file = os.path.join(save_dir, f"{filename_prefix}_10th_percentile.csv")
            percentile_10th_df.to_csv(percentile_10th_file, index=False)
            saved_files.append(percentile_10th_file)
            
            # Store for all-wells combined
            all_wells_data['percentile_10th'] = percentile_10th_data
        
        # 3. 5TH PERCENTILE DETAILED CSV
        if 'yearly_low_flow_periods_5th' in results and results['yearly_low_flow_periods_5th']:
            percentile_5th_data = []
            for year, year_periods in results['yearly_low_flow_periods_5th'].items():
                for period in year_periods:
                    percentile_5th_data.append({
                        'Well': well_display,
                        'Year': year,
                        'Start Date': period['start_date'],
                        'End Date': period['end_date'],
                        'Duration (Days)': period['duration_days'],
                        'Percentile Threshold': '5th',
                        'Threshold Value (m)': results['percentile_5th']
                    })
            
            percentile_5th_df = pd.DataFrame(percentile_5th_data)
            percentile_5th_file = os.path.join(save_dir, f"{filename_prefix}_5th_percentile.csv")
            percentile_5th_df.to_csv(percentile_5th_file, index=False)
            saved_files.append(percentile_5th_file)
            
            # Store for all-wells combined
            all_wells_data['percentile_5th'] = percentile_5th_data
        
        # 4. SUMMARY STATISTICS CSV
        # Get date ranges for summary
        if 'yearly_low_flow_periods_10th' in results and results['yearly_low_flow_periods_10th']:
            periods_10th = []
            for year_periods in results['yearly_low_flow_periods_10th'].values():
                periods_10th.extend(year_periods)
            if periods_10th:
                first_10th = periods_10th[0]
                last_10th = periods_10th[-1]
                date_range_10th = f"{first_10th['start_date']} to {last_10th['end_date']}"
            else:
                date_range_10th = "No continuous periods"
        else:
            date_range_10th = "No continuous periods"
        
        if 'yearly_low_flow_periods_5th' in results and results['yearly_low_flow_periods_5th']:
            periods_5th = []
            for year_periods in results['yearly_low_flow_periods_5th'].values():
                periods_5th.extend(year_periods)
            if periods_5th:
                first_5th = periods_5th[0]
                last_5th = periods_5th[-1]
                date_range_5th = f"{first_5th['start_date']} to {last_5th['end_date']}"
            else:
                date_range_5th = "No continuous periods"
        else:
            date_range_5th = "No continuous periods"
        
        # Calculate total days and periods
        total_days_10th = 0
        total_days_5th = 0
        total_periods_10th = 0
        total_periods_5th = 0
        
        if 'yearly_low_flow_periods_10th' in results:
            for year_periods in results['yearly_low_flow_periods_10th'].values():
                total_periods_10th += len(year_periods)
                total_days_10th += sum(p['duration_days'] for p in year_periods)
        
        if 'yearly_low_flow_periods_5th' in results:
            for year_periods in results['yearly_low_flow_periods_5th'].values():
                total_periods_5th += len(year_periods)
                total_days_5th += sum(p['duration_days'] for p in year_periods)
        
        summary_data = [{
            'Well': well_display,
            '10th Percentile Threshold': results['percentile_10th'],
            '5th Percentile Threshold': results['percentile_5th'],
            'Total Days Below 10th Percentile': total_days_10th,
            'Total Days Below 5th Percentile': total_days_5th,
            'Continuous Periods Below 10th Percentile': total_periods_10th,
            'Continuous Periods Below 5th Percentile': total_periods_5th,
            '10th Percentile Date Range': date_range_10th,
            '5th Percentile Date Range': date_range_5th,
            'Analysis Method': results.get('analysis_method', 'N/A')
        }]
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = os.path.join(save_dir, f"{filename_prefix}_summary.csv")
        summary_df.to_csv(summary_file, index=False)
        saved_files.append(summary_file)
        
        # Store for all-wells combined
        all_wells_data['summary'] = summary_data
        
        print(f"Saved low flow analysis CSV files for {well_display}: {len(saved_files)} files")
        return {
            'saved_files': saved_files,
            'all_wells_data': all_wells_data,
            'well_name': well_display
        }
        
    except Exception as e:
        print(f"Error saving low flow analysis CSV for {well_name}: {e}")
        return None


def create_all_wells_duration_analysis_csv(all_wells_results, save_dir, mapping_dict=None):
    """
    Create combined all-wells CSV files for duration analysis.
    
    Args:
        all_wells_results: List of results from individual well analysis
        save_dir: Directory to save the combined CSV
        mapping_dict: Dictionary mapping well names to display names
    
    Returns:
        list: List of saved combined CSV file paths
    """
    try:
        saved_files = []
        
        # Group data by analysis type
        all_summaries = []
        all_yearly = []
        all_declining = []
        all_rising = []
        
        for result in all_wells_results:
            if result and 'all_wells_data' in result:
                data = result['all_wells_data']
                
                # Add well name to each record
                well_name = result['well_name']
                
                if 'summary' in data:
                    for summary in data['summary']:
                        if isinstance(summary, dict):
                            summary_copy = summary.copy()
                            summary_copy['Well'] = well_name
                            all_summaries.append(summary_copy)
                        else:
                            # Handle non-dict summary items
                            all_summaries.append({
                                'Well': well_name,
                                'Data': str(summary)
                            })
                
                if 'yearly' in data:
                    for yearly in data['yearly']:
                        if isinstance(yearly, dict):
                            yearly_copy = yearly.copy()
                            yearly_copy['Well'] = well_name
                            all_yearly.append(yearly_copy)
                        else:
                            # Handle non-dict yearly items
                            all_yearly.append({
                                'Well': well_name,
                                'Data': str(yearly)
                            })
                
                if 'declining' in data:
                    for declining in data['declining']:
                        if isinstance(declining, dict):
                            declining_copy = declining.copy()
                            declining_copy['Well'] = well_name
                            all_declining.append(declining_copy)
                        else:
                            # Handle non-dict declining items
                            all_declining.append({
                                'Well': well_name,
                                'Data': str(declining)
                            })
                
                if 'rising' in data:
                    for rising in data['rising']:
                        if isinstance(rising, dict):
                            rising_copy = rising.copy()
                            rising_copy['Well'] = well_name
                            all_rising.append(rising_copy)
                        else:
                            # Handle non-dict rising items
                            all_rising.append({
                                'Well': well_name,
                                'Data': str(rising)
                            })
        
        # Save combined CSV files
        if all_summaries:
            summary_df = pd.DataFrame(all_summaries)
            # Reorder columns to put 'Well' first
            if 'Well' in summary_df.columns:
                other_columns = [col for col in summary_df.columns if col != 'Well']
                summary_df = summary_df[['Well'] + other_columns]
            summary_file = os.path.join(save_dir, "all_wells_duration_analysis_summary.csv")
            summary_df.to_csv(summary_file, index=False)
            saved_files.append(summary_file)
            # Clean up intermediate dataframe
            del summary_df
        
        if all_yearly:
            yearly_df = pd.DataFrame(all_yearly)
            # Reorder columns to put 'Well' first
            if 'Well' in yearly_df.columns:
                other_columns = [col for col in yearly_df.columns if col != 'Well']
                yearly_df = yearly_df[['Well'] + other_columns]
            yearly_file = os.path.join(save_dir, "all_wells_duration_analysis_yearly.csv")
            yearly_df.to_csv(yearly_file, index=False)
            saved_files.append(yearly_file)
            # Clean up intermediate dataframe
            del yearly_df
        
        if all_declining:
            declining_df = pd.DataFrame(all_declining)
            # Reorder columns to put 'Well' first
            if 'Well' in declining_df.columns:
                other_columns = [col for col in declining_df.columns if col != 'Well']
                declining_df = declining_df[['Well'] + other_columns]
            declining_file = os.path.join(save_dir, "all_wells_duration_analysis_declining_periods.csv")
            declining_df.to_csv(declining_file, index=False)
            saved_files.append(declining_file)
            # Clean up intermediate dataframe
            del declining_df
        
        if all_rising:
            rising_df = pd.DataFrame(all_rising)
            # Reorder columns to put 'Well' first
            if 'Well' in rising_df.columns:
                other_columns = [col for col in rising_df.columns if col != 'Well']
                rising_df = rising_df[['Well'] + other_columns]
            rising_file = os.path.join(save_dir, "all_wells_duration_analysis_rising_periods.csv")
            rising_df.to_csv(rising_file, index=False)
            saved_files.append(rising_file)
            # Clean up intermediate dataframe
            del rising_df
        
        # Clean up intermediate data lists
        del all_summaries, all_yearly, all_declining, all_rising
        
        print(f"Created {len(saved_files)} combined all-wells duration analysis CSV files")
        return saved_files
        
    except Exception as e:
        print(f"Error creating all-wells duration analysis CSV: {e}")
        return []


def create_all_wells_yearly_water_level_statistics_csv(all_wells_results, save_dir, mapping_dict=None):
    """
    Create combined all-wells CSV files for yearly water level statistics.
    
    Args:
        all_wells_results: Dictionary where keys are well names and values are results from individual well analysis
        save_dir: Directory to save the combined CSV
        mapping_dict: Dictionary mapping well names to display names
    
    Returns:
        list: List of saved combined CSV file paths
    """
    try:
        saved_files = []
        
        # Group data by analysis type
        all_yearly = []
        
        # Handle the case where all_wells_results is a dictionary (well_name -> results)
        if isinstance(all_wells_results, dict):
            for well_name, result in all_wells_results.items():
                if result is None:
                    print(f"  ⚠ Skipping {well_name}: result is None")
                    continue
                    
                if isinstance(result, dict) and 'error' in result:
                    print(f"  ⚠ Skipping {well_name}: {result['error']}")
                    continue
                
                # Check if result has the expected structure
                if isinstance(result, dict) and 'yearly_stats' in result:
                    yearly_stats = result['yearly_stats']
                    if isinstance(yearly_stats, list):
                        for yearly in yearly_stats:
                            if isinstance(yearly, dict):
                                yearly_copy = yearly.copy()
                                well_display = generate_well_display_name(well_name, mapping_dict)
                                yearly_copy['Well'] = well_display
                                all_yearly.append(yearly_copy)
                            else:
                                # Handle non-dict yearly items
                                well_display = generate_well_display_name(well_name, mapping_dict)
                                all_yearly.append({
                                    'Well': well_display,
                                    'Data': str(yearly)
                                })
                    else:
                        print(f"  ⚠ {well_name}: yearly_stats is not a list, got {type(yearly_stats)}")
                else:
                    print(f"  ⚠ {well_name}: unexpected result structure: {type(result)}")
                    if isinstance(result, dict):
                        print(f"     Available keys: {list(result.keys())}")
        
        # Save combined CSV files
        if all_yearly:
            yearly_df = pd.DataFrame(all_yearly)
            
            # Reorder columns to put 'Well' first
            if 'Well' in yearly_df.columns:
                # Get all columns except 'Well', then put 'Well' first
                other_columns = [col for col in yearly_df.columns if col != 'Well']
                yearly_df = yearly_df[['Well'] + other_columns]
            
            yearly_file = os.path.join(save_dir, "all_wells_yearly_water_level_statistics.csv")
            yearly_df.to_csv(yearly_file, index=False)
            saved_files.append(yearly_file)
            print(f"  ✓ Created combined yearly statistics CSV: {len(all_yearly)} records")
            # Clean up intermediate dataframe
            del yearly_df
        else:
            print("  ⚠ No yearly statistics data available for combined CSV")
        
        # Clean up intermediate data list
        del all_yearly
        
        print(f"Created {len(saved_files)} combined all-wells yearly water level statistics CSV files")
        return saved_files
        
    except Exception as e:
        print(f"Error creating all-wells yearly water level statistics CSV: {e}")
        import traceback
        traceback.print_exc()
        return []


def create_all_wells_low_flow_analysis_csv(all_wells_results, save_dir, mapping_dict=None):
    """
    Create combined all-wells CSV files for low flow analysis.
    
    Args:
        all_wells_results: List of results from individual well analysis
        save_dir: Directory to save the combined CSV
        mapping_dict: Dictionary mapping well names to display names
    
    Returns:
        list: List of saved combined CSV file paths
    """
    try:
        saved_files = []
        
        # Group data by analysis type
        all_summaries = []
        all_yearly = []
        all_percentile_10th = []
        all_percentile_5th = []
        
        for result in all_wells_results:
            if result and 'all_wells_data' in result:
                data = result['all_wells_data']
                
                # Add well name to each record
                well_name = result['well_name']
                
                if 'summary' in data:
                    for summary in data['summary']:
                        if isinstance(summary, dict):
                            summary_copy = summary.copy()
                            summary_copy['Well'] = well_name
                            all_summaries.append(summary_copy)
                        else:
                            # Handle non-dict summary items
                            all_summaries.append({
                                'Well': well_name,
                                'Data': str(summary)
                            })
                
                if 'yearly' in data:
                    for yearly in data['yearly']:
                        if isinstance(yearly, dict):
                            yearly_copy = yearly.copy()
                            yearly_copy['Well'] = well_name
                            all_yearly.append(yearly_copy)
                        else:
                            # Handle non-dict yearly items
                            all_yearly.append({
                                'Well': well_name,
                                'Data': str(yearly)
                            })
                
                if 'percentile_10th' in data:
                    for period in data['percentile_10th']:
                        if isinstance(period, dict):
                            period_copy = period.copy()
                            period_copy['Well'] = well_name
                            all_percentile_10th.append(period_copy)
                        else:
                            # Handle non-dict period items
                            all_percentile_10th.append({
                                'Well': well_name,
                                'Data': str(period)
                            })
                
                if 'percentile_5th' in data:
                    for period in data['percentile_5th']:
                        if isinstance(period, dict):
                            period_copy = period.copy()
                            period_copy['Well'] = well_name
                            all_percentile_5th.append(period_copy)
                        else:
                            # Handle non-dict period items
                            all_percentile_5th.append({
                                'Well': well_name,
                                'Data': str(period)
                            })
        
        # Save combined CSV files
        if all_summaries:
            summary_df = pd.DataFrame(all_summaries)
            # Reorder columns to put 'Well' first
            if 'Well' in summary_df.columns:
                other_columns = [col for col in summary_df.columns if col != 'Well']
                summary_df = summary_df[['Well'] + other_columns]
            summary_file = os.path.join(save_dir, "all_wells_low_flow_analysis_summary.csv")
            summary_df.to_csv(summary_file, index=False)
            saved_files.append(summary_file)
            # Clean up intermediate dataframe
            del summary_df
        
        if all_yearly:
            yearly_df = pd.DataFrame(all_yearly)
            # Reorder columns to put 'Well' first
            if 'Well' in yearly_df.columns:
                other_columns = [col for col in yearly_df.columns if col != 'Well']
                yearly_df = yearly_df[['Well'] + other_columns]
            yearly_file = os.path.join(save_dir, "all_wells_low_flow_analysis_yearly.csv")
            yearly_df.to_csv(yearly_file, index=False)
            saved_files.append(yearly_file)
            # Clean up intermediate dataframe
            del yearly_df
        
        if all_percentile_10th:
            percentile_10th_df = pd.DataFrame(all_percentile_10th)
            # Reorder columns to put 'Well' first
            if 'Well' in percentile_10th_df.columns:
                other_columns = [col for col in percentile_10th_df.columns if col != 'Well']
                percentile_10th_df = percentile_10th_df[['Well'] + other_columns]
            percentile_10th_file = os.path.join(save_dir, "all_wells_low_flow_analysis_10th_percentile.csv")
            percentile_10th_df.to_csv(percentile_10th_file, index=False)
            saved_files.append(percentile_10th_file)
            # Clean up intermediate dataframe
            del percentile_10th_df
        
        if all_percentile_5th:
            percentile_5th_df = pd.DataFrame(all_percentile_5th)
            # Reorder columns to put 'Well' first
            if 'Well' in percentile_5th_df.columns:
                other_columns = [col for col in percentile_5th_df.columns if col != 'Well']
                percentile_5th_df = percentile_5th_df[['Well'] + other_columns]
            percentile_5th_file = os.path.join(save_dir, "all_wells_low_flow_analysis_5th_percentile.csv")
            percentile_5th_df.to_csv(percentile_5th_file, index=False)
            saved_files.append(percentile_5th_file)
            # Clean up intermediate dataframe
            del percentile_5th_df
        
        # Clean up intermediate data lists
        del all_summaries, all_yearly, all_percentile_10th, all_percentile_5th
        
        print(f"Created {len(saved_files)} combined all-wells low flow analysis CSV files")
        return saved_files
        
    except Exception as e:
        print(f"Error creating all-wells low flow analysis CSV: {e}")
        return []


def save_simple_time_evolution_analysis_csv(results, save_dir, well_name, mapping_dict=None):
    """
    Save time evolution analysis results to CSV for a single well.
    
    Args:
        results: Analysis results dictionary
        save_dir: Directory to save the CSV
        mapping_dict: Dictionary mapping well names to display names
        well_name: Name of the well
    
    Returns:
        str: Path to saved CSV file
    """
    try:
        if 'error' in results or not results.get('windows'):
            return None
        
        # Get well display name
        well_display = generate_well_display_name(well_name, mapping_dict)
        
        # Create filename
        rolling_window = results.get('rolling_window', TIME_EVOLUTION_WINDOW_SIZE)
        step_size = results.get('step_size', 365)
        filename = generate_well_filename(well_name, "_time_evolution_analysis.csv", mapping_dict)
        file_path = os.path.join(save_dir, filename)
        
        # Create DataFrame and save
        df = pd.DataFrame(results['windows'])
        df.to_csv(file_path, index=False)
        
        print(f"Saved time evolution analysis CSV for {well_display}: {file_path}")
        return file_path
        
    except Exception as e:
        print(f"Error saving time evolution CSV for {well_name}: {e}")
        return None


__all__ = [
    'save_basic_mk_analysis_csv',
    'save_duration_analysis_csv',
    'save_yearly_water_level_statistics_csv',
    'save_low_flow_analysis_csv',
    'create_all_wells_duration_analysis_csv',
    'create_all_wells_yearly_water_level_statistics_csv',
    'create_all_wells_low_flow_analysis_csv',
    'save_simple_time_evolution_analysis_csv'
]

