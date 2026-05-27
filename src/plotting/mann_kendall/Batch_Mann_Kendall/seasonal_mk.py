"""
Seasonal MK Comprehensive Batch Analysis

This module contains the comprehensive batch wrapper function for Seasonal MK analysis.
"""

import pandas as pd
import os
import gc
import traceback
import sys

# Use ONLY absolute imports to avoid relative import issues when loaded dynamically
# Relative imports cause "attempted relative import with no known parent package" errors
# when modules are loaded dynamically, even inside try/except blocks
from src.plotting.mann_kendall.constants import ALPHA_LEVEL
from src.plotting.mann_kendall.analysis import perform_comprehensive_seasonal_analysis
from src.plotting.mann_kendall.utils import (
    ensure_datetime_index,
    generate_well_display_name,
    generate_well_folder_name,
    generate_well_filename,
    get_well_folder_name_consistent
)


def run_comprehensive_seasonal_mk_batch_analysis_separate_files(data, well_columns, mapping_dict=None, 
                                                              save_directory='seasonal_mk_batch_results',
                                                              alpha=0.05, start_date=None, end_date=None, 
                                                              create_folders=False, save_plots=False,
                                                              custom_save_location=None):
    """
    Run comprehensive SEASONAL Mann-Kendall batch analysis with individual and pairwise analysis types.
    """
    print("Running COMPREHENSIVE SEASONAL Mann-Kendall batch analysis...")
    print("Creating individual well files + combined all-wells files")
    print("Using original data frequency (no resampling)")
    
    # Check and fix datetime index FIRST - before any other operations
    if not isinstance(data.index, pd.DatetimeIndex):
        if 'DateTime' in data.columns:
            print("Converting 'DateTime' column to index...")
            # Create a new DataFrame with DateTime as index
            data = data.set_index('DateTime').copy()
            # Convert index to datetime if it's not already
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)
            print("DateTime index set successfully")
        else:
            print("ERROR: Data must have datetime index or 'DateTime' column")
            print("Available columns:", list(data.columns))
            return [], []
    else:
        print("Index is already datetime")
    
    # Create save directory
    if custom_save_location:
        save_dir = custom_save_location
    else:
        save_dir = save_directory
    
    print(f"Save location: {save_dir}")
    
    # Always create the save directory for seasonal analysis
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialise results storage for all analysis types (including new individual ones)
    all_wells_results = {
        'temporal_pairwise_monthly': [],
        'temporal_pairwise_season': [],
        'monthly_individual': [],  # NEW: individual month analysis
        'seasonal_individual': [],  # NEW: individual season analysis
        'monthly_all_years': [],
        'monthly_by_year': [],
        'nz_seasons_by_year': [],
        'nz_seasons_individual_years': [],
        'individual_years': [],
        'nz_seasons_all_years': [],
        'decade_windows': []
    }
    
    # Process each well
    wells_processed = 0
    wells_skipped = 0
    
    for well in well_columns:
        # Get well data
        well_data = data[well].dropna()
        if len(well_data) == 0:
            print(f"  ⚠ {well}: no data after dropna, skipping")
            wells_skipped += 1
            continue
        
        # Check if data has datetime index
        if not isinstance(well_data.index, pd.DatetimeIndex):
            print(f"  ⚠ {well}: no datetime index, skipping")
            wells_skipped += 1
            continue
        
        # Use original data without resampling
        print(f"  Processing {well}: {len(well_data)} original points (no resampling)")
        original_data = well_data
        
        # Get well display name
        well_display = generate_well_display_name(well, mapping_dict)
        
        try:
            # ALL ANALYSIS TYPES IN ONE CALL
            print(f"  Running comprehensive seasonal analysis for {well}...")
            print(f"    Data span: {original_data.index.min().strftime('%Y-%m')} to {original_data.index.max().strftime('%Y-%m')}")
            print(f"    Total years: {len(original_data.index.year.unique())}")
            print(f"    Total months: {len(original_data.index.month.unique())}")
            
            enhanced_results = perform_comprehensive_seasonal_analysis(original_data, alpha)
            
            # Extract results
            monthly_pairwise = enhanced_results.get('temporal_pairwise_monthly', [])
            seasonal_pairwise = enhanced_results.get('temporal_pairwise_season', [])
            monthly_individual = enhanced_results.get('monthly_individual', [])  # NEW
            seasonal_individual = enhanced_results.get('seasonal_individual', [])  # NEW
            monthly_all_years = enhanced_results.get('monthly_all_years', [])
            monthly_by_year = enhanced_results.get('monthly_by_year', [])
            nz_seasons_by_year = enhanced_results.get('nz_seasons_by_year', [])
            nz_seasons_individual_years = enhanced_results.get('nz_seasons_individual_years', [])
            individual_years = enhanced_results.get('individual_years', [])
            nz_seasons_all_years = enhanced_results.get('nz_seasons_all_years', [])
            decade_windows = enhanced_results.get('decade_windows', [])
            
            # Add to all-wells results
            for result in monthly_pairwise:
                result_copy = result.copy()
                result_copy['well'] = well_display
                all_wells_results['temporal_pairwise_monthly'].append(result_copy)
            
            for result in seasonal_pairwise:
                result_copy = result.copy()
                result_copy['well'] = well_display
                all_wells_results['temporal_pairwise_season'].append(result_copy)
            
            # NEW: Add individual analysis results (always available)
            for result in monthly_individual:
                result_copy = result.copy()
                result_copy['well'] = well_display
                all_wells_results['monthly_individual'].append(result_copy)
            
            for result in seasonal_individual:
                result_copy = result.copy()
                result_copy['well'] = well_display
                all_wells_results['seasonal_individual'].append(result_copy)
            
            # Add remaining analyses to all-wells results
            for analysis_type in ['monthly_all_years', 'monthly_by_year', 'nz_seasons_by_year', 
                                'nz_seasons_individual_years', 'individual_years', 'nz_seasons_all_years', 'decade_windows']:
                for result in enhanced_results[analysis_type]:
                    result_copy = result.copy()
                    result_copy['well'] = well_display
                    all_wells_results[analysis_type].append(result_copy)
            
            print(f" All analysis types completed for {well}")
            print(f"   • Monthly pairwise: {len(monthly_pairwise)} (requires 2+ years)")
            print(f"   • Seasonal pairwise: {len(seasonal_pairwise)} (requires 2+ years)")
            print(f"   • Monthly individual: {len(monthly_individual)} (always available)")
            print(f"   • Seasonal individual: {len(seasonal_individual)} (always available)")
            print(f"   • Monthly all years: {len(monthly_all_years)} (combined analysis)")
            print(f"   • Monthly by year: {len(monthly_by_year)} (yearly breakdown)")
            print(f"   • NZ seasons by year: {len(nz_seasons_by_year)} (seasonal yearly)")
            print(f"   • NZ seasons all years: {len(nz_seasons_all_years)} (combined seasons)")
            print(f"   • Individual years: {len(individual_years)} (yearly trends)")
            print(f"   • Decade windows: {len(decade_windows)} (decade trends)")
            
            wells_processed += 1
            
            # Clean up intermediate results for this well
            del enhanced_results, monthly_pairwise, seasonal_pairwise, monthly_individual, seasonal_individual
            del monthly_all_years, monthly_by_year, nz_seasons_by_year, nz_seasons_individual_years
            del individual_years, nz_seasons_all_years, decade_windows
            
        except Exception as e:
            print(f" ✗ Error in seasonal analysis for {well}: {str(e)}")
            traceback.print_exc()
            wells_skipped += 1
            continue
        
        # Clean up well-specific data
        del well_data, original_data
    
    # Save results to CSV files
    try:
        saved_files = []
        
        print("\nCreating your preferred folder structure...")
        print(f"DEBUG: Total wells processed: {wells_processed}")
        print(f"DEBUG: Total wells skipped: {wells_skipped}")
        print(f"DEBUG: All wells results keys: {list(all_wells_results.keys())}")
        for key, results in all_wells_results.items():
            print(f"DEBUG: {key}: {len(results)} results")
        
        # Create main folders
        all_wells_combined_dir = os.path.join(save_dir, "All_Wells_Combined")
        os.makedirs(all_wells_combined_dir, exist_ok=True)
        
        # Create analysis type subfolders in All_Wells_Combined
        analysis_subfolders = {
            'monthly_pairwise': 'temporal_pairwise_monthly',
            'seasonal_pairwise': 'temporal_pairwise_season',
            'monthly_individual': 'monthly_individual',  # NEW
            'seasonal_individual': 'seasonal_individual',  # NEW
            'monthly_all_years': 'monthly_all_years',
            'monthly_by_year': 'monthly_by_year',
            'nz_seasons_by_year': 'nz_seasons_by_year',
            'nz_seasons_individual_years': 'nz_seasons_individual_years',
            'individual_years': 'individual_years',
            'nz_seasons_all_years': 'nz_seasons_all_years',
            'decade_windows': 'decade_windows'
        }
        
        for subfolder, analysis_key in analysis_subfolders.items():
            subfolder_path = os.path.join(all_wells_combined_dir, subfolder)
            os.makedirs(subfolder_path, exist_ok=True)
            
            # Save all-wells combined results for this analysis type
            if all_wells_results[analysis_key]:
                df = pd.DataFrame(all_wells_results[analysis_key])
                
                # Remove 'well' column for decade_windows
                if subfolder == 'decade_windows' and 'well' in df.columns:
                    df = df.drop(columns=['well'])
                
                csv_file = os.path.join(subfolder_path, f"{subfolder}_all_wells.csv")
                df.to_csv(csv_file, index=False)
                saved_files.append(csv_file)
                print(f"✓ All-wells {subfolder} saved: {csv_file} ({len(df)} records)")
            else:
                # More informative message about why no results
                if subfolder in ['monthly_pairwise', 'seasonal_pairwise']:
                    print(f"ℹ {subfolder}: No pairwise results (requires 2+ years with data in same month/season)")
                elif subfolder in ['monthly_individual', 'seasonal_individual']:
                    print(f"ℹ {subfolder}: No individual results (requires at least 1 data point per month/season)")
                elif subfolder in ['monthly_all_years', 'nz_seasons_all_years']:
                    print(f"ℹ {subfolder}: No combined results (requires sufficient data across all years)")
                elif subfolder in ['monthly_by_year', 'nz_seasons_by_year', 'individual_years']:
                    print(f"ℹ {subfolder}: No yearly results (requires data in individual years)")
                elif subfolder == 'decade_windows':
                    print(f"ℹ {subfolder}: No decade results (requires data spanning multiple decades)")
                else:
                    print(f"ℹ {subfolder}: No results available")
        
        # Create individual well folders with detailed breakdowns
        for well in well_columns:
            if well in data.columns:
                well_display = generate_well_folder_name(well, mapping_dict)
                # Use consistent naming for folder structure
                well_dir = os.path.join(save_dir, get_well_folder_name_consistent(well, mapping_dict))
                os.makedirs(well_dir, exist_ok=True)
                print(f"\nCreating detailed structure for {well} (mapped to: {well_display})...")
                
                # Get well's individual results from all_wells_results
                for subfolder, analysis_key in analysis_subfolders.items():
                    analysis_dir = os.path.join(well_dir, subfolder)
                    os.makedirs(analysis_dir, exist_ok=True)
                    
                    # Filter results for this specific well
                    well_results = [r for r in all_wells_results[analysis_key] if r.get('well') == well_display]
                    
                    if well_results:
                        print(f"    Creating {len(well_results)} results for {subfolder}")
                        if subfolder == 'monthly_pairwise':
                            # Create separate files for each month
                            month_groups = {}
                            for result in well_results:
                                month = result.get('period', 'Unknown')
                                if month not in month_groups:
                                    month_groups[month] = []
                                month_groups[month].append(result)
                            
                            for month, month_results in month_groups.items():
                                month_df = pd.DataFrame(month_results)
                                # Remove 'well' column for individual well files
                                if 'well' in month_df.columns:
                                    month_df = month_df.drop(columns=['well'])
                                month_file = os.path.join(analysis_dir, generate_well_filename(well, f"_{month.lower()}_pairwise_results.csv", mapping_dict))
                                month_df.to_csv(month_file, index=False)
                                saved_files.append(month_file)
                                print(f"  ✓ {month} pairwise results saved")
                        
                        elif subfolder == 'seasonal_pairwise':
                            # Create separate files for each season
                            season_groups = {}
                            for result in well_results:
                                season = result.get('period', 'Unknown')
                                if season not in season_groups:
                                    season_groups[season] = []
                                season_groups[season].append(result)
                            
                            for season, season_results in season_groups.items():
                                season_df = pd.DataFrame(season_results)
                                # Remove 'well' column for individual well files
                                if 'well' in season_df.columns:
                                    season_df = season_df.drop(columns=['well'])
                                season_file = os.path.join(analysis_dir, generate_well_filename(well, f"_{season.lower()}_pairwise_results.csv", mapping_dict))
                                season_df.to_csv(season_file, index=False)
                                saved_files.append(season_file)
                                print(f"  ✓ {season} pairwise results saved")
                        
                        elif subfolder == 'monthly_individual':
                            # Create separate files for each month
                            month_groups = {}
                            for result in well_results:
                                month = result.get('period', 'Unknown')
                                if month not in month_groups:
                                    month_groups[month] = []
                                month_groups[month].append(result)
                            
                            for month, month_results in month_groups.items():
                                month_df = pd.DataFrame(month_results)
                                # Remove 'well' column for individual well files
                                if 'well' in month_df.columns:
                                    month_df = month_df.drop(columns=['well'])
                                month_file = os.path.join(analysis_dir, generate_well_filename(well, f"_{month.lower()}_individual_results.csv", mapping_dict))
                                month_df.to_csv(month_file, index=False)
                                saved_files.append(month_file)
                                print(f"  ✓ {month} individual results saved")
                        
                        elif subfolder == 'seasonal_individual':
                            # Create separate files for each season
                            season_groups = {}
                            for result in well_results:
                                season = result.get('period', 'Unknown')
                                if season not in season_groups:
                                    season_groups[season] = []
                                season_groups[season].append(result)
                            
                            for season, season_results in season_groups.items():
                                season_df = pd.DataFrame(season_results)
                                # Remove 'well' column for individual well files
                                if 'well' in season_df.columns:
                                    season_df = season_df.drop(columns=['well'])
                                season_file = os.path.join(analysis_dir, generate_well_filename(well, f"_{season.lower()}_individual_results.csv", mapping_dict))
                                season_df.to_csv(season_file, index=False)
                                saved_files.append(season_file)
                                print(f"  ✓ {season} individual results saved")
                        
                        else:
                            # For other analysis types, create single file
                            analysis_df = pd.DataFrame(well_results)
                            
                            # Remove 'well' column for all individual well files
                            if 'well' in analysis_df.columns:
                                analysis_df = analysis_df.drop(columns=['well'])
                            
                            analysis_file = os.path.join(analysis_dir, generate_well_filename(well, f"_{subfolder}_results.csv", mapping_dict))
                            analysis_df.to_csv(analysis_file, index=False)
                            saved_files.append(analysis_file)
                            print(f"  ✓ {subfolder} results saved")
                    else:
                        print(f"    ⚠ No results for {subfolder} - no files created")
        
        # Create summary file
        summary_data = []
        for well in well_columns:
            if well in data.columns:
                well_data = data[well].dropna()
                if len(well_data) >= 1:
                    summary_data.append({
                        'well': well,
                        'well_display': mapping_dict.get(well, well) if mapping_dict else well,
                        'total_points': len(well_data),
                        'start_date': well_data.index[0],
                        'end_date': well_data.index[-1],
                        'mean_level': well_data.mean(),
                        'min_level': well_data.min(),
                        'max_level': well_data.max(),
                        'std_level': well_data.std()
                    })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            # Reorder columns to put 'Well' first
            if 'Well' in summary_df.columns:
                other_columns = [col for col in summary_df.columns if col != 'Well']
                summary_df = summary_df[['Well'] + other_columns]
            summary_file = os.path.join(save_dir, "Summary_All_Wells.csv")
            summary_df.to_csv(summary_file, index=False)
            saved_files.append(summary_file)
            print(f"\n✓ Summary_All_Wells.csv created with {len(summary_df)} wells")
        
        # Show final result counts
        total_results = 0
        results_by_type = {}
        for analysis_type, results in all_wells_results.items():
            total_results += len(results)
            results_by_type[analysis_type] = len(results)
        
        if total_results == 0:
            print(f"\n⚠️  WARNING: No seasonal analysis results were generated!")
            print(f"   This usually means:")
            print(f"   • Wells have insufficient data for any analysis")
            print(f"   • Data quality issues prevented analysis")
        else:
            print(f"\n✓ Seasonal analysis completed successfully with {total_results} total results")
            print(f"   Results breakdown:")
            for analysis_type, count in results_by_type.items():
                if count > 0:
                    if analysis_type in ['monthly_individual', 'seasonal_individual']:
                        print(f"     • {analysis_type}: {count} results (always available)")
                    elif analysis_type in ['temporal_pairwise_monthly', 'temporal_pairwise_season']:
                        print(f"     • {analysis_type}: {count} results (requires 2+ years)")
                    else:
                        print(f"     • {analysis_type}: {count} results")
            
            print(f"   Individual month/season analysis available for all wells with data")
            print(f"   Pairwise analysis available for wells with multiple years")
        
        print(f"\nYour preferred structure created successfully!")
        print(f"Individual well folders with detailed breakdowns")
        print(f"All-wells combined results for each analysis type")
        print(f"Summary file for overview")
        print(f"Total files created: {len(saved_files)}")
        print(f"Results saved to: {save_dir}")
        
        # Summary of processing
        print(f"\nSEASONAL MK ANALYSIS SUMMARY:")
        print(f"  Wells processed successfully: {wells_processed}")
        print(f"  Wells skipped: {wells_skipped}")
        print(f"  Total wells: {len(well_columns)}")
        
        # Clean up large data structures and force garbage collection
        if 'all_wells_results' in locals():
            del all_wells_results
        if 'summary_data' in locals():
            del summary_data
        if 'summary_df' in locals():
            del summary_df
        
        # Force garbage collection to free memory
        gc.collect()
        
        return saved_files, save_dir
        
    except Exception as e:
        print(f"Error saving results: {e}")
        traceback.print_exc()
        return [], None


__all__ = ['run_comprehensive_seasonal_mk_batch_analysis_separate_files']

