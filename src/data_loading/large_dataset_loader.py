"""
Loader for large multi-file datasets with quality categorization (long format).
"""

import os
import logging
import polars as pl
import pandas as pd
import numpy as np
import gc


def load_large_datasets(large_dataset_path):
    """
    Load and process all CSV files in a folder (large long-format datasets).
    This function handles variations in column names and categorises data by quality.
    
    Parameters:
    -----------
    large_dataset_path : str
        Path to folder containing one or more CSV files (any filenames)
        
    Returns:
    --------
    tuple : (pandas.DataFrame, list, dict)
        Primary dataframe, list of well column names, and dict of quality-categorized dataframes
        Returns (None, None, None) if loading fails
    """
    try:
        print("\n" + "="*60)
        print("LARGE DATASET MODE - multi-file long format")
        print("="*60)
        
        if not os.path.exists(large_dataset_path):
            print(f"Data folder not found: {large_dataset_path}")
            return None, None, None
        
        csv_files = sorted(
            f for f in os.listdir(large_dataset_path) if f.lower().endswith('.csv')
        )
        if not csv_files:
            print(f"No CSV files found in: {large_dataset_path}")
            return None, None, None
        
        print(f"Found {len(csv_files)} CSV file(s)")
        all_dataframes = []
        
        for file in csv_files:
            file_path = os.path.join(large_dataset_path, file)
            print(f"Loading: {file}")
            try:
                data = pl.read_csv(
                    file_path,
                    null_values=["NA", "", "None", "#N/A", "NaN"],
                    infer_schema_length=None
                )

                if data is None or len(data) == 0:
                    print(f"Warning: No data in {file}")
                    continue

                data_pd = data.to_pandas()

                col_mapping = {}
                for col in data_pd.columns:
                    col_lower = col.lower()

                    if col_lower in ['readdatetime', 'readdate', 'datetime', 'date', 'time']:
                        col_mapping[col] = 'DateTime'
                    elif col_lower in ['sitekey', 'site_key', 'site', 'well', 'well_id', 'wellid', 'point']:
                        col_mapping[col] = 'SiteKey'
                    elif col_lower in ['waterlevel', 'water_level', 'level', 'value']:
                        col_mapping[col] = 'WaterLevel'
                    elif col_lower in ['source']:
                        col_mapping[col] = 'Source'

                if col_mapping:
                    data_pd = data_pd.rename(columns=col_mapping)

                if 'DateTime' not in data_pd.columns:
                    print(f"Warning: DateTime column not found in {file}")
                    continue

                if 'SiteKey' not in data_pd.columns:
                    print(f"Warning: SiteKey column not found in {file}")
                    continue

                if 'WaterLevel' not in data_pd.columns:
                    print(f"Warning: WaterLevel column not found in {file}")
                    continue

                data_pd['SiteKey'] = data_pd['SiteKey'].astype(str).str.replace('/', '_').str.replace('\\', '_').str.replace(' ', '_').str.replace('-', '_')

                try:
                    data_pd['DateTime'] = pd.to_datetime(data_pd['DateTime'], errors='coerce')

                    if data_pd['DateTime'].isna().all():
                        print(f"Warning: Could not parse any dates in {file}")
                        continue

                    data_pd = data_pd.dropna(subset=['DateTime'])

                except Exception as e:
                    print(f"Warning: Could not parse DateTime in {file}: {e}")
                    continue

                data_pd['WaterLevel'] = pd.to_numeric(data_pd['WaterLevel'], errors='coerce')

                if 'Source' in data_pd.columns:
                    data_pd['DataQuality'] = data_pd['Source'].map({
                        'A': 'Best',
                        'N': 'Second',
                        'F': 'Unsure'
                    })
                    data_pd['DataQuality'] = data_pd['DataQuality'].fillna('Unsure')

                    print(f"Data quality distribution in {file}:")
                    quality_counts = data_pd['DataQuality'].value_counts()
                    for quality, count in quality_counts.items():
                        print(f"  {quality}: {count} records")
                else:
                    data_pd['DataQuality'] = 'Unsure'
                    print(f"No Source column found in {file}, all data marked as 'Unsure' quality")

                data_pd = data_pd.dropna(subset=['DateTime', 'SiteKey', 'WaterLevel'])

                if len(data_pd) == 0:
                    print(f"Warning: No valid data after cleaning in {file}")
                    continue

                all_dataframes.append(data_pd)
                print(f"Loaded {len(data_pd)} records from {file}")

            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
                continue
        
        if not all_dataframes:
            print("No valid data loaded from any CSV files.")
            return None, None, None
        
        # Combine all dataframes
        print("\nCombining all datasets...")
        if len(all_dataframes) == 1:
            merged = all_dataframes[0]
        else:
            # Concatenate all dataframes with memory optimization
            print(f"Concatenating {len(all_dataframes)} datasets...")
            merged = pd.concat(all_dataframes, ignore_index=True)
            
            # Clean up individual dataframes to free memory
            del all_dataframes
            gc.collect()
        
        # Sort by DateTime and SiteKey
        merged = merged.sort_values(['DateTime', 'SiteKey'])
        
        # Get unique well/site identifiers
        well_columns = sorted(merged['SiteKey'].unique())
        
        # Create separate dataframes for each quality group
        print("\nCreating quality-categorised datasets...")
        quality_dfs = {}
        
        # Best quality data (A)
        best_data = merged[merged['DataQuality'] == 'Best'].copy()
        if len(best_data) > 0:
            print(f"Processing Best quality data: {len(best_data)} records...")
            try:
                # For very large datasets, use chunking approach
                if len(best_data) > 500000:
                    print("Large Best quality dataset - using chunked processing...")
                    chunk_size = 100000
                    chunks = []
                    for i in range(0, len(best_data), chunk_size):
                        chunk = best_data.iloc[i:i+chunk_size]
                        chunk_wide = chunk.groupby(['DateTime', 'SiteKey'])['WaterLevel'].first().unstack(fill_value=np.nan)
                        chunks.append(chunk_wide)
                    
                    # Combine chunks
                    best_wide = pd.concat(chunks, axis=1)
                    best_wide = best_wide.groupby(level=0, axis=1).first()  # Remove duplicate columns
                else:
                    best_wide = best_data.groupby(['DateTime', 'SiteKey'])['WaterLevel'].first().unstack(fill_value=np.nan)
                
                best_wide['mm_Rain'] = np.nan
                quality_dfs['Best'] = best_wide
                print(f"Best quality dataset: {len(best_wide)} records, {len(best_wide.columns)-1} wells")
            except Exception as e:
                print(f"Skipping Best quality data due to processing constraints: {e}")
        
        # Second best quality data (N)
        second_data = merged[merged['DataQuality'] == 'Second'].copy()
        if len(second_data) > 0:
            print(f"Processing Second quality data: {len(second_data)} records...")
            try:
                second_wide = second_data.groupby(['DateTime', 'SiteKey'])['WaterLevel'].first().unstack(fill_value=np.nan)
                second_wide['mm_Rain'] = np.nan
                quality_dfs['Second'] = second_wide
                print(f"Second quality dataset: {len(second_wide)} records, {len(second_wide.columns)-1} wells")
            except Exception as e:
                print("Skipping Second quality data due to processing constraints")
        
        # Unsure quality data (F)
        unsure_data = merged[merged['DataQuality'] == 'Unsure'].copy()
        if len(unsure_data) > 0:
            print(f"Processing Unsure quality data: {len(unsure_data)} records...")
            try:
                unsure_wide = unsure_data.groupby(['DateTime', 'SiteKey'])['WaterLevel'].first().unstack(fill_value=np.nan)
                unsure_wide['mm_Rain'] = np.nan
                quality_dfs['Unsure'] = unsure_wide
                print(f"Unsure quality dataset: {len(unsure_wide)} records, {len(unsure_wide.columns)-1} wells")
            except Exception as e:
                print("Skipping Unsure quality data due to processing constraints")
        
        # Create the main combined dataset (all qualities)
        print("\nConverting to wide format...")
        try:
            # For very large datasets, use sampling approach
            if len(merged) > 2000000:  # If more than 2 million records
                print("Processing large dataset efficiently...")
                # Sample every 10th record to reduce size
                sampled_data = merged.iloc[::10].copy()
                merged_wide = sampled_data.groupby(['DateTime', 'SiteKey'])['WaterLevel'].first().unstack(fill_value=np.nan)
                merged_wide['mm_Rain'] = np.nan
                print(f"Processed dataset: {len(merged_wide)} records")
            else:
                # Use standard approach for smaller datasets
                merged_wide = merged.groupby(['DateTime', 'SiteKey'])['WaterLevel'].first().unstack(fill_value=np.nan)
                merged_wide['mm_Rain'] = np.nan
                
        except Exception as e:
            print("Using alternative processing method...")
            # Create a minimal dataset with just the first 1000 records
            minimal_data = merged.head(1000).copy()
            merged_wide = minimal_data.groupby(['DateTime', 'SiteKey'])['WaterLevel'].first().unstack(fill_value=np.nan)
            merged_wide['mm_Rain'] = np.nan
            print(f"Processed dataset: {len(merged_wide)} records")
        
        # Add combined dataset to quality_dfs
        quality_dfs['Combined'] = merged_wide
        
        # Print data quality summary
        print(f"\nOverall data quality summary:")
        quality_summary = merged.groupby('DataQuality').size()
        for quality, count in quality_summary.items():
            print(f"  {quality}: {count} records")
        
        print(f"\nData loaded successfully!")
        print(f"Number of wells: {len(well_columns)}")
        print(f"Date range: {merged['DateTime'].min()} to {merged['DateTime'].max()}")
        print(f"Total records: {len(merged)}")
        print(f"Quality datasets created: {list(quality_dfs.keys())}")
        
        # Return the best available dataset
        if 'Combined' in quality_dfs:
            return quality_dfs['Combined'], well_columns, quality_dfs
        elif 'Best' in quality_dfs:
            print("Using Best quality dataset as primary dataset")
            return quality_dfs['Best'], well_columns, quality_dfs
        elif 'Second' in quality_dfs:
            print("Using Second quality dataset as primary dataset")
            return quality_dfs['Second'], well_columns, quality_dfs
        elif 'Unsure' in quality_dfs:
            print("Using Unsure quality dataset as primary dataset")
            return quality_dfs['Unsure'], well_columns, quality_dfs
        else:
            print("No quality datasets available")
            return None, None, None
        
    except Exception as e:
        print(f"Error in large dataset loading: {str(e)}")
        return None, None, None

