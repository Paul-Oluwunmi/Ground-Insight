"""
Data Processing Module
"""

import polars as pl
from functools import lru_cache
import gc
from datetime import datetime

def cleanup():
    """Local cleanup function"""
    gc.collect()

@lru_cache(maxsize=16)
def load_and_process_data(groundwater_file: str, rainfall_file: str):
    """
    Load and process groundwater and rainfall data using Polars.
    Returns processed DataFrame and well column names.
    """
    try:
        print("\nLoading and processing data...")
        
        # Load groundwater data with optimized settings
        gw_data = (pl.scan_csv(groundwater_file, 
                              null_values=["NA", ""],
                              infer_schema_length=10000)
                   .with_columns([
                       # Convert DateTime with proper format
                       pl.col("DateTime").str.strptime(pl.Datetime, 
                                                     format="%d/%m/%Y %H:%M", 
                                                     strict=False)
                   ])
                   .collect(streaming=True))
        
        # Load rainfall data with explicit schema
        rain_data = (pl.scan_csv(
            rainfall_file,
            null_values=["NA", ""],
            schema={
                "DateTime": pl.Utf8,
                "mm_Rain": pl.Float64
            })
            .with_columns([
                # Convert DateTime with proper format
                pl.col("DateTime").str.strptime(pl.Datetime, 
                                              format="%d/%m/%Y %H:%M", 
                                              strict=False)
            ])
            .collect(streaming=True))
        
        # Get well columns (excluding DateTime and mm_Rain)
        well_columns = [col for col in gw_data.columns if col != "DateTime"]
        
        # Merge groundwater and rainfall data
        merged_data = (gw_data
                      .join(rain_data
                           .select(["DateTime", "mm_Rain"]),
                           on="DateTime",
                           how="left")
                      .sort("DateTime"))
        
        # Basic data quality check (no automatic resampling)
        print("\n=== Data Quality Check ===")
        print("Raw data frequencies:")
        print("  - Groundwater Level (GWL): 15-minute intervals")
        print("  - Rainfall: 10-minute intervals")
        print("  - Merged data: Mixed timestamps preserved")
        
        # Check time intervals
        interval_stats = check_time_intervals(merged_data)
        if "error" not in interval_stats:
            print(f"\nRaw merged data:")
            print(f"  - Total records: {len(merged_data)}")
            print(f"  - Time span: {interval_stats['total_intervals']} intervals")
            print(f"  - Min interval: {interval_stats['min_seconds']} seconds ({interval_stats['min_seconds']/60:.1f} min)")
            print(f"  - Max interval: {interval_stats['max_seconds']} seconds ({interval_stats['max_seconds']/60:.1f} min)")
            print(f"  - Median interval: {interval_stats['median_seconds']} seconds ({interval_stats['median_seconds']/60:.1f} min)")
            if interval_stats['zero_intervals'] > 0:
                print(f"  - Duplicate timestamps: {interval_stats['zero_intervals']}")
        
        # Only clean duplicates if they exist (preserve raw data otherwise)
        if interval_stats.get('zero_intervals', 0) > 0:
            print(f"\nCleaning {interval_stats['zero_intervals']} duplicate timestamps...")
            merged_data = clean_duplicate_timestamps(merged_data, method="mean")
            print("Duplicates removed. Raw data frequencies preserved.")
        else:
            print("\nNo duplicate timestamps found. Raw data preserved as-is.")
        
        print("\nNote: Raw mixed-frequency data preserved. Resampling available if needed by specific analyses.")
        print("=== Raw Data Loading Complete ===\n")
        
        # Convert to pandas for compatibility with existing visualization code
        # Note: Only convert at the final step to maintain Polars optimizations
        pandas_df = merged_data.to_pandas()
        
        # Ensure DateTime is timezone-naive
        pandas_df['DateTime'] = pandas_df['DateTime'].dt.tz_localize(None)
        
        print("\nData loaded and processed successfully!")
        print(f"Number of wells: {len(well_columns)}")
        print("\nWell names:", ", ".join(well_columns))
        print("\nColumns in data:", merged_data.columns)
        
        return pandas_df, well_columns
        
    except Exception as e:
        print(f"Error whilst processing data: {e}")
        return None, None
    finally:
        cleanup()

def validate_data(df: pl.DataFrame) -> bool:
    """
    Validate the loaded data for common issues.
    Returns True if data is valid, False otherwise.
    """
    try:
        # Check for null values
        null_counts = df.null_count()
        if null_counts.sum() > 0:
            print("Warning: Dataset contains null values")
            print(null_counts)
        
        # Check for duplicate timestamps
        duplicates = df.select(pl.col("DateTime")).is_duplicated().sum()
        if duplicates > 0:
            print(f"Warning: Found {duplicates} duplicate timestamps")
        
        # Check for chronological order
        is_sorted = df.select(pl.col("DateTime")).is_sorted()
        if not is_sorted:
            print("Warning: Data is not in chronological order")
        
        return True
        
    except Exception as e:
        print(f"Error during data validation: {e}")
        return False

def clean_duplicate_timestamps(df: pl.DataFrame, method: str = "mean") -> pl.DataFrame:
    """
    Remove duplicate timestamps by aggregating duplicate records.
    
    Parameters:
    -----------
    df : pl.DataFrame
        Input dataframe with potential duplicate timestamps
    method : str
        Aggregation method: "mean", "first", "last", "median"
    
    Returns:
    --------
    pl.DataFrame : Cleaned dataframe without duplicate timestamps
    """
    try:
        # Check if duplicates exist
        duplicates = df.select(pl.col("DateTime")).is_duplicated().sum()
        
        if duplicates == 0:
            print("No duplicate timestamps found.")
            return df
        
        print(f"Removing {duplicates} duplicate timestamps using '{method}' method...")
        
        # Get numeric and non-numeric columns
        numeric_cols = [col for col in df.columns if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32] and col != "DateTime"]
        non_numeric_cols = [col for col in df.columns if col not in numeric_cols and col != "DateTime"]
        
        # Define aggregation expressions
        agg_exprs = []
        
        # Aggregate numeric columns
        for col in numeric_cols:
            if method == "mean":
                agg_exprs.append(pl.col(col).mean().alias(col))
            elif method == "median":
                agg_exprs.append(pl.col(col).median().alias(col))
            elif method == "first":
                agg_exprs.append(pl.col(col).first().alias(col))
            elif method == "last":
                agg_exprs.append(pl.col(col).last().alias(col))
            else:
                agg_exprs.append(pl.col(col).mean().alias(col))  # Default to mean
        
        # Take first value for non-numeric columns
        for col in non_numeric_cols:
            agg_exprs.append(pl.col(col).first().alias(col))
        
        # Group by DateTime and aggregate
        cleaned_df = (df
                     .group_by("DateTime")
                     .agg(agg_exprs)
                     .sort("DateTime"))
        
        print(f"Data cleaned: {len(df)} → {len(cleaned_df)} records")
        return cleaned_df
        
    except Exception as e:
        print(f"Error cleaning duplicate timestamps: {e}")
        return df

def check_time_intervals(df: pl.DataFrame) -> dict:
    """
    Analyze time intervals in the dataset.
    
    Returns:
    --------
    dict : Statistics about time intervals
    """
    try:
        # Calculate time differences in seconds
        intervals_df = (df
                       .sort("DateTime")
                       .with_columns([
                           (pl.col("DateTime").diff().dt.total_seconds()).alias("interval_seconds")
                       ])
                       .filter(pl.col("interval_seconds").is_not_null()))
        
        if len(intervals_df) == 0:
            return {"error": "No valid intervals found"}
        
        intervals = intervals_df.select("interval_seconds").to_series().to_list()
        
        stats = {
            "min_seconds": min(intervals),
            "max_seconds": max(intervals),
            "median_seconds": sorted(intervals)[len(intervals)//2],
            "mean_seconds": sum(intervals) / len(intervals),
            "total_intervals": len(intervals)
        }
        
        # Check for irregularities
        median_interval = stats["median_seconds"]
        tolerance = median_interval * 0.1
        irregular = [abs(x - median_interval) > tolerance for x in intervals]
        stats["irregular_count"] = sum(irregular)
        stats["irregular_percentage"] = (stats["irregular_count"] / len(intervals)) * 100
        
        # Check for zero intervals (duplicates)
        zero_intervals = sum(1 for x in intervals if x == 0)
        stats["zero_intervals"] = zero_intervals
        
        return stats
        
    except Exception as e:
        print(f"Error analyzing time intervals: {e}")
        return {"error": str(e)}



def get_data_summary(df: pl.DataFrame) -> dict:
    """
    Generate summary statistics for the dataset.
    """
    try:
        summary = {
            "time_range": {
                "start": df.select(pl.col("DateTime").min()).item(),
                "end": df.select(pl.col("DateTime").max()).item()
            },
            "record_count": len(df),
            "wells": {
                col: {
                    "mean": df.select(pl.col(col).mean()).item(),
                    "min": df.select(pl.col(col).min()).item(),
                    "max": df.select(pl.col(col).max()).item()
                }
                for col in df.columns if col not in ["DateTime", "mm_Rain"]
            }
        }
        return summary
        
    except Exception as e:
        print(f"Error generating data summary: {e}")
        return None



