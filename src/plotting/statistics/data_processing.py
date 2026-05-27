"""
Data processing and transformation functions
"""

import polars as pl
from .constants import MONTH_ORDER


def prepare_data(data):
    """
    Prepare data by converting to Polars and adding Year, Month, Season columns.
    
    Parameters:
    -----------
    data : polars.DataFrame or pandas.DataFrame
        Input data with DateTime column
        
    Returns:
    --------
    polars.DataFrame : Prepared data with additional columns
    """
    # Ensure the data is in Polars format
    if not isinstance(data, pl.DataFrame):
        data = pl.from_pandas(data)
    
    # Convert DateTime and create month column with proper ordering
    data = data.with_columns([
        pl.col("DateTime").cast(pl.Date).alias("DateTime"),
        pl.col("DateTime").dt.year().alias("Year"),
        pl.col("DateTime").dt.strftime("%b").alias("Month"),
        pl.col("DateTime").dt.month().alias("month_num"),
    ])
    
    # Add Season column
    data = data.with_columns([
        pl.when(pl.col("month_num").is_in([12, 1, 2])).then(pl.lit("Summer"))
         .when(pl.col("month_num").is_in([3, 4, 5])).then(pl.lit("Autumn"))
         .when(pl.col("month_num").is_in([6, 7, 8])).then(pl.lit("Winter"))
         .when(pl.col("month_num").is_in([9, 10, 11])).then(pl.lit("Spring"))
         .otherwise(None).alias("Season")
    ])
    
    return data


def prepare_well_data(data, well_column):
    """
    Prepare data for a specific well column.
    
    Parameters:
    -----------
    data : polars.DataFrame
        Full dataset
    well_column : str
        Name of well column
        
    Returns:
    --------
    polars.DataFrame : Filtered and prepared well data
    """
    # Get well data and filter nulls
    well_data = data.filter(pl.col(well_column).is_not_null())
    well_data = well_data.with_columns([
        pl.col(well_column).cast(pl.Float64, strict=False)
    ])
    
    return well_data


def prepare_all_years_data(data, well_column):
    """
    Prepare data for all years visualization.
    
    Parameters:
    -----------
    data : polars.DataFrame
        Full dataset
    well_column : str
        Name of well column
        
    Returns:
    --------
    polars.DataFrame : Prepared data for all years
    """
    # Create all_years_data
    all_years_data = pl.DataFrame({
        "DateTime": data["DateTime"],
        well_column: data[well_column].cast(pl.Float64, strict=False)
    })
    
    # Add month name column
    all_years_data = all_years_data.with_columns([
        pl.col("DateTime").dt.strftime("%b").alias("Month")
    ])
    
    # Add month_num and Season columns
    all_years_data = all_years_data.with_columns([
        pl.col("DateTime").dt.month().alias("month_num"),
    ])
    all_years_data = all_years_data.with_columns([
        pl.when(pl.col("month_num").is_in([12, 1, 2])).then(pl.lit("Summer"))
         .when(pl.col("month_num").is_in([3, 4, 5])).then(pl.lit("Autumn"))
         .when(pl.col("month_num").is_in([6, 7, 8])).then(pl.lit("Winter"))
         .when(pl.col("month_num").is_in([9, 10, 11])).then(pl.lit("Spring"))
         .otherwise(None).alias("Season")
    ])
    
    return all_years_data

