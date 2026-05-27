================================================================================
MANN-KENDALL ANALYSIS CONFIGURATION FILE
================================================================================

HOW TO USE:
-----------
1. Open config.txt in any text editor (Notepad, VS Code, etc.)
2. Edit the values you want to change
3. Save the file
4. Run your analysis - the new settings will be used automatically

IMPORTANT:
----------
- DO NOT edit constants.py or any other Python files
- ONLY edit config.txt to change settings
- This prevents accidentally breaking the code

CONFIGURATION OPTIONS:
----------------------

ROLLING MEAN SETTINGS:
- USE_ROLLING_MEAN: True or False
  Set to True to enable rolling mean smoothing
  
- ROLLING_WINDOW_SIZE: Number (e.g., 7)
  Rolling window size in days (for daily data)

ANALYSIS SETTINGS:
- ALPHA_LEVEL: Number (e.g., 0.05)
  Significance level for statistical tests (0.05 = 95% confidence)
  
- RESAMPLE_FREQUENCY: D, W, or M
  D = Daily, W = Weekly, M = Monthly

SAVE LOCATION SETTINGS:
- CUSTOM_SAVE_LOCATION: None or a path (e.g., C:/MyResults)
  Set to None to save in current directory
  Set to a path like "C:/MyResults" to save there
  
- CREATE_FOLDERS: True or False
  Automatically create output folders if they don't exist
  
- SAVE_PLOTS: True or False
  Whether to save plot images

BATCH ANALYSIS SETTINGS:
- RUN_ORIGINAL_MK: True or False
- RUN_SEASONAL_MK: True or False
- RUN_TIME_EVOLUTION: True or False
- RUN_HYDROLOGICAL_YEAR: True or False
- RUN_DURATION_LOW_FLOW: True or False
- RUN_YEARLY_STATISTICS: True or False

TIME EVOLUTION SETTINGS:
- TIME_EVOLUTION_WINDOW_SIZE: Number (e.g., 1825)
  Window size in days (1825 = 5 years)
  
- TIME_EVOLUTION_STEP_SIZE: Number (e.g., 1825)
  Step size between windows in days
  
- TIME_EVOLUTION_OVERLAP: True or False
  Whether windows should overlap
  
- TIME_EVOLUTION_MIN_POINTS: Number (e.g., 1)
  Minimum data points required per window

TREND ANALYSIS SETTINGS:
- IGNORE_ZERO_VALUES_IN_TREND: True or False
  Set to True to exclude zero values from analysis

EXAMPLES:
---------
To change the rolling window size to 30 days:
  ROLLING_WINDOW_SIZE = 30

To change the significance level to 0.01 (99% confidence):
  ALPHA_LEVEL = 0.01

To save results to a custom folder:
  CUSTOM_SAVE_LOCATION = C:/MyAnalysisResults

To disable rolling mean:
  USE_ROLLING_MEAN = False

================================================================================

