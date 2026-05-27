"""
Script to extract the main callback from the backup file and write it to callbacks.py
with proper modular imports.
"""

import ast
import re

# Read the backup file
backup_file = r'c:\Users\oluwunmi\Downloads\groundwater_module\src\plotting\Extreme_Value_Analysis_POT.py'
with open(backup_file, 'r', encoding='utf-8') as f:
    backup_code = f.read()

# Parse to find the callback function
tree = ast.parse(backup_code)

# Find the update_eva_analysis function
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == 'update_eva_analysis':
        # Get the function source code
        func_start = node.lineno - 1  # Convert to 0-based index
        func_end = node.end_lineno
        
        # Extract function code
        lines = backup_code.split('\n')
        func_code = '\n'.join(lines[func_start:func_end])
        
        # Replace function calls with imports from modular structure
        # calculate_threshold -> from .helpers import calculate_threshold
        # find_peaks_data -> from ..data_processing import find_peaks_data
        # calculate_exceedances -> from ..data_processing import calculate_exceedances
        # etc.
        
        # Write to a temporary file for inspection
        with open('extracted_callback_temp.py', 'w', encoding='utf-8') as f:
            f.write(func_code)
        
        print(f"Extracted callback function: lines {func_start+1} to {func_end}")
        print(f"Function code length: {len(func_code)} characters")
        break

