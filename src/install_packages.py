#!/usr/bin/env python3
"""
Automatic Package Installation Script for Groundwater Analysis Tool
This script will install all required packages for the groundwater analysis tool.
"""

import subprocess
import sys
import importlib
import os

def install_package(package):
    """Install a package using pip if it's not already installed"""
    try:
        # Extract package name without version specifiers
        package_name = package.split('>=')[0].split('==')[0].split('<')[0].split('>')[0]
        importlib.import_module(package_name)
        print(f" {package_name} is already installed")
        return True
    except ImportError:
        print(f"Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"{package} installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f" Failed to install {package}: {e}")
            return False

def install_requirements():
    """Install all required packages from requirements.txt"""
    print("Groundwater Analysis Tool - Package Installation")
    print("="*60)
    print("Checking and installing required packages...")
    print("="*60)
    
    # List of required packages with versions
    required_packages = [
        "pandas>=1.5.0",
        "numpy>=1.21.0", 
        "polars>=0.20.0",
        "matplotlib>=3.5.0",
        "plotly>=5.0.0",
        "dash>=2.0.0",
        "scipy>=1.9.0",
        "scikit-learn>=1.1.0",
        "statsmodels>=0.13.0",
        "pymannkendall>=1.4.0",
        "PyWavelets>=1.4.0",
        "pyextremes>=2.0.0",
        "emcee>=3.1.0",
        "corner>=2.2.0",
        "openpyxl>=3.0.0",
        "ipython>=8.0.0",
        "jupyter>=1.0.0"
    ]
    
    # Check if requirements.txt exists and install from it
    # Look for requirements.txt in parent directory (where notebook is located)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    requirements_file = os.path.join(parent_dir, "requirements.txt")
    if os.path.exists(requirements_file):
        print(f"Found {requirements_file}, installing packages...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
            print("✓ All packages installed from requirements.txt")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error installing from requirements.txt: {e}")
            print("Installing packages individually...")
    else:
        print("requirements.txt not found, installing packages individually...")
    
    # Install packages individually
    failed_packages = []
    for package in required_packages:
        if not install_package(package):
            failed_packages.append(package)
    
    print("="*60)
    if failed_packages:
        print(" Some packages failed to install:")
        for package in failed_packages:
            print(f"   - {package}")
        print("\nYou may need to install these manually or check for conflicts.")
        return False
    else:
        print(" Package installation completed successfully!")
        print("You can now run the groundwater analysis tool.")
        return True

def main():
    """Main function"""
    try:
        success = install_requirements()
        if success:
            print("\n All packages are ready!")
            print("You can now run the groundwater analysis tool by opening Gwldd.ipynb")
        else:
            print("\n Some packages failed to install.")
            print("Please check the error messages above and install manually if needed.")
    except Exception as e:
        print(f"An error occurred during installation: {e}")
        return False
    
    return success

if __name__ == "__main__":
    main()
