"""
Main execution module for groundwater analysis tool.
Contains the main_unified function and visualization runners.
"""

from src.data_loading import (
    load_indexed_format,
    load_and_process_data,
    load_multiple_csv_enhanced,
    load_canterbury_large_datasets,
    detect_data_format,
    clean_filename,
    convert_units,
    remove_duplicates
)


def main_unified(use_folder_mode, csv_folder_path, rainfall_filename, mapping_filename,
                 use_large_dataset_mode, canterbury_data_path,
                 groundwater_filename, data_folder, CONVERT_MM_TO_METERS, REMOVE_DUPLICATES):
    """
    Universal main execution function that handles all scenarios.
    
    Parameters:
    -----------
    use_folder_mode : bool
        Whether to use folder mode for loading multiple CSV files
    csv_folder_path : str
        Path to folder containing CSV files (if use_folder_mode is True)
    rainfall_filename : str
        Name of rainfall CSV file
    mapping_filename : str
        Name of mapping CSV file
    use_large_dataset_mode : bool
        Whether to use large dataset mode (multi-file long format)
    canterbury_data_path : str
        Path to folder of large CSV files (if use_large_dataset_mode is True)
    groundwater_filename : str
        Name of groundwater CSV file (if single file mode)
    data_folder : str
        Path to folder containing data files
    CONVERT_MM_TO_METERS : bool
        Whether to convert units from mm to meters
    REMOVE_DUPLICATES : bool
        Whether to remove duplicate records
        
    Returns:
    --------
    tuple : (data, well_columns, mapping_dict, quality_dfs)
        Loaded data, well columns, mapping dictionary, and quality dataframes
    """
    try:
        print("\n" + "="*60)
        print("GROUNDWATER ANALYSIS TOOL - UNIFIED MODE")
        print("="*60)
        
        if use_folder_mode:
            print(f"\n FOLDER MODE: Loading data from {csv_folder_path}")
            if rainfall_filename and rainfall_filename != 'None':
                print(f" Loading rainfall data from: {rainfall_filename}")
            else:
                print(" No rainfall data will be loaded")
            data, well_columns, mapping_dict = load_multiple_csv_enhanced(
                csv_folder_path,
                data_folder=data_folder,
                rainfall_csv_name=rainfall_filename, 
                mapping_csv_name=mapping_filename,
                convert_mm_to_m=CONVERT_MM_TO_METERS, 
                remove_duplicates_flag=REMOVE_DUPLICATES
            )
            quality_dfs = None
        elif use_large_dataset_mode:
            print("\n LARGE DATASET MODE: Processing multi-file long-format data...")
            data, well_columns, quality_dfs = load_canterbury_large_datasets(canterbury_data_path)
            mapping_dict = None
        else:
            print(f"\n SINGLE FILE MODE: Loading data from {groundwater_filename}")
            if rainfall_filename and rainfall_filename != 'None':
                print(f"  Loading rainfall data from: {rainfall_filename}")
            else:
                print(" No rainfall data will be loaded")
            data, well_columns, mapping_dict = load_and_process_data(
                groundwater_filename,
                data_folder=data_folder,
                rainfall_filename=rainfall_filename, 
                mapping_filename=mapping_filename,
                convert_mm_to_m=CONVERT_MM_TO_METERS, 
                remove_duplicates_flag=REMOVE_DUPLICATES
            )
            quality_dfs = None
        
        if data is not None and well_columns:
            # Check if we have quality-categorised data available
            if quality_dfs and len(quality_dfs) > 0:
                print("\n" + "="*60)
                print("QUALITY-CATEGORISED DATA AVAILABLE")
                print("="*60)
                print("We have three quality-categorised datasets:")
                print("• Best Quality (A): Most reliable trends, highest confidence")
                print("• Second Best (N): Good trends, moderate confidence")
                print("• Unsure (F): Trends with lower confidence, may need verification")
                print("\nWould you like to use quality-categorised data for analysis?")
                
                while True:
                    quality_choice = input("Enter 'y' to use quality-categorised data, 'n' to use combined data: ").strip().lower()
                    if quality_choice in ['y', 'n']:
                        break
                    print("Please enter 'y' or 'n'.")
                
                if quality_choice == 'y':
                    print("\n" + "="*60)
                    print("SELECT QUALITY GROUP FOR ANALYSIS")
                    print("="*60)
                    print("Which quality group would you like to analyze?")
                    print("1. Best Quality (A) - Most reliable trends, highest confidence")
                    print("2. Second Best (N) - Good trends, moderate confidence")
                    print("3. Unsure (F) - Trends with lower confidence, may need verification")
                    print("4. Back to combined data choice")
                    
                    while True:
                        quality_group_choice = input("Enter your choice (1-4): ").strip()
                        if quality_group_choice in ['1', '2', '3', '4']:
                            break
                        print("Please enter a number between 1 and 4.")
                    
                    if quality_group_choice == '1':
                        print("\nUsing Best Quality (A) data for analysis.")
                        data = quality_dfs['Best']
                        well_columns = [col for col in data.columns if col != 'mm_Rain']
                    elif quality_group_choice == '2':
                        print("\nUsing Second Best (N) data for analysis.")
                        data = quality_dfs['Second']
                        well_columns = [col for col in data.columns if col != 'mm_Rain']
                    elif quality_group_choice == '3':
                        print("\nUsing Unsure (F) data for analysis.")
                        data = quality_dfs['Unsure']
                        well_columns = [col for col in data.columns if col != 'mm_Rain']
                    else:  # choice == '4'
                        print("\nUsing combined data (all qualities) for analysis.")
                        # data and well_columns remain as the combined dataset
                else:
                    print("\nUsing combined data (all qualities) for analysis.")
            
            print("\n" + "="*60)
            print("Select visualisation type:")
            print("1. Static Interactive Plot")
            print("2. Julian Plot")
            print("3. Subplots")
            print("4. Overlay")
            print("5. Monthly Statistics")
            print("6. Statistical Analysis")
            print("7. Extreme Value Analysis MAXIMA")
            print("8. Extreme Value Analysis POT")
            print("9. Wavelet Analysis")
            print("10. Batch Mann-Kendall Analysis")
            print("11. All")
            print("12. Exit")
            print("="*60)
            while True:
                choice = input("\nEnter your choice (1-12): ")
                if choice in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']:
                    break
                print("Invalid input. Please enter a number between 1 and 12.")
            if choice == '12':
                print("\nExiting programme...")
                return None, None, None, None
            run_visualisation(data, well_columns, choice, mapping_dict)
            return data, well_columns, mapping_dict, quality_dfs
        else:
            print("\n Failed to load data. Please check your file names and data format.")
            return None, None, None, None
    except Exception as e:
        print(f" Error in main_unified: {str(e)}")
        return None, None, None, None


def run_visualisation(data, well_columns, choice, mapping_dict=None):
    """Run selected visualisation"""
    try:
        if choice == '1':
            print("\nCreating static interactive plot...")
            from src.plotting.static_plots import plot_interactive_hydrograph
            fig = plot_interactive_hydrograph(data, well_columns)
            if fig is not None:
                fig.show()
       
        elif choice == '2':
            print("\nCreating Julian plot...")
            from src.plotting.julian_plot import run_julian_plot
            run_julian_plot(data, well_columns)
               
        elif choice == '3':
            print("\nLaunching subplot application...")
            from src.plotting.subplot import run_subplot
            run_subplot(data, well_columns)
       
        elif choice == '4':
            print("\nLaunching overlay application...")
            from src.plotting.overlay import run_overlay
            run_overlay(data, well_columns)
           
        elif choice == '5':
            print("\nLaunching Monthly Statistics...")
            from src.plotting.statistics import run_statistics
            run_statistics(data, well_columns)
                             
        elif choice == '6':
            print("\nLaunching Statistical Analysis...")
            from src.plotting.statistical_tests import run_statistical_analysis
            run_statistical_analysis(data, well_columns)

        elif choice == '7':
            print("\nLaunching Extreme Value Analysis...")
            from src.plotting.Extreme_Value_Analysis import run_eva_analysis
            run_eva_analysis(data, well_columns)

        elif choice == '8':
            print("\nLaunching POT Analysis...")
            from src.plotting.Extreme_Value_Analysis_POT import run_pot_dashboard
            run_pot_dashboard(data, well_columns)

        elif choice == '9':
            print("\nLaunching Wavelet Analysis...")
            from src.plotting.wavelet_analysis import run_wavelet_analysis
            run_wavelet_analysis(data, well_columns)

        elif choice == '10':
            print("\nLaunching Batch Mann-Kendall Trend Analysis...")
            # Import directly from the file to avoid triggering src.plotting.__init__.py
            # which has circular imports with Extreme_Value_Analysis
            import sys
            import os
            from importlib.util import spec_from_file_location, module_from_spec
            from types import ModuleType
            
            # Ensure src is in sys.path for absolute imports to work
            src_dir = os.path.dirname(__file__)
            parent_dir = os.path.dirname(src_dir)
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
            
            # CRITICAL: Create mock src.plotting package BEFORE loading Batch_Mann_Kendall.py
            # This prevents Python from importing the real src.plotting.__init__.py
            if 'src.plotting' not in sys.modules:
                plotting_pkg = ModuleType('src.plotting')
                plotting_pkg.__path__ = [os.path.join(src_dir, 'plotting')]
                plotting_pkg.__name__ = 'src.plotting'
                plotting_pkg.__package__ = 'src.plotting'
                sys.modules['src.plotting'] = plotting_pkg
            
            # Get the path to Batch_Mann_Kendall.py
            batch_mk_path = os.path.join(src_dir, 'plotting', 'Batch_Mann_Kendall.py')
            
            # Load the module directly with proper package context
            spec = spec_from_file_location('src.plotting.Batch_Mann_Kendall', batch_mk_path)
            batch_mk_module = module_from_spec(spec)
            batch_mk_module.__package__ = 'src.plotting'
            batch_mk_module.__name__ = 'src.plotting.Batch_Mann_Kendall'
            spec.loader.exec_module(batch_mk_module)
            
            # Extract the functions
            run_yearly_water_level_statistics = batch_mk_module.run_yearly_water_level_statistics
            run_hydrological_year_analysis_both_types = batch_mk_module.run_hydrological_year_analysis_both_types
            run_comprehensive_original_mk_batch_analysis_separate_files = batch_mk_module.run_comprehensive_original_mk_batch_analysis_separate_files
            run_comprehensive_seasonal_mk_batch_analysis_separate_files = batch_mk_module.run_comprehensive_seasonal_mk_batch_analysis_separate_files
            
            # Run the batch analysis
            print("\n" + "=" * 80)
            print("BATCH MANN-KENDALL ANALYSIS")
            print("=" * 80)
            print("\nThis analysis will run the following analyses for all wells:")
            print("  1. Yearly Water Level Statistics")
            print("  2. Hydrological Year Analysis (Raw + Rolling Mean)")
            print("  3. Original Mann-Kendall Trend Analysis")
            print("  4. Seasonal Mann-Kendall Trend Analysis")
            print("\n" + "=" * 80 + "\n")
            
            run_yearly_water_level_statistics(data, well_columns)
            print("\n" + "-" * 80 + "\n")
            
            run_hydrological_year_analysis_both_types(data, well_columns)
            print("\n" + "-" * 80 + "\n")
            
            run_comprehensive_original_mk_batch_analysis_separate_files(data, well_columns)
            print("\n" + "-" * 80 + "\n")
            
            run_comprehensive_seasonal_mk_batch_analysis_separate_files(data, well_columns)
            print("\n" + "=" * 80)
            print("BATCH MANN-KENDALL ANALYSIS COMPLETED")
            print("=" * 80)
            
            # Create Batch_Mann_Kendall package entry BEFORE importing modules
            batch_mk_package = sys.modules.get('src.plotting.mann_kendall.Batch_Mann_Kendall')
            if batch_mk_package is None:
                class BatchMockPackage:
                    __path__ = [os.path.join(mk_package_path, 'Batch_Mann_Kendall')]
                    __name__ = 'src.plotting.mann_kendall.Batch_Mann_Kendall'
                    __package__ = 'src.plotting.mann_kendall.Batch_Mann_Kendall'
                batch_mk_package = BatchMockPackage()
                sys.modules['src.plotting.mann_kendall.Batch_Mann_Kendall'] = batch_mk_package
            
            # Now import Batch_Mann_Kendall modules - their absolute imports should work
            # and the relative imports in the modules they import should also work
            try:
                original_mk = importlib.import_module('src.plotting.mann_kendall.Batch_Mann_Kendall.original_mk')
                seasonal_mk = importlib.import_module('src.plotting.mann_kendall.Batch_Mann_Kendall.seasonal_mk')
                hydrological_year = importlib.import_module('src.plotting.mann_kendall.Batch_Mann_Kendall.hydrological_year')
                yearly_stats = importlib.import_module('src.plotting.mann_kendall.Batch_Mann_Kendall.yearly_statistics')
            except Exception as import_error:
                print(f"Error importing Batch_Mann_Kendall modules: {import_error}")
                import traceback
                traceback.print_exc()
                raise
            
            run_comprehensive_original_mk_batch_analysis_separate_files = original_mk.run_comprehensive_original_mk_batch_analysis_separate_files
            run_comprehensive_seasonal_mk_batch_analysis_separate_files = seasonal_mk.run_comprehensive_seasonal_mk_batch_analysis_separate_files
            run_hydrological_year_analysis_both_types = hydrological_year.run_hydrological_year_analysis_both_types
            run_yearly_water_level_statistics = yearly_stats.run_yearly_water_level_statistics
            
            # Run all batch Mann-Kendall analyses
            print("\n" + "="*80)
            print("RUNNING COMPREHENSIVE BATCH MANN-KENDALL ANALYSIS")
            print("="*80)
            
            # 1. Original MK Analysis
            print("\n[1/4] Running Original Mann-Kendall Analysis...")
            run_comprehensive_original_mk_batch_analysis_separate_files(
                data, well_columns, mapping_dict=mapping_dict
            )
            
            # 2. Seasonal MK Analysis
            print("\n[2/4] Running Seasonal Mann-Kendall Analysis...")
            run_comprehensive_seasonal_mk_batch_analysis_separate_files(
                data, well_columns, mapping_dict=mapping_dict
            )
            
            # 3. Hydrological Year Analysis
            print("\n[3/4] Running Hydrological Year Analysis...")
            run_hydrological_year_analysis_both_types(
                data, well_columns, mapping_dict=mapping_dict
            )
            
            # 4. Yearly Statistics
            print("\n[4/4] Running Yearly Water Level Statistics...")
            run_yearly_water_level_statistics(
                data, well_columns, mapping_dict=mapping_dict
            )
            
            print("\n" + "="*80)
            print("BATCH MANN-KENDALL ANALYSIS COMPLETED")
            print("="*80)

        elif choice == '11':
             print("\nRunning all visualisations...")
             run_all_visualisations(data, well_columns, mapping_dict)
           
        else:
            print("\nInvalid choice. Please enter 1-12.")
           
    except Exception as e:
        print(f"Error in visualisation: {str(e)}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()


def run_all_visualisations(data, well_columns, mapping_dict=None):
    """Run all visualisation types"""
    try:
        # Import all visualisation functions
        from src.plotting.static_plots import plot_interactive_hydrograph
        from src.plotting.julian_plot import run_julian_plot
        from src.plotting.subplot import run_subplot
        from src.plotting.overlay import run_overlay
        from src.plotting.statistical_tests import run_statistical_analysis
        from src.plotting.statistics import run_statistics
        from src.plotting.Extreme_Value_Analysis import run_eva_analysis
        from src.plotting.Extreme_Value_Analysis_POT import run_pot_dashboard
        from src.plotting.wavelet_analysis import run_wavelet_analysis
        # Simple import, just like EVA
        from src.plotting.Batch_Mann_Kendall import (
            run_yearly_water_level_statistics,
            run_hydrological_year_analysis_both_types,
            run_comprehensive_original_mk_batch_analysis_separate_files,
            run_comprehensive_seasonal_mk_batch_analysis_separate_files
        )

        print("\nCreating static interactive plot...")
        fig = plot_interactive_hydrograph(data, well_columns)
        if fig is not None:
            fig.show()
           
        print("\nCreating Julian plot...")
        run_julian_plot(data, well_columns)
       
        print("\nLaunching subplot application...")
        run_subplot(data, well_columns)
       
        print("\nLaunching overlay application...")
        run_overlay(data, well_columns)
       
        print("\nLaunching Monthly Statistics...")
        run_statistics(data, well_columns)
       
        print("\nLaunching Statistical Analysis...")
        run_statistical_analysis(data, well_columns)
       
        print("\nLaunching Extreme Value Analysis MAXIMA...")
        run_eva_analysis(data, well_columns)
       
        print("\nLaunching Extreme Value Analysis POT...")
        run_pot_dashboard(data, well_columns)

        print("\nLaunching Wavelet Analysis...")  
        run_wavelet_analysis(data, well_columns)
       
        print("\nLaunching Batch Mann-Kendall Trend Analysis...")
        print("\n[1/4] Running Original Mann-Kendall Analysis...")
        run_comprehensive_original_mk_batch_analysis_separate_files(
            data, well_columns, mapping_dict=mapping_dict
        )
        print("\n[2/4] Running Seasonal Mann-Kendall Analysis...")
        run_comprehensive_seasonal_mk_batch_analysis_separate_files(
            data, well_columns, mapping_dict=mapping_dict
        )
        print("\n[3/4] Running Hydrological Year Analysis...")
        run_hydrological_year_analysis_both_types(
            data, well_columns, mapping_dict=mapping_dict
        )
        print("\n[4/4] Running Yearly Water Level Statistics...")
        run_yearly_water_level_statistics(
            data, well_columns, mapping_dict=mapping_dict
        )

    except Exception as e:
        print(f"Error in all visualisations: {str(e)}")

