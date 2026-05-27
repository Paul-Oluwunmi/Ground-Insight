# Ground Insight

A Python toolkit for groundwater level analysis, rainfall relationships, and interactive visualisation. This folder is the active working copy of the project (last updated April 2026).

## Overview

Ground Insight provides a modular framework for:

- **Data loading** — indexed, long, and wide CSV formats; single file, multi-file folder, and large Canterbury datasets
- **Trend analysis** — Mann-Kendall (original and seasonal), hydrological year statistics, time evolution
- **Extreme value analysis** — Block Maxima (GEV) and Peaks Over Threshold (GPD), including Bayesian inference
- **Wavelet analysis** — CWT, coherence, cross-wavelet, and partial coherence
- **Interactive dashboards** — Dash and Plotly visualisations

## Requirements

- Python 3.8+ (tested with 3.10)
- Dependencies listed in `requirements.txt`

```bash
pip install -r requirements.txt
```

Or from Python or the notebook:

```python
from src.install_packages import install_requirements
install_requirements()
```

## Quick start

### 1. Open the notebook

Start Jupyter in this folder and open `Gwldd.ipynb`:

```bash
cd C:\Users\oluwunmi\Desktop\Ground_Insight
jupyter notebook Gwldd.ipynb
```

### 2. Set your paths

In the configuration cell, point `module_path` and `data_folder` at this project directory (or wherever your CSVs are stored). Example:

```python
module_path = r"C:\Users\oluwunmi\Desktop\Ground_Insight"
data_folder = r"C:\Users\oluwunmi\Desktop\Ground_Insight"

groundwater_filename = "DunTestData.csv"
rainfall_filename = "DunRain.csv"
mapping_filename = "None"

# Optional modes
use_folder_mode = False
csv_folder_path = r"C:\Users\oluwunmi\Desktop\Ground_Insight\Waikato_data"
use_large_dataset_mode = False
canterbury_data_path = r"C:\Users\oluwunmi\Desktop\Ground_Insight\Canterbury_data"
```

Sample data files are included at the project root (`DunTestData.csv`, `DunRain.csv`, `TaurTestData.csv`, and so on). Regional datasets are in `Canterbury_data`, `Hawkesbay_data`, and `Waikato_data`.

### 3. Run the analysis

Run the notebook cells in order. The main entry point loads data and prompts you to choose a visualisation:

```python
from src.main import main_unified

data, well_columns, mapping_dict, quality_dfs = main_unified(
    use_folder_mode=use_folder_mode,
    csv_folder_path=csv_folder_path,
    rainfall_filename=rainfall_filename,
    mapping_filename=mapping_filename,
    use_large_dataset_mode=use_large_dataset_mode,
    canterbury_data_path=canterbury_data_path,
    groundwater_filename=groundwater_filename,
    data_folder=data_folder,
    CONVERT_MM_TO_METERS=False,
    REMOVE_DUPLICATES=False,
)
```

Once the data have loaded, choose from the interactive menu (static plots, Julian plot, EVA, wavelets, batch Mann-Kendall, and more).

## Project layout

```
Ground_Insight/
├── src/
│   ├── data_loading/              # Loaders, format detection, unit conversion
│   ├── plotting/
│   │   ├── mann_kendall/          # Trend analysis and batch exports
│   │   ├── extreme_value_analysis/    # Block Maxima (GEV)
│   │   ├── extreme_value_analysis_pot/  # Peaks Over Threshold (GPD)
│   │   ├── wavelet_analysis/      # Wavelet dashboards
│   │   ├── statistics/            # Monthly statistics
│   │   ├── statistical_tests/
│   │   ├── julian_plot/
│   │   ├── overlay/
│   │   ├── static_plots/
│   │   ├── subplot/
│   │   └── Batch_Mann_Kendall.py
│   ├── main.py                    # main_unified() and visualisation menu
│   └── install_packages.py
├── Canterbury_data/               # Large regional datasets
├── Hawkesbay_data/
├── Waikato_data/
├── Gwldd.ipynb                    # Main workflow notebook
├── requirements.txt
└── README.md
```

## Data loading modes

| Mode | When to use | Key inputs |
|------|-------------|------------|
| **Single file** | One groundwater CSV (+ optional rainfall) | `groundwater_filename`, `data_folder` |
| **Folder** | Many well CSVs in one directory | `use_folder_mode=True`, `csv_folder_path` |
| **Large dataset** | Canterbury GNS-style multi-file layout | `use_large_dataset_mode=True`, `canterbury_data_path` |

Groundwater CSVs should include a `DateTime` column. Rainfall files use `DateTime` and `mm_Rain`.

## Analysis modules

### Mann-Kendall (`src/plotting/mann_kendall/`)

Batch trend analysis, seasonal MK, hydrological year statistics, and exports. User settings are in `src/plotting/mann_kendall/config.txt`:

```
USE_ROLLING_MEAN = True
ROLLING_WINDOW_SIZE = 7
ALPHA_LEVEL = 0.05
RESAMPLE_FREQUENCY = D
```

### Extreme value analysis

```python
from src.plotting.extreme_value_analysis import run_eva_analysis       # Block Maxima (GEV)
from src.plotting.extreme_value_analysis_pot import run_pot_dashboard  # Peaks Over Threshold (GPD)
```

### Wavelet analysis

```python
from src.plotting.wavelet_analysis import run_wavelet_analysis

run_wavelet_analysis(
    data=data,
    well_columns=well_columns,
    rainfall_col="mm_Rain",
)
```

### Batch Mann-Kendall

```python
from src.plotting.Batch_Mann_Kendall import (
    run_yearly_water_level_statistics,
    run_hydrological_year_analysis_both_types,
    run_comprehensive_original_mk_batch_analysis_separate_files,
    run_comprehensive_seasonal_mk_batch_analysis_separate_files,
)
```

## Visualisation menu

When you run `main_unified()`, you can select:

1. Static interactive plot  
2. Julian plot  
3. Subplots  
4. Overlay  
5. Monthly statistics  
6. Statistical analysis  
7. Extreme value analysis (Block Maxima)  
8. Extreme value analysis (POT)  
9. Wavelet analysis  
10. Batch Mann-Kendall  
11. All  
12. Exit  

## Data on GitHub

CSV and regional data folders are excluded from the repository (see `DATA.md`). Clone the repo, then add your own data locally before running the notebook.

## Publishing to GitHub

From this folder:

```bash
git init
git add .
git commit -m "Initial commit: Ground Insight groundwater analysis toolkit"
```

Repository: **https://github.com/Paul-Oluwunmi/Ground-Insight**

```bash
git branch -M main
git remote add origin https://github.com/Paul-Oluwunmi/Ground-Insight.git
git push -u origin main
```

## Acknowledgements

Built with Dash, Plotly, PyWavelets, emcee, pandas, polars, SciPy, statsmodels, and pymannkendall.
