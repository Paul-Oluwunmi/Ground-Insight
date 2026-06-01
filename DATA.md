# Data files

Large CSV and Excel datasets are **not** included in the GitHub repository (see `.gitignore`). After cloning, add your own data locally before running `Gwldd.ipynb`.

The folder names `large_dataset` and `folder_mode_data` are **examples** named after their loading mode. You can rename folders or use your own paths; CSV filenames inside each folder can also be anything you choose.

---

## How the example folders are used

| Folder | Loading mode in `Gwldd.ipynb` | What Ground Insight expects |
|--------|-------------------------------|-----------------------------|
| **`large_dataset/`** | `use_large_dataset_mode = True` | One or more large **long-format** CSVs (any filenames). Many wells per file. |
| **`folder_mode_data/`** | `use_folder_mode = True`, `csv_folder_path` → this folder | Many small CSVs (often one monitoring site per file). Format is auto-detected per file. |

You can keep several folder-mode directories (e.g. `folder_mode_data_2`) for different datasets — point `csv_folder_path` at whichever one you want to load.

Rainfall and mapping files for folder mode usually sit in **`data_folder`** (project root), not inside the data subfolder — see [Folder mode](#folder-mode).

---

## Large dataset mode (`large_dataset/`)

**Purpose:** A folder of large **long-format** CSVs: many sites and timestamps spread across one or more files, with optional quality flags. Filenames do not matter — Ground Insight loads every `.csv` in the folder.

**Enable in the notebook:**

```python
use_large_dataset_mode = True
large_dataset_path = str(PROJECT_ROOT / "large_dataset")
```

**Directory layout** — flat folder, any `.csv` names:

```
large_dataset/
├── period_1.csv
├── period_2.csv
└── regional_export.csv
```

Use any filenames you like (e.g. `period_1.csv`, `regional_export.csv`). At least one valid CSV must load successfully.

**Typical columns** (names are normalised on load; variants are accepted):

| Role | Accepted column names (examples) |
|------|----------------------------------|
| Well / site ID | `SiteKey`, `site_key`, `Site`, `Well`, `Well_ID` |
| Timestamp | `ReadDateTime`, `ReadDate`, `DateTime`, `Date`, `Time` |
| Groundwater level | `WaterLevel`, `water_level`, `Level`, `Value` |
| Data quality (optional) | `Source` — values `A`, `N`, `F` map to Best / Second / Unsure |

**Example header** (long format):

```text
SiteKey,ReadDateTime,WaterLevel,Source,EastingNZTM,NorthingNZTM,Depth,Comment,...
BX23/1573,25/08/2025 13:15,-8.363,N,1562841,5178926,14.88,...
```

Extra metadata columns (easting, screen depths, etc.) are kept in the raw load but the analysis pivot uses `DateTime`, `SiteKey`, and `WaterLevel`.

**Output:** The loader builds wide tables per quality tier (`Best`, `Second`, `Unsure`) plus a combined view. Well names come from `SiteKey` (slashes and spaces converted to underscores for column names).

---

## Folder mode

**Purpose:** A directory of many CSV files — common when each file is one site or one export from a regional database.

**Enable in the notebook:**

```python
use_folder_mode = True
csv_folder_path = str(PROJECT_ROOT / "folder_mode_data")   # or any folder you choose
use_large_dataset_mode = False
```

**Directory layout** — flat list of `.csv` files (no required subfolders):

```
folder_mode_data/
├── site_63_12.csv
├── site_64_01.csv
└── ...   (one file per site is typical; 500+ files is fine)
```

Every `.csv` in the folder is read and merged on `DateTime`. Optional rainfall (`rainfall_filename`) and mapping (`mapping_filename`) are loaded from **`data_folder`** (usually the project root), not from inside `csv_folder_path`.

### Supported CSV shapes (auto-detected per file)

**1. Indexed** — multiple sites in one file:

```text
index,SiteName,MeasurementName,Time,Value
0,Well...222 Comminutor Stn,Elevation Above Sea Level,9/12/1997,9.783
```

Required concepts: `SiteName`, `Time`, `Value` (and related index columns as in your export).

**2. Long** — one or more sites per file:

```text
DateTime,name,level
```
or two columns only (site name taken from the filename):

```text
timestamp,value
2/05/1985,93.4857875
```

**3. Wide** — one row per time, one column per well:

```text
DateTime,Well_A,Well_B,Well_C
2019-03-06 00:00,0.609,0.3,0.395
```

The first row may be well names with dates in the first column (as in `DunTestData.csv`); format detection treats that as wide format.

After loading, all folder-mode data are combined into one dataframe: index `DateTime`, one column per well, plus `mm_Rain` (filled from a separate rainfall CSV if provided).

---

## Single-file mode (project root)

**Purpose:** One groundwater CSV (and optionally one rainfall CSV) in `data_folder`, without data subfolders.

```python
use_folder_mode = False
use_large_dataset_mode = False
groundwater_filename = "DunTestData.csv"
rainfall_filename = "DunRain.csv"
data_folder = str(PROJECT_ROOT)
```

**Example wide groundwater file** (`DunTestData.csv`):

```text
Dun_58,I44_0005,I44_0006,I44_0007,I44_1094
6/03/2019 00:00,0.609,0.3,0.395,-0.184
```

First row = well IDs; first column = timestamps.

**Example rainfall file** (`DunRain.csv`): may include title rows, then:

```text
6/03/2019 00:00,0
6/03/2019 00:10,0
```

Merged as `DateTime` + `mm_Rain`.

---

## Quick reference

| Mode | Flag | Path variable | Typical layout |
|------|------|---------------|----------------|
| Single file | defaults | `data_folder` | 1–2 CSVs at project root |
| Folder | `use_folder_mode=True` | `csv_folder_path` | Many CSVs, mixed formats OK |
| Large dataset | `use_large_dataset_mode=True` | `large_dataset_path` | Any `.csv` files, long format |

Update paths in the **PATHS AND SETTINGS** cell in `Gwldd.ipynb`. For citation and contact, see `README.md` and `CITATION.cff`.
