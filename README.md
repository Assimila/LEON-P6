
# S1 Deforestation Tracker

A complete, reproducible workflow for:
1. Establishing a **forest baseline (2020)**
2. Detecting **deforestation from Sentinel‑1 SAR time series**
3. **Validating** against **Hansen** and **Tree Height (GEDI + Tessera)** products
4. Computing **forest change KPIs** (forest loss, annual rates, forest→crop conversion)

All spatial outputs use **EPSG:32636 (UTM Zone 36N)**.

---

## Repository Structure

```txt
.
├── environment.yml
├── 0_download_crops_gee.js
├── 0_download_lc_esri_gee.js
├── 0_download_natural_forest_gee.js
├── 1_forest_baseline.ipynb
├── 2_deforestation_tracker.ipynb
├── 3_validation.ipynb
└── 4_kpi.ipynb
```

---

## 1) Forest Baseline (2020)

**Notebook:** `1_forest_baseline.ipynb`

### Inputs
- **Natural forest mask**  
  Download using GEE script:  
  `0_download_natural_forest_gee.js`
- **ESRI Annual Land Cover (2020)**  
  Download using GEE script:  
  `0_download_lc_esri_gee.js`
- **CLMS / Copernicus Tree Density (100 m)**  
  Download from **Copernicus Dataspace**:  
  https://browser.dataspace.copernicus.eu/

### Purpose
Create a robust **Forest_2020** baseline combining structural (tree density), land cover, and natural forest datasets.

### Output
`forest_extent_2020_32636.tif`  
*(10 m, uint8, 1 = forest, CRS EPSG:32636)*

---

## 2) Sentinel‑1 Deforestation Tracker

**Notebook:** `2_deforestation_tracker.ipynb`

### Summary of Method
- Download **Sentinel‑1 VH polarisation time series** for the AOI,  
  **downsampled to 30 m** to reduce noise and stabilise the temporal signal.
- Apply **multi‑temporal and spatial speckle filtering** to the GRD stack.  
- Compute **temporal variability metrics**  
- **Logistic curve fitting** to detect breakpoints  
- Produce **continuous date of disturbance** (e.g., 2021.45, 2023.88)  
- Extract **integer deforestation year** (`floor(value)`)  
- Generate **deforestation mask** (2020–2024)

### Outputs
- `s1_deforestation_year_20-24.tif` *(float32, 30 m)*
- `s1_deforestation_mask_20-24.tif` *(uint8, 30 m)*

---

## 3) Validation

**Notebook:** `3_validation.ipynb`

### Validation Datasets

#### Hansen Global Forest Change
- Loss codes **21–24** = 2021–2024  
- Reprojected to EPSG:32636 for pixel‑wise comparison  

#### Tree Height Predictions (GEDI + Tessera)
Annual canopy height maps (50 m, COG, 2020–2024):  

#### Tree‑Height‑Based Deforestation Map
`deforestation_map_tess_2020_2024_6m.tif`  
- Deforestation if height drops from **≥ 6 m → < 6 m**  
- Values:  
  - **21** (2021 deforestation)  
  - **22** (2022)  
  - **23** (2023)  
  - **24** (2024)  
  - **0** (no deforestation)  
- Resolution: **50 m**, CRS: **EPSG:32636**, Int16  

### Validation Metrics
- True Positive / False Positive / False Negative / True Negative  
- Precision, Recall, F1 Score  
- **Agreement maps**:  
  - S1 vs Hansen  
  - S1 vs Tree Height  

---

## 4) KPI Computation

**Notebook:** `4_kpi.ipynb`

### Inputs
- `forest_extent_2020_32636.tif` (10 m)
- `s1_deforestation_year_20-24.tif` (float32; floor → 2021–2024)
- `meancrops_24-25_32636.tif` (crop probability; cropland = >0.1)

### KPIs (per county)
### Inputs
- **MeanCrops 24-25**  
  Download using GEE script: `0_download_crops_gee.js`
  - `meancrops_24-25_32636.tif` 
- **Forest baseline (2020)**  
  from Step 1:  
  - `forest_extent_2020_32636.tif`
- **S1 Deforestation Date**  
  from Step 3:
  - `s1_deforestation_year_20-24.tif`

#### KPI 0 — Forest Baseline (ha)
`Forest_2020_ha`

#### KPI 1 — Annual Forest Loss (ha)
- `Loss_2021_ha`, `Loss_2022_ha`, `Loss_2023_ha`, `Loss_2024_ha`
- `Total_Loss_ha`
- `Rate_2021_%`, `Rate_2022_%`, `Rate_2023_%`, `Rate_2024_%`
- `Avg_Annual_Rate_%`

#### KPI 2 — Forest → Crop Conversion
- `Crop_2021_ha`, `Crop_2022_ha`, `Crop_2023_ha`, `Crop_2024_ha`
- `Crop_2021_%`, `Crop_2022_%`, `Crop_2023_%`, `Crop_2024_%`
- `Total_Crop_ha`, `KPI_2_Total_Crop_%`

### Output
`forest_kpi_yearly.csv`

---

## Environment

Create the analysis environment:

```bash
conda env create -f environment.yml