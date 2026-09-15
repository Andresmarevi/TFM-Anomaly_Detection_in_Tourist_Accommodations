# 🏨 Automated Anomaly Detection Pipeline for Spanish Tourist Accommodations

**Master's Thesis Project (TFM) - Unsupervised Machine Learning**

An end-to-end automated data pipeline that scrapes, processes, and analyzes price anomalies in Spanish tourist accommodations using **Isolation Forest** and **Local Outlier Factor (LOF)** models, with an interactive Streamlit dashboard for results exploration.

---

## 📋 Project Overview

This project implements a complete MLOps workflow to detect anomalies in Booking.com accommodation data across Spain. It combines:

- **Web Scraping**: Automated data collection from Booking.com
- **ETL & Feature Engineering**: Data cleaning, transformation, and feature extraction
- **Unsupervised Learning**: Dual anomaly detection models (IF + LOF)
- **Analysis & Validation**: Consensus analysis and model agreement assessment
- **Interactive Dashboard**: Streamlit-based visualization and exploration tool
- **Automated Reporting**: Jupyter notebooks exported as HTML reports

### Key Objectives

1. Identify price anomalies in the Spanish tourism market
2. Detect both global anomalies (Isolation Forest) and local density-based anomalies (LOF)
3. Analyze model agreement and disagreement patterns
4. Provide business rules validation for detected anomalies
5. Enable interactive exploration through a web dashboard

---

## 🏗️ Project Structure

```
6. Automated pipeline/
├── main.py                          # Main orchestrator - runs full pipeline
├── requirements.txt                 # Python dependencies
├── run_pipeline.bat                 # Windows batch script to run pipeline
│
├── scraper/                         # Web scraping module
│   ├── main_pipeline.py             # Orchestrates scraping & enrichment phases
│   ├── src/
│   │   ├── booking_scraper.py       # Booking.com API queries & HTML parsing
│   │   ├── enrichment.py            # Data cleaning & geographic enrichment
│   │   ├── alojamiento.py           # Accommodation data class
│   │   ├── utils.py                 # Helper functions (validators, loaders)
│   │   └── payload.json             # Booking.com GraphQL query templates
│   ├── data/
│   │   ├── accommodation_type_dict.json
│   │   ├── localizaciones.csv       # Spanish location master data
│   │   ├── Piscinas.csv             # Pool amenity reference
│   │   └── TodoEspañaSinCeros.csv   # Geographic coordinates reference
│   └── output/
│       ├── raw/                     # Raw scraped data (unprocessed)
│       ├── final/                   # Cleaned & enriched data (ready for ML)
│       └── trace/                   # Scraping metadata & execution logs
│
├── pipeline/                        # ML pipeline & modeling module
│   ├── run_pipeline.py              # Processes snapshots and builds temporal outputs
│   ├── src/
│   │   ├── etl.py                   # Extract, Transform, Load & feature engineering
│   │   ├── model.py                 # Isolation Forest & LOF training
│   │   ├── evaluate.py              # Results analysis & consensus validation
│   │   ├── temporal_summary.py       # Temporal aggregation of archived snapshots
│   └── models/                      # Trained model artifacts
│       ├── isolation_forest_final.joblib
│       └── lof_final_params.json
│
├── data/
│   ├── processed/
│   │   ├── dataset_anomaly_detection_ready.csv  # Latest ML-ready dataset
│   │   ├── dataset_raw_features.csv             # Latest features and business rules
│   │   └── history/                              # Dated processed snapshots
│   └── results/                     # Model outputs & analysis tables
│       ├── model_results_full.csv   # Complete predictions + features
│       ├── temporal_evolution.csv   # Metrics by extraction date and horizon
│       ├── history/                  # Dated model results by snapshot
│       ├── model_results.csv        # Summary metrics
│       ├── model_config.json        # Model hyperparameters & config
│       ├── anomaly_analysis_summary.csv
│       ├── anomaly_by_property_type.csv
│       ├── anomaly_by_region.csv
│       ├── anomaly_model_comparison.csv
│       ├── consensus_anomalies.csv  # Cases where IF & LOF agree
│       ├── disagreement_cases.csv   # Cases where IF & LOF differ
│       ├── top_if_anomalies.csv
│       └── top_lof_anomalies.csv
│
├── dashboard/                       # Interactive Streamlit web app
│   └── app.py                       # Main Streamlit dashboard
│
├── notebooks/                       # Analysis & validation notebooks
│   ├── EDA_anomaly_detection.ipynb          # Exploratory Data Analysis
│   ├── Model_Training.ipynb                 # Model training & sensitivity analysis
│   ├── Anomaly_Analysis_Validation.ipynb    # Results validation & insights
│   ├── figures/                     # Generated plots & visualizations
│   ├── models/                      # Model snapshots
│   └── outputs/                     # Notebook execution outputs
│
└── reports/                         # Generated HTML reports
    ├── EDA_anomaly_detection_report.html
    ├── Model_Training_report.html
    ├── Anomaly_Analysis_Validation_report.html
    └── figures/                     # High-resolution report figures
```

---

## 🔄 Pipeline Workflow

### Overall Flow

```
[Web Scraping] → [ETL + Features] → [IF + LOF] → [Historical Archive] → [Temporal Summary] → [Dashboard]
   scraper/       etl.py              model.py      run_pipeline.py       temporal_summary.py  app.py
```

The ML pipeline processes each dated scraper snapshot independently. The latest
outputs keep their stable generic filenames, while dated copies are stored for
temporal comparison. Each snapshot is identified by:

- `extraction_date`: date on which the data was collected.
- `lookahead_period`: booking horizon, either `1m` or `3m`.

These metadata fields are retained for traceability and temporal analysis, but
are not used as model features.

When more than one extraction exists in the same ISO week for the same horizon,
the pipeline keeps the latest extraction date and ignores older duplicates for
temporal processing. This rule is applied while processing scraper inputs and
while loading archived model results. Original files are not deleted, so they
remain available for audit purposes.

### Detailed Phases

#### Phase 1: Web Scraping (`scraper/main_pipeline.py`)

**Purpose**: Collect accommodation data from Booking.com

**Steps**:

1. **Booking API Queries** (`booking_scraper.py`)

   - Query Booking.com GraphQL API for accommodation listings
   - Collect data for 1-month and 3-month lookahead periods
   - Extract: price, rooms, bathrooms, location, ratings, amenities
   - Geographic filtering using cell-based grid system
2. **Data Enrichment** (`enrichment.py`)

   - Geographic data enrichment (region, province, municipality)
   - Pool amenity matching
   - Coordinate validation
   - Data cleaning & standardization
   - Output: Clean CSV files ready for ML

**Inputs**:

- Booking.com website/API
- Reference data (locations, coordinates, amenities)

**Outputs**:

- `scraper/output/final/Booking*.csv` - Clean accommodation data

The standard command generates one file per horizon, for example
`Booking2026-09-11_1m.csv` and `Booking2026-09-11_3m.csv`.

---

#### Phase 2: Machine Learning Pipeline (`pipeline/run_pipeline.py`)

##### Step 2.1: ETL & Feature Engineering (`src/etl.py`)

**Purpose**: Transform raw data into ML-ready features

**Process**:

- Load cleaned scraper data from `scraper/output/final/`
- Data quality validation (duplicates, missing values, outliers)
- Statistical transformations (log normalization for prices)
- Feature engineering:
  - Derived metrics (price per sqm, price per room, sqm per room, etc.)
  - Categorical encoding (accommodation type, region)
  - Feature scaling & normalization (RobustScaler)
- **Business Rules Generation**: Create flags for anomalies based on domain logic:
  - `extreme_price_high` / `extreme_price_low`
  - `suspected_placeholder_sqm`
  - `extreme_price_per_sqm`
  - `overcrowded_layout`, `excess_bathrooms`
  - `tiny_villa`, `huge_apartment`
  - `any_flag` - Composite flag when any rule triggers
- **Outputs**:
  - `data/processed/dataset_anomaly_detection_ready.csv` - Scaled features for ML models (features only)
   - `data/processed/dataset_raw_features.csv` - Raw features, business rules, price, sqm and snapshot metadata
   - `data/processed/history/` - Dated copies of processed snapshots

The input snapshot can be selected with the optional `PIPELINE_INPUT_FILE`
environment variable. Without it, the most recent `Booking*.csv` file is used.

---

##### Step 2.2: Model Training (`src/model.py`)

**Purpose**: Train dual anomaly detection models

**Models**:

1. **Isolation Forest (Global Anomalies)**

   - Detects anomalies through feature space partitioning
   - Effective for high-dimensional data
   - Configuration: `contamination=0.05` (5% anomaly rate)
   - Output: `if_score` (anomaly score), `if_anomaly` (-1=anomaly, 1=normal)
2. **Local Outlier Factor (Local Anomalies)**

   - Density-based approach - finds local density deviations
   - Captures neighborhood-specific anomalies
   - Configuration: `n_neighbors=20`, `contamination=0.05`
   - Output: `lof_score`, `lof_anomaly` (-1=anomaly, 1=normal)

**Outputs**:

- Model files saved to `pipeline/models/`
- Predictions saved to `data/results/model_results_full.csv` (including snapshot metadata, price, sqm, and business rule flags)
- Dated model results saved to `data/results/history/model_results_full_<date>_<period>.csv`
- Configuration saved to `data/results/model_config.json`
- Raw model results to `data/results/model_results.csv` (predictions only)

`pipeline/run_pipeline.py` processes only snapshots that do not already have
all expected archived outputs. It then rebuilds the temporal summary and runs
the validation analysis. Notebook-to-HTML export is enabled by default by the
Python orchestrator and can be disabled with `EXPORT_REPORTS=0`.

---

##### Step 2.3: Analysis & Validation (`src/evaluate.py`)

**Purpose**: Analyze results, measure agreement, generate insights

**Analysis Components**:

1. **Model Consensus Analysis**

   - Identify cases where IF and LOF agree (consensus anomalies)
   - Identify disagreement cases (different perspectives)
   - Calculate agreement metrics
2. **Anomaly Characterization**

   - Anomaly distribution by property type
   - Regional anomaly patterns
   - Price anomaly indicators
   - Statistical profiles of anomalous vs. normal listings
3. **Business Rules Validation**

   - Compare model outputs with business rules:
     - Extreme high/low prices
     - Suspicious price-per-sqm ratios
     - Placeholder/dummy data indicators
     - Structural anomalies (extreme bathrooms/rooms)
   - Quantify model agreement with business rules
4. **Representative Cases Selection**

   - Identify most significant anomalies
   - Create summary tables for each model

**Outputs** (to `data/results/`):

- `model_results.csv` - Summary statistics
- `consensus_anomalies.csv` - High-confidence anomalies (both models agree)
- `disagreement_cases.csv` - Cases with model disagreement
- `anomaly_by_property_type.csv` - Anomaly patterns by accommodation type
- `anomaly_by_region.csv` - Geographic anomaly distribution
- `top_if_anomalies.csv` - Top Isolation Forest anomalies
- `top_lof_anomalies.csv` - Top LOF anomalies
- Various visualizations for reporting

##### Step 2.4: Temporal Summary (`src/temporal_summary.py`)

This module aggregates the selected weekly archived model outputs by
`extraction_date` and `lookahead_period`. It creates
`data/results/temporal_evolution.csv` with:

- Number of listings per snapshot.
- Median price and median price per square metre.
- Isolation Forest, LOF and consensus anomaly rates.
- Business-rule signal rate.

The selection contains at most one latest snapshot per ISO week and horizon
(`1m` or `3m`). The summary can still contain multiple rows for one extraction
date when different check-in or checkout dates are present, because those rows
represent different planned stays rather than duplicate extractions.

The summary makes it possible to compare market levels and anomaly prevalence
over time without mixing different booking horizons or retraining one model on
all historical snapshots at once.

---

#### Phase 3: Reporting & Dashboarding

**Jupyter Notebooks** (`notebooks/`):

- **EDA_anomaly_detection.ipynb**: Raw data exploration and quality audit
- **Model_Training.ipynb**: Model training, hyperparameter sensitivity analysis
- **Anomaly_Analysis_Validation.ipynb**: Results interpretation and validation

**Dashboard** (`dashboard/app.py`):

- Interactive Streamlit web application
- Real-time data exploration
- Filtering by anomaly type, region, property type
- Statistical visualizations (Plotly)
- Model comparison tools
- Business rule overlay analysis
- Temporal market evolution by extraction date and booking horizon

**HTML Reports** (`reports/`):

- Auto-generated from notebooks
- Publication-ready visualizations
- Static snapshots for thesis documentation

---

## ⚙️ Dependencies

```
pandas           - Data manipulation & analysis
numpy            - Numerical computing
scikit-learn     - Machine learning (IF, LOF)
requests         - HTTP client for web scraping
streamlit        - Dashboard framework
plotly           - Interactive visualizations
scipy            - Statistical functions
joblib           - Model serialization
```

See `requirements.txt` for complete dependency list and versions.

---

## 🚀 Quick Start

### 1. Installation

```powershell
# Navigate to the project
cd "6. Automated pipeline"

# Create and activate the virtual environment (Windows PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install -r requirements.txt

# (Optional) For Jupyter notebook export as HTML
pip install jupyter nbconvert
```

### 2. Running the Complete Pipeline

#### Option A: Windows Batch Script (Recommended)

```powershell
.\run_pipeline.bat
```

#### Option B: Python Command

```powershell
venv\Scripts\python.exe main.py
```

#### Option C: Run Individual Phases

**Phase 1 - Scraping Only**:

```powershell
cd scraper
..\venv\Scripts\python.exe main_pipeline.py [1m|3m|all]
# 1m  = 1-month lookahead scrape
# 3m  = 3-month lookahead scrape  
# all = both (default)
```

**Phase 2 - ML Pipeline Only**:

```powershell
cd pipeline
..\venv\Scripts\python.exe run_pipeline.py
```

### 3. Run Interactive Dashboard

```powershell
cd dashboard
..\venv\Scripts\python.exe -m streamlit run app.py
```

The dashboard will be available at `http://localhost:8501`

### 4. Weekly automatic execution on Windows

The intended scheduled workflow is:

```text
Wednesday 07:00
   ↓
run_pipeline.bat
   ↓
main.py
   ↓
new 1m and 3m Booking snapshots
   ↓
incremental ETL + IF/LOF + temporal summary
   ↓
dashboard reads the updated CSV results
```

Configure Windows Task Scheduler with:

1. Create a basic task named, for example, `Booking anomaly pipeline`.
2. Trigger: weekly, Wednesday, at `07:00`.
3. Action: start the file `run_pipeline.bat`.
4. Start in: the project root directory.
5. Enable running whether the user is logged on or not if the machine must run unattended.
6. Configure the task to avoid starting a new instance while a previous run is still active.
7. Set an execution time limit appropriate for the scraper and historical processing.

The batch file uses the project's `venv` interpreter, so Python does not need to be
resolved through the system `PATH`. Dependencies should be installed once before
creating the task:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

Scheduled runs do not reinstall packages or export notebook reports. Their output
is recorded in `pipeline_cron.log`, while the Python orchestration log is stored in
`pipeline_execution.log`. A non-zero exit code indicates that the task failed.

The dashboard does not need to be restarted after a successful run. Its cache
signature includes the modification time and size of the latest result file, so a
new pipeline output is reloaded automatically on the next Streamlit rerun. The
`Refresh Data Pipeline` button can be used to force an immediate rerun.

### 5. View Reports

Open HTML reports from `reports/` in any web browser:

- `EDA_anomaly_detection_report.html`
- `Model_Training_report.html`
- `Anomaly_Analysis_Validation_report.html`

---

## 📊 Dashboard Features

The Streamlit dashboard provides interactive exploration of anomaly detection results:

### Key Pages/Sections

1. **Overview**: total listings, IF/LOF/consensus counts, Jaccard similarity,
   Spearman correlation, and model intersection charts.
2. **Price anomaly analysis**: box and violin plots for absolute, per-square-metre,
   property-type-relative, and province-relative prices.
3. **Market anomaly explorer**: ranked low-price and high-price anomaly
   candidates with a supporting value score.
4. **Listing summary**: search by listing name and inspect model scores, price
   context, detection status, and triggered business rules.
5. **Segment analysis**: compare counts and rates by property type, region, or
   province, with a minimum-observation filter.
6. **Business rules validation**: compare IF, LOF, and consensus detections with
   interpretable heuristic signals.
7. **Final consensus ranking**: rank strict consensus, IF-only, or LOF-only
   cases using percentile-normalized model scores and contextual explanations.
8. **Temporal market evolution**: weekly market levels, persistent anomalies,
   seasonal analysis, horizon comparisons, and coverage diagnostics.

All pages use the global sidebar filters for region, property type, model status,
price, surface, and contextual price anomalies. Temporal views apply their own
historical filters because they read archived snapshots rather than only the
latest result.

---

## 📈 Data Schema

### Input Data (Scraper Output)

| Column          | Type   | Description                                     |
| --------------- | ------ | ----------------------------------------------- |
| id              | string | Unique accommodation ID                         |
| nombre          | string | Property name                                   |
| tipo            | string | Accommodation type (apt, villa, bungalow, etc.) |
| numHabitaciones | int    | Number of bedrooms                              |
| numBanos        | int    | Number of bathrooms                             |
| metros          | float  | Property size in sqm                            |
| estrellas       | float  | Booking.com rating (0-5)                        |
| precio          | float  | Nightly price (EUR)                             |
| capacidad       | int    | Guest capacity                                  |
| nombreCCAA      | string | Spanish region (Autonomous Community)           |
| nombreProvincia | string | Province name                                   |
| coordenadaX     | float  | Longitude                                       |
| coordenadaY     | float  | Latitude                                        |
| piscina         | bool   | Has pool                                        |
| primeraLinea    | bool   | Beachfront property                             |

### ML-Ready Dataset

Input features for anomaly detection include:

- **Price metrics**: price, price_per_sqm, log_price
- **Property characteristics**: num_rooms, num_bathrooms, property_size, guest_capacity
- **Location**: region, province, latitude, longitude
- **Amenities**: pool, beachfront
- **Quality**: rating, review_count
- **Encoded categories**: accommodation_type (one-hot)

### Model Output (`model_results_full.csv`)

| Column                  | Type   | Description                                     |
| ----------------------- | ------ | ----------------------------------------------- |
| id                      | string | Accommodation ID                                |
| *all input features*  | mixed  | Original features                               |
| if_score                | float  | Isolation Forest anomaly score                  |
| if_anomaly              | int    | IF prediction (-1=anomaly, 1=normal)            |
| lof_score               | float  | LOF anomaly score                               |
| lof_anomaly             | int    | LOF prediction (-1=anomaly, 1=normal)           |
| is_if_anomaly           | bool   | True if IF detected anomaly                     |
| is_lof_anomaly          | bool   | True if LOF detected anomaly                    |
| status                  | string | Detection status: "Normal", "IF", "LOF", "Both" |
| *business_rule_flags* | bool   | Various business rule indicators                |

---

## 🎯 Key Findings & Metrics

### Anomaly Detection Statistics

After pipeline execution, review `data/results/` for:

- **Detection Rates**

  - Total anomalies detected by IF and LOF
  - Consensus anomalies (both models agree)
  - Disagreement rate between models
- **Geographic Distribution**

  - Regions with highest anomaly concentration
  - Province-level anomaly density
- **Property Type Analysis**

  - Anomaly rates by accommodation type
  - Type-specific anomaly patterns
- **Price Anomalies**

  - Distribution of anomalous prices
  - Comparison: normal vs. anomalous listings
  - Extreme price indicators

### Temporal Evolution

The historical view is based on modelled, enriched snapshots rather than raw
scraper files. Raw files remain available for auditability, while the temporal
dashboard uses `model_results_full_<date>_<period>.csv` because these files
contain the geographic, price and anomaly information needed for interpretation.

Review `data/results/temporal_evolution.csv` for the aggregated temporal metrics.

---

## 🔍 Interpretation Guide

### Model Outputs

**Isolation Forest (`if_anomaly`, `if_score`)**:

- Detects **global outliers** in feature space
- Effective for detecting listings that deviate significantly across multiple dimensions
- Score: negative = more anomalous
- Good for: finding truly unusual properties (very high/low prices, suspicious configurations)

**Local Outlier Factor (`lof_anomaly`, `lof_score`)**:

- Detects **local density anomalies**
- Captures observations that are outliers in their neighborhood
- Effective for regional market anomalies
- Good for: finding regional pricing anomalies, local market oddities

### Using the Results

1. **High Confidence Anomalies**: Look at `consensus_anomalies.csv`

   - Both models agree → Very likely anomalous
   - Use for business rule enforcement
2. **Model-Specific Insights**:

   - **IF-only anomalies**: Global market outliers, structural anomalies
   - **LOF-only anomalies**: Regional market inefficiencies, local pricing errors
3. **Disagreement Analysis**: See `disagreement_cases.csv`

   - Different perspectives on what constitutes an anomaly
   - Investigate to understand market segments
4. **Business Rule Validation**:

   - Verify model detection against known business rules
   - Fine-tune model configuration based on rule agreement

---

## 🛠️ Configuration & Customization

### Model Hyperparameters

Edit in `pipeline/src/model.py`:

**Isolation Forest**:

```python
IsolationForest(
    contamination=0.05,     # Expected % of anomalies
    random_state=RANDOM_STATE,
    n_estimators=100,       # Number of trees
)
```

**Local Outlier Factor**:

```python
LocalOutlierFactor(
    n_neighbors=20,         # Neighborhood size
    contamination=0.05,     # Expected % of anomalies
)
```

### ETL Configuration

Modify in `pipeline/src/etl.py`:

- Feature selection
- Scaling & normalization methods
- Outlier handling
- Category encoding approach

### Scraper Configuration

Adjust in `scraper/main_pipeline.py`:

- Geographic coverage area
- Search date ranges
- Data enrichment rules
- Output formats

### Dashboard Customization

Modify in `dashboard/app.py`:

- Color scheme (`COLORS` dictionary)
- Filter options
- Visualization types
- Business rule definitions

---

## 📊 Performance & Scaling

### Data Volume

Current configuration handles:

- **~1,000-10,000** accommodations per scrape
- **50+ features** post-engineering
- **Real-time** anomaly scoring on new data

### Processing Times

Typical execution:

- **Scraping**: 20-30 minutes (1m+3m periods, threaded)
- **ETL**: 2-3 minutes
- **Model training**: 1-2 minutes
- **Analysis & reporting**: 1-2 minutes
- **Total pipeline**: ~35 minutes

### Optimization Tips

1. **Parallel scraping**: `booking_scraper.py` uses `ThreadPoolExecutor`
2. **Memory efficiency**: Vectorized pandas operations
3. **Model persistence**: Serialized models for faster re-scoring
4. **Caching**: Dashboard uses Streamlit `@st.cache_data`

---

## 🐛 Troubleshooting

### Common Issues

| Issue                                    | Solution                                                             |
| ---------------------------------------- | -------------------------------------------------------------------- |
| `FileNotFoundError: dataset not found` | Run scraper first:`cd scraper && python main_pipeline.py`          |
| Model import errors                      | Install scikit-learn:`pip install scikit-learn`                    |
| Dashboard won't load data                | Check CSV separator (`;` vs `,`) in `dashboard/app.py` line 50 |
| Encoding errors in CSV                   | Use UTF-8 encoding, check enrichment.py output                       |
| Jupyter notebook export fails            | Install nbconvert:`pip install nbconvert jupyter`                  |
| Booking scraper blocked/slow             | Check internet connection, Booking.com may rate-limit requests       |

### Logging

All execution logs are saved to:

- `pipeline_execution.log` - Main pipeline log
- Console output during execution
- Scraper trace data in `scraper/output/trace/`

---

## 📂 File Descriptions

### Core Scripts

| File                 | Purpose                                          |
| -------------------- | ------------------------------------------------ |
| `main.py`          | Master orchestrator - runs full 2-phase pipeline |
| `run_pipeline.bat` | Windows batch launcher                           |
| `requirements.txt` | Python package dependencies                      |

### Scraper Module

| File                               | Purpose                       |
| ---------------------------------- | ----------------------------- |
| `scraper/main_pipeline.py`       | Scraper phase orchestrator    |
| `scraper/src/booking_scraper.py` | Booking API client & parser   |
| `scraper/src/enrichment.py`      | Data cleaning & enrichment    |
| `scraper/src/alojamiento.py`     | Data class for accommodations |
| `scraper/src/utils.py`           | Helper functions              |

### Pipeline Module

| File                         | Purpose                                 |
| ---------------------------- | --------------------------------------- |
| `pipeline/run_pipeline.py` | ML pipeline orchestrator                |
| `pipeline/src/etl.py`      | Extract, transform, feature engineering |
| `pipeline/src/model.py`    | Model training & prediction             |
| `pipeline/src/evaluate.py` | Analysis & validation                   |
| `pipeline/src/temporal_summary.py` | Historical metric aggregation     |

### Analysis & Visualization

| File                                            | Purpose                         |
| ----------------------------------------------- | ------------------------------- |
| `dashboard/app.py`                            | Interactive Streamlit dashboard |
| `notebooks/EDA_anomaly_detection.ipynb`       | Exploratory data analysis       |
| `notebooks/Model_Training.ipynb`              | Model training details          |
| `notebooks/Anomaly_Analysis_Validation.ipynb` | Results analysis                |

## Quick Reference Commands

```bash
# Full pipeline (Phase 1 + 2)
python main.py

# Scraping only (Phase 1)
cd scraper && python main_pipeline.py

# ML pipeline only (Phase 2)  
cd pipeline && python run_pipeline.py

# Dashboard
cd dashboard && streamlit run app.py

# View generated reports
reports/EDA_anomaly_detection_report.html
reports/Model_Training_report.html
reports/Anomaly_Analysis_Validation_report.html

# Check results
data/results/model_results_full.csv           # All predictions
data/results/temporal_evolution.csv           # Evolution by date and horizon
data/results/history/                         # Dated model outputs
data/results/consensus_anomalies.csv          # High-confidence
data/results/anomaly_by_region.csv            # Geographic analysis
```

---

**Project Version**: 1.0
