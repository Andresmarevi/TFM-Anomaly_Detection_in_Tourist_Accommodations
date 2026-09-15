#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis: Tourist Accommodation Anomaly Detection
# 
# The purpose of this notebook is to explore, clean, engineer and prepare the scraped accommodation
# dataset so that it is ready for unsupervised anomaly detection with **Isolation Forest** and
# **Local Outlier Factor (LOF)**.
# 
# **Notebook structure**
# 
# 1. Data loading & overview
# 2. Data quality audit (types, missing values, duplicates, consistency)
# 3. Univariate analysis
# 4. Geographic analysis
# 5. Bivariate / multivariate analysis (correlations, price drivers)
# 6. Data cleaning
# 7. Feature engineering
# 8. Variable treatment & transformations
# 9. Preliminary anomaly detection (business rules)
# 10. Dimensionality reduction preview (PCA)
# 11. Risks and limitations of the dataset
# 12. Final dataset preparation for Isolation Forest / LOF

# ## 1. Setup

# In[1]:


import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

try:
    display
except NameError:
    display = print

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)

plt.style.use("default")

print("Libraries loaded")


# In[2]:
import os
import re
import pandas as pd
from pathlib import Path

# Resolver la raíz del proyecto (sube 3 niveles: src -> pipeline -> raíz)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
SCRAPER_FINAL_DIR = ROOT_DIR / "scraper" / "output" / "final"

# Buscar el archivo indicado para reprocesamiento o, por defecto, el más reciente.
configured_input = os.environ.get("PIPELINE_INPUT_FILE")
if configured_input:
    latest_raw_file = Path(configured_input).resolve()
    if not latest_raw_file.exists():
        raise FileNotFoundError(f"No se encontró el archivo indicado: {latest_raw_file}")
else:
    raw_files = list(SCRAPER_FINAL_DIR.glob("Booking*_3m.csv")) or list(SCRAPER_FINAL_DIR.glob("Booking*.csv"))
    if not raw_files:
        raise FileNotFoundError(f"No se encontró ningún archivo Booking*.csv en: {SCRAPER_FINAL_DIR}")
    latest_raw_file = max(raw_files, key=os.path.getctime)

print(f"Cargando dataset más reciente: {latest_raw_file.name}")

filename_match = re.search(
    r"Booking(?P<extraction_date>\d{4}-\d{2}-\d{2})_(?P<lookahead_period>1m|3m)",
    latest_raw_file.stem,
)
if not filename_match:
    raise ValueError(
        f"El nombre del archivo no contiene una fecha y periodo válidos: "
        f"{latest_raw_file.name}"
    )

extraction_date = filename_match.group("extraction_date")
lookahead_period = filename_match.group("lookahead_period")

# Some historical final snapshots were exported without a header. Their column
# order is the same as the enriched scraper output, so it can be restored here.
source_columns = [
    "id", "nombre", "idHabitacion", "nombreHabitacion", "idLocalizacion",
    "nombreCCAA", "nombreProvincia", "nombreMunicipio", "nombreLocalizacion",
    "tipo", "numHabitaciones", "numBanos", "metros", "piscina_flag",
    "playa_flag", "estrellas", "precio", "oferta", "precioSinOferta",
    "capacidad", "desayuno", "cancelacion", "fechaActual", "fechaEntrada",
    "fechaSalida",
]

# Load raw data
df_raw = pd.read_csv(
    latest_raw_file,
    sep=";",
    encoding="latin1",
    low_memory=False
)
if not {"id", "nombre", "metros", "precio"}.issubset(df_raw.columns):
    if df_raw.shape[1] != len(source_columns):
        raise ValueError(
            f"Formato inesperado en {latest_raw_file.name}: "
            f"se esperaban {len(source_columns)} columnas y hay {df_raw.shape[1]}"
        )
    df_raw = pd.read_csv(
        latest_raw_file,
        sep=";",
        header=None,
        names=source_columns,
        encoding="latin1",
        low_memory=False,
    )
    print("Cabecera ausente detectada; nombres de columnas restaurados.")

print("=" * 60)
print("DIMENSIONS")
print("=" * 60)
print(f"Rows: {df_raw.shape[0]:,}")
print(f"Columns: {df_raw.shape[1]}")

print("\nFirst rows:")
print(df_raw.head())


# In[3]:


# Translate column names to English for a fully English notebook.

column_mapping = {
    "id": "id",
    "nombre": "name",
    "idHabitacion": "room_id",
    "nombreHabitacion": "room_name",
    "idLocalizacion": "location_id",
    "nombreCCAA": "region",
    "nombreProvincia": "province",
    "nombreMunicipio": "municipality",
    "nombreLocalizacion": "locality",
    "tipo": "property_type",
    "numHabitaciones": "num_rooms",
    "numBanos": "num_bathrooms",
    "metros": "sqm",
    "piscina_flag": "pool_flag",
    "playa_flag": "beach_flag",
    "estrellas": "stars",
    "precio": "price",
    "oferta": "discount_amount",
    "precioSinOferta": "price_without_discount",
    "capacidad": "capacity",
    "desayuno": "breakfast",
    "cancelacion": "cancellation",
    "fechaActual": "current_date",
    "fechaEntrada": "checkin_date",
    "fechaSalida": "checkout_date",
}

df = df_raw.rename(columns=column_mapping)
df.columns.tolist()

numeric_source_columns = [
    "id", "room_id", "location_id", "num_rooms", "num_bathrooms",
    "metros", "pool_flag", "beach_flag", "stars", "price",
    "discount_amount", "price_without_discount", "capacity",
    "breakfast", "cancellation",
]
for column in numeric_source_columns:
    if column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

checkin_date = (
    df["checkin_date"].dropna().astype(str).iloc[0]
    if "checkin_date" in df and not df["checkin_date"].dropna().empty
    else ""
)
checkout_date = (
    df["checkout_date"].dropna().astype(str).iloc[0]
    if "checkout_date" in df and not df["checkout_date"].dropna().empty
    else ""
)


# In[4]:


print("=" * 60)
print("GENERAL INFO")
print("=" * 60)

df.info()

variable_summary = pd.DataFrame({
    "variable": df.columns,
    "dtype": df.dtypes.astype(str),
    "non_null": df.notnull().sum(),
    "nulls": df.isnull().sum()
})

display(variable_summary)


# In[5]:


numeric_vars = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_vars = df.select_dtypes(include=["object", "string"]).columns.tolist()

print("=" * 60)
print("VARIABLE TYPES")
print("=" * 60)
print(f"Numeric variables ({len(numeric_vars)}): {numeric_vars}")
print(f"Categorical variables ({len(categorical_vars)}): {categorical_vars}")

cardinality = pd.DataFrame({
    "variable": df.columns,
    "unique_values": [df[c].nunique() for c in df.columns]
}).sort_values("unique_values")

display(cardinality)


# In[6]:


print("=" * 60)
print("EXECUTIVE SUMMARY")
print("=" * 60)

print(f"Records: {len(df):,}")
print(f"Variables: {len(df.columns)}")
print(f"Numeric variables: {len(numeric_vars)}")
print(f"Categorical variables: {len(categorical_vars)}")
print(f"Exact duplicates: {df.duplicated().sum()}")

display(df[numeric_vars].describe())


# ### Initial conclusions
# 
# - The dataset contains 109,352 records and 25 variables.
# - There are 716 exact duplicate rows.
# - `sqm` and `stars` are stored as text/mixed types and need numeric conversion.
# - `capacity`, `current_date`, `checkin_date` and `checkout_date` have cardinality 1 (constant across the whole dataset), they are not useful for our analysis.
# - Identifier variables (`id`, `room_id`, `location_id`, `name`, `room_name`) have very
#   high or maximal cardinality and are not usable as model features, we will only use them as traceability keys.

# ## 2. Data Quality Audit

# In[7]:


# Working copy + numeric conversion
df_eda = df.copy()

df_eda["sqm"] = pd.to_numeric(df_eda["sqm"], errors="coerce")
df_eda["stars"] = pd.to_numeric(df_eda["stars"], errors="coerce")

print(df_eda.dtypes)


# In[8]:


nulls = pd.DataFrame({
    "variable": df_eda.columns,
    "nulls": df_eda.isnull().sum(),
    "pct": round(df_eda.isnull().mean() * 100, 2)
}).sort_values("pct", ascending=False)

display(nulls)
display(nulls[nulls["nulls"] > 0])


# In[9]:


for var in ["stars", "sqm"]:
    print("\n" + "=" * 60)
    print(f"NULLS IN {var.upper()} BY PROPERTY TYPE")
    print("=" * 60)
    table = pd.crosstab(df_eda["property_type"], df_eda[var].isna(), normalize="index") * 100
    display(table.round(2))


# In[10]:


duplicates = df_eda.duplicated()
print("=" * 60)
print("DUPLICATES")
print("=" * 60)
print(f"Exact duplicate rows: {duplicates.sum()}")
print(f"Percentage of total: {duplicates.sum() / len(df_eda) * 100:.2f}%")


# In[11]:


cardinality = pd.DataFrame({
    "variable": df_eda.columns,
    "unique_values": [df_eda[c].nunique() for c in df_eda.columns]
}).sort_values("unique_values")

display(cardinality)


# In[12]:


print("=" * 60)
print("CONSISTENCY CHECKS")
print("=" * 60)

print("Rooms = 0:", (df_eda["num_rooms"] == 0).sum())
print("Bathrooms = 0:", (df_eda["num_bathrooms"] == 0).sum())
print("Sqm <= 0:", (df_eda["sqm"] <= 0).sum())
print("Sqm < 10:", (df_eda["sqm"] < 10).sum())
print("Price <= 0:", (df_eda["price"] <= 0).sum())
print("Negative discount:", (df_eda["discount_amount"] < 0).sum())


# In[13]:


print("=" * 60)
print("HIGHEST PRICES")
print("=" * 60)
display(
    df_eda[["name", "property_type", "province", "sqm", "price"]]
    .sort_values("price", ascending=False).head(20)
)

print("=" * 60)
print("SMALLEST SQM")
print("=" * 60)
display(
    df_eda[["name", "property_type", "sqm", "price"]]
    .sort_values("sqm", ascending=True).head(20)
)


# ### Data quality conclusions
# 
# - **Missing values**: the variables with the highest share of missing data are `stars` (91.17%, 99,693 records) and `sqm` (15.46%, 16,903 records). After we check with `property_type` we can check that it is not random: hotels report stars far more consistently than private rentals (apartments, villas), so `stars` behaves more like a hotel-only attribute than a universal feature.
# - **Duplicates**: 716 exact duplicate rows (0.65%).
# - **Extreme values** already visible at this stage (near-zero sqm, very high prices) are exactly the kind of records the we are meant to detect later, we should not mark them as "errors" without justification (see Section 9).

# ## 3. Univariate Analysis

# In[14]:


numeric_analysis_vars = [
    "price", "price_without_discount", "discount_amount",
    "sqm", "num_rooms", "num_bathrooms", "stars"
]
numeric_analysis_vars


# In[15]:


display(df_eda[numeric_analysis_vars].describe().T)


# In[16]:


for var in numeric_analysis_vars:
    print("\n" + "=" * 60)
    print(var.upper())
    print("=" * 60)
    print(df_eda[var].quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]))


# In[17]:


for var in numeric_analysis_vars:
    plt.figure(figsize=(10, 5))
    df_eda[var].hist(bins=50)
    plt.title(f"Distribution of {var}")
    plt.xlabel(var)
    plt.ylabel("Frequency")
    ##plt.show()


# In[18]:


plt.figure(figsize=(10, 5))
np.log1p(df_eda["price"]).hist(bins=50)
plt.title("Log-transformed price distribution")
plt.xlabel("log(price)")
plt.ylabel("Frequency")
#plt.show()


# In[19]:


# sqm is as right-skewed as price -- check the log transform here too
plt.figure(figsize=(10, 5))
np.log1p(df_eda["sqm"]).hist(bins=50)
plt.title("Log-transformed sqm distribution")
plt.xlabel("log(sqm)")
plt.ylabel("Frequency")
#plt.show()


# In[20]:


for var in ["price", "sqm", "num_rooms", "num_bathrooms"]:
    plt.figure(figsize=(10, 3))
    plt.boxplot(df_eda[var].dropna(), vert=False)
    plt.title(var)
    #plt.show()


# In[21]:


type_freq = df_eda["property_type"].value_counts()
display(type_freq)
type_freq.plot(kind="bar", figsize=(12, 5))
plt.title("Distribution by property type")
#plt.show()


# In[22]:


region_freq = df_eda["region"].value_counts()
display(region_freq)
region_freq.plot(kind="bar", figsize=(12, 5))
plt.title("Distribution by region (CCAA)")
#plt.show()


# In[23]:


province_freq = df_eda["province"].value_counts().head(20)
display(province_freq)
province_freq.plot(kind="bar", figsize=(12, 5))
plt.title("Top 20 provinces by number of listings")
#plt.show()


# In[24]:


binary_vars = ["pool_flag", "beach_flag", "cancellation", "breakfast"]
for var in binary_vars:
    print("\n" + "=" * 60)
    print(var.upper())
    print("=" * 60)
    display(df_eda[var].value_counts(dropna=False))


# In[25]:


for var in numeric_analysis_vars:
    print("\n", var)
    print("Skewness:", round(df_eda[var].skew(), 2))
    print("Kurtosis:", round(df_eda[var].kurtosis(), 2))


# ### Univariate analysis conclusions
# 
# - `price` is detected as right-skewed with lots of extreme values; the log transform substantially reduces this asymmetry.
# - `sqm` shows the same pattern and is a candidate for a log transform as well.
# - `stars` is unusable as a continuous feature given ~91% missingness; therefore we will translatee this variable into a boolean (has stars/ does not have stars) to make this variable more meaningful.
# - Skewness/kurtosis confirm that tree-based methods (Isolation Forest) are relatively robust to these distributions, but LOF (distance-based) will benefit from log transforms and scaling.

# ## 4. Geographic Analysis

# In[26]:


price_by_region = (
    df_eda.groupby("region")["price"]
    .agg(["count", "mean", "median", "min", "max"])
    .sort_values("mean", ascending=False)
)
display(price_by_region)


# In[27]:


plt.figure(figsize=(12, 6))
price_by_region["mean"].sort_values().plot(kind="barh")
plt.title("Mean price by region")
plt.xlabel("Mean price (EUR)")
#plt.show()


# In[28]:


plt.figure(figsize=(12, 6))
price_by_region["median"].sort_values().plot(kind="barh", color="orange")
plt.title("Median price by region")
plt.xlabel("Median price (EUR)")
#plt.show()


# In[29]:


price_by_province = df_eda.groupby("province")["price"].agg(["count", "mean", "median"])
display(price_by_province.sort_values("mean", ascending=False).head(20))


# In[30]:


display(price_by_province.sort_values("median", ascending=False).head(20))
display(price_by_province.sort_values("median", ascending=True).head(20))


# In[31]:


province_freq.plot(kind="bar", figsize=(12, 5))
plt.title("Top 20 provinces by number of listings")
#plt.show()


# ### Geographic analysis conclusions
# 
# - Listings are geographically concentrated: Andalusia, the Canary Islands, Catalonia and Valencia dominate the volume of listings.
# - Mean and median price diverge notably by region, confirming that geography must be accounted for (e.g. via `price_relative_to_province`, Section 7) so that a naturally expensive region isn't detcted as "anomalous" by the models.
# - **Geographic bias risk**: regions with very few listings could have unstable local statistics (mean/median/z-scores), this will be flagged again in Section 11 (Risks).
# 
# **Note**: at this stage `region` still contains "Canarias" and "Islas Canarias" asseparate categories (same entity, inconsistent labeling). This is intentionally left unresolved here and fixed in Section 6, where all cleaning steps are grouped together.

# ## 5. Bivariate / Multivariate Analysis

# In[32]:


df_eda["price_per_sqm"] = df_eda["price"] / df_eda["sqm"]
df_eda["price_per_sqm"].describe()


# In[33]:


display(
    df_eda[["name", "property_type", "province", "sqm", "price", "price_per_sqm"]]
    .sort_values("price_per_sqm", ascending=False).head(20)
)


# In[34]:


corr_vars = [
    "price", "price_without_discount", "discount_amount",
    "sqm", "num_rooms", "num_bathrooms", "stars", "price_per_sqm"
]

corr_pearson = df_eda[corr_vars].corr(method="pearson")
corr_spearman = df_eda[corr_vars].corr(method="spearman")

print("Pearson correlation:")
display(corr_pearson)

print("Spearman correlation:")
display(corr_spearman)


# In[35]:


# Verify the price / price_without_discount / discount_amount relationship directly,
has_discount = df_eda["discount_amount"] > 0
print(f"Listings with a nonzero discount: {has_discount.sum()} ({has_discount.mean()*100:.2f}%)")
print(f"Listings with price_without_discount == 0: {(df_eda['price_without_discount'] == 0).sum()} "
      f"({(df_eda['price_without_discount'] == 0).mean()*100:.2f}%)")


# In[36]:


fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.heatmap(corr_pearson, annot=True, cmap="coolwarm", center=0, ax=axes[0])
axes[0].set_title("Pearson correlation")

sns.heatmap(corr_spearman, annot=True, cmap="coolwarm", center=0, ax=axes[1])
axes[1].set_title("Spearman correlation (rank, robust to non-linearity/outliers)")

plt.tight_layout()
#plt.show()


# In[37]:


plt.figure(figsize=(10, 6))
plt.scatter(df_eda["sqm"], df_eda["price"], alpha=0.1)
plt.xlabel("Sqm")
plt.ylabel("Price (EUR)")
plt.title("Price vs. sqm")
#plt.show()


# In[38]:


plt.figure(figsize=(10, 6))
plt.scatter(df_eda["sqm"], np.log1p(df_eda["price"]), alpha=0.1)
plt.xlabel("Sqm")
plt.ylabel("log(price)")
plt.title("log(price) vs. sqm")
#plt.show()


# In[39]:


plt.figure(figsize=(10, 6))
plt.scatter(df_eda["sqm"], df_eda["price_per_sqm"], alpha=0.1)
plt.ylim(0, df_eda["price_per_sqm"].quantile(0.99))
plt.xlabel("Sqm")
plt.ylabel("Price per sqm (EUR)")
plt.title("Price per sqm vs. sqm")
#plt.show()


# In[40]:


plt.figure(figsize=(10, 6))
plt.scatter(df_eda["sqm"], df_eda["num_rooms"], alpha=0.1)
plt.xlabel("Sqm")
plt.ylabel("Number of rooms")
plt.title("Number of rooms vs. sqm")
#plt.show()


# In[41]:


price_by_type = (
    df_eda.groupby("property_type")["price"]
    .agg(["count", "mean", "median"])
    .sort_values("median", ascending=False)
)
display(price_by_type)

plt.figure(figsize=(14, 6))
sns.boxplot(data=df_eda, x="property_type", y="price")
plt.xticks(rotation=90)
plt.ylim(0, 5000)
plt.title("Price distribution by property type")
plt.close()
#plt.show()


# In[42]:


for flag in ["pool_flag", "beach_flag"]:
    plot_data = df_eda[[flag, "price"]].dropna()
    print(plot_data.groupby(flag)["price"].median())
    if plot_data.empty or plot_data[flag].nunique() == 0:
        continue
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=plot_data, x=flag, y="price")
    plt.ylim(0, 5000)
    plt.title(f"Price by {flag}")
    plt.close()
    #plt.show()


# In[43]:


print(df_eda.groupby("stars")["price"].median())

plt.figure(figsize=(10, 6))
sns.violinplot(
    data=df_eda.dropna(subset=["stars"]),
    x="stars", y="price", cut=0
)
plt.ylim(0, 5000)
plt.title("Price distribution by star rating (violin plot)")
#plt.show()


# In[44]:


df_eda["price_per_sqm"].describe(percentiles=[0.90, 0.95, 0.99, 0.999])


# ### Bivariate / multivariate analysis conclusions
# 
# - **Price vs. sqm**: positive relationship, but with heavy dispersion for similar surface areas — expected for a mixed-type tourist market.
# - **Pearson vs. Spearman**: Spearman correlations are higher than Pearson for `price`–`sqm` and `price`–`num_rooms`, confirming a *monotonic but non-linear* relationship. This matters for LOF, which is sensitive to the geometry implied by the raw scale, with the help of log transforms (Section 8) we can linearize these relationships.
# - **`price_per_sqm`** is the single most extreme, heavy-tailed variable in the set and is expected to be a strong anomaly signal.
# - **Property type, pool, beach and star rating** all shift the price distribution meaningfully, which are exactly the segment-level effects that`price_relative_to_type` / `price_relative_to_province` (Section 7) are designed to neutralize before anomaly detection.
# 
# ### Correlation implications for modeling
# 
# - **`price` vs `price_without_discount`**: the verification cell above shows `price_without_discount` is 0 for the large majority of listings, while only a small minority currently have an active discount. Pearson/Spearman correlations between the two are therefore not very informative on their own (close to 0), but the relationship is still definitional wherever a discount exists: keeping both as model inputs risks near-duplicate/redundant features. Therefore, only `price` (or `log_price`) plus `discount_pct` should enter the final model matrix.
# - **`price` vs `discount_amount`**: with so few nonzero discounts, this correlation is driven by a small subgroup. It is worthy to check discounted listings separately rather than trusting a dataset-wide correlation coefficient here (e.g. large absolute discounts concentrated on already-expensive listings).
# - **`price` vs `price_per_sqm`**: may be redundant, but `price_per_sqm` normalizes for size. We will keep `price_per_sqm` and treat raw `price` as a secondary variable model feature.
# - **`num_rooms` vs `sqm`**: moderate positive correlation; their *ratio* (`sqm_per_room`, Section 7) is more informative for anomaly detection than either of the variables by themselves.
# - **`num_rooms` vs `num_bathrooms`**: correlated as expected (bigger properties have more of both); `baths_per_room` captures deviations from that norm.

# ## 6. Data Cleaning

# In[45]:


df_clean = df_eda.copy()
print(df_clean.shape)


# In[46]:


rows_before = len(df_clean)
df_clean = df_clean.drop_duplicates()
rows_after = len(df_clean)

print(f"Rows removed: {rows_before - rows_after}")
print(f"New shape: {df_clean.shape}")


# **Duplicate removal**: 716 exact duplicate records (detected during the quality audit) were removed. Since these rows were identical across every variable, they were treated as artifacts of the scraping process rather than genuine repeate observations.

# In[47]:


df_clean["region"] = df_clean["region"].replace({"Canarias": "Canary Islands", "Islas Canarias": "Canary Islands"})
df_clean["region"].value_counts()


# **Category homogenization**: the `region` variable contained two labels for the same entity ("Canarias" / "Islas Canarias"). These are unified into a single category.

# In[48]:


constant_vars = ["capacity", "current_date", "checkin_date", "checkout_date"]
df_clean = df_clean.drop(columns=constant_vars)
df_clean.shape


# **Removal of constant variables**: `capacity`, `current_date`, `checkin_date` and `checkout_date` take a single value across the entire dataset. A feature with zero variance carries no discriminative power for either Isolation Forest or LOF and adds only noise/cost.

# In[49]:


(df_clean.isnull().mean().sort_values(ascending=False) * 100)


# In[50]:


df_clean["breakfast"] = (
    df_clean["breakfast"]
    .fillna("")
    .astype(str)
    .str.strip()
    .map({"1": 1})
    .fillna(0)
    .astype(int)
)
df_clean["breakfast"].value_counts(dropna=False)


# **`breakfast` treatment**: the raw variable only ever appeared as blank or `"1"`,
# with 48 nulls tied to otherwise-incomplete records. It was recoded as an explicit
# binary flag (1 = breakfast included, 0 = not stated / not included), which is directly
# usable by both models without further encoding.

# In[51]:


df_clean.isnull().sum().sort_values(ascending=False)


# In[52]:


rows_before = df_clean.shape[0]
df_clean = df_clean.dropna(subset=["price", "num_rooms", "num_bathrooms"])
rows_after = df_clean.shape[0]

print(f"Rows removed: {rows_before - rows_after}")
print(f"New shape: {df_clean.shape}")


# In[53]:


# Remove records with no geographic information
rows_before = df_clean.shape[0]
df_clean = df_clean.dropna(subset=["region", "province"])
rows_after = df_clean.shape[0]

print(f"Rows removed: {rows_before - rows_after}")
print(f"New shape: {df_clean.shape}")


# In[54]:


# 1. Delete data errors (sqm < 10), keeping null
rows_before = df_clean.shape[0]
df_clean = df_clean[(df_clean['sqm'] >= 10) | (df_clean['sqm'].isna())]
rows_after_filter = df_clean.shape[0]

print(f"Data errors (sqm < 10) deleted: {rows_before - rows_after_filter}")

# 2. Imput values for null (median)
df_clean['sqm'] = df_clean.groupby(['province', 'property_type'])['sqm'].transform(lambda x: x.fillna(x.median()))

# 3. Backup input (for null medians)
df_clean['sqm'] = df_clean.groupby(['property_type'])['sqm'].transform(lambda x: x.fillna(x.median()))

# 4. Delete rest of nulls
df_clean = df_clean.dropna(subset=['sqm'])
rows_final = df_clean.shape[0]

print(f"Rest of nulls deleted: {rows_after_filter - rows_final}")
print(f"New shape: {df_clean.shape}")


# In[55]:


display(df_clean.head())


# ### Cleaning conclusions
# 
# - **Duplicates**: 716 exact duplicates removed.
# - **Homogenization**: geographic category inconsistencies (region naming) resolved.
# - **Constants**: zero-variance columns dropped.
# - **Missing critical fields**: rows missing `price`, `num_rooms`, `num_bathrooms`, `region` or `province` were dropped, since these are core anomaly-detection features that cannot be safely imputed without risking synthetic, non-representative anomalies.
# - `stars` is deliberately **not** dropped or imputed here (see Section 7/8), where it will be reframed as a `has_stars` binary flag instead of discarding the information.
# - 396 rows deleted due to square meter inconsistencies (less than 10 square meters and nulls after input values)

# ## 7. Feature Engineering

# In[56]:


# Before building room/bathroom ratios, check whether num_rooms == 0 / num_bathrooms == 0
print("=" * 60)
print("num_rooms == 0 BY PROPERTY TYPE")
print("=" * 60)
display(
    pd.crosstab(df_clean["property_type"], df_clean["num_rooms"] == 0, normalize="index")
    .round(2).sort_values(True, ascending=False)
)

print("=" * 60)
print("num_bathrooms == 0 BY PROPERTY TYPE")
print("=" * 60)
display(
    pd.crosstab(df_clean["property_type"], df_clean["num_bathrooms"] == 0, normalize="index")
    .round(2).sort_values(True, ascending=False)
)


# Both `num_rooms == 0` and `num_bathrooms == 0` concentrate in `Hostal o pensión`, `Habitación en casa particular`, `Hotel` and similar single-room-booking types. This is consistent with `num_rooms`/`num_bathrooms` counting *additional* rooms/bathrooms beyond the one being booked, rather than an error (a single hostel room legitimately has "0 extra rooms"). Treating these as missing could silently drop ~19% of the dataset when building ratio features below. Instead, we will use `np.maximum(x, 1)`, which treats 0 as 1 (the unit itself) without discarding the record.

# In[57]:


df_clean["price_per_sqm"] = df_clean["price"] / df_clean["sqm"]


# In[58]:


df_clean["discount_pct"] = np.where(
    df_clean["price"] > 0,
    df_clean["discount_amount"] / df_clean["price"],
    0
)


# In[59]:


df_clean["sqm_per_room"] = df_clean["sqm"] / np.maximum(df_clean["num_rooms"], 1)


# In[60]:


df_clean["baths_per_room"] = df_clean["num_bathrooms"] / np.maximum(df_clean["num_rooms"], 1)


# In[61]:


df_clean["price_per_room"] = df_clean["price"] / np.maximum(df_clean["num_rooms"], 1)


# In[62]:


# Price per bathroom
df_clean["price_per_bath"] = df_clean["price"] / np.maximum(df_clean["num_bathrooms"], 1)

# Rooms per sqm
df_clean["rooms_per_sqm"] = np.where(
    df_clean["sqm"] > 0,
    df_clean["num_rooms"] / df_clean["sqm"],
    np.nan
)

# How many standard deviations away from the mean price of its own property type / province.
df_clean["price_zscore_by_type"] = (
    df_clean.groupby("property_type")["price"]
    .transform(lambda s: (s - s.mean()) / s.std())
)

df_clean["price_zscore_by_province"] = (
    df_clean.groupby("province")["price"]
    .transform(lambda s: (s - s.mean()) / s.std())
)

# Price relative to the median of its own segment
df_clean["price_relative_to_type"] = (
    df_clean["price"]
    / df_clean.groupby("property_type")["price"].transform("median")
)

df_clean["price_relative_to_province"] = (
    df_clean["price"]
    / df_clean.groupby("province")["price"].transform("median")
)

# has_stars: 1 if it has a star rating, 0 if it does not
df_clean["has_stars"] = df_clean["stars"].notna().astype(int)

new_engineered_vars = [
    "price_per_bath", "rooms_per_sqm",
    "price_zscore_by_type", "price_zscore_by_province",
    "price_relative_to_type", "price_relative_to_province",
    "has_stars"
]
display(df_clean[new_engineered_vars].describe())


# In[63]:


derived_vars = [
    "price_per_sqm", "discount_pct", "sqm_per_room",
    "baths_per_room", "price_per_room"
]
display(df_clean[derived_vars].describe())


# In[64]:


display(
    df_clean[["name", "property_type", "price", "num_rooms", "price_per_room"]]
    .sort_values("price_per_room", ascending=False).head(20)
)


# In[65]:


display(
    df_clean[["name", "property_type", "sqm", "num_rooms", "sqm_per_room"]]
    .sort_values("sqm_per_room", ascending=False).head(20)
)


# ### Feature engineering conclusions
# 
# **Original derived variables** `price_per_sqm`, `discount_pct`, `sqm_per_room`, `baths_per_room`, `price_per_room`: capture relationships between raw attributes that are more informative in combination rather than in isolation.
# 
# **New variables added in this review**:
# 
# | Variable | Use for anomaly detection |
# |---|---|
# | `price_per_bath` | Identifies properties whose price is unusually high or low relative to the number of bathrooms, regardless of the total number of rooms |
# | `rooms_per_sqm` | Captures unusually overcrowded properties by measuring how many rooms are packed into the available living space |
# | `price_zscore_by_type` / `price_zscore_by_province` |  Highlights listings that are unusually expensive or cheap compared with similar properties in the same property type or province |
# | `price_relative_to_type` / `price_relative_to_province` | Measures how a property's price compares to its local market benchmark, making it easier to spot listings that deviate from typical market behavior |
# | `has_stars` | Recovers signal from a 91%-missing variable instead of  directly discarding `stars` |
# 
# The two most valuable additions for the modeling stage are the **segment-relative variables** (`price_zscore_by_*`, `price_relative_to_*`): raw price and raw `price_per_sqm` conflate "this listing is unusual" with "this region/type is naturally expensive." Segment-relative features are able to separate those two effects, which is essential since Isolation Forest and LOF have no built-in concept of subgroup context.

# ## 8. Variable Treatment and Transformations
# 
# We have to make some decisions on how each problematic variable should enter the models.

# In[66]:


df_clean["log_price"] = np.log1p(df_clean["price"])
df_clean["log_sqm"] = np.log1p(df_clean["sqm"])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
df_clean["log_price"].hist(bins=50, ax=axes[0])
axes[0].set_title("log(price)")
df_clean["log_sqm"].hist(bins=50, ax=axes[1])
axes[1].set_title("log(sqm)")
#plt.show()


# - **`price`** → we are going to apply `log1p`. Strong right skew (which was already confirmed by skewness/kurtosis in Section 3).
# - **`sqm`** → we are applying `log1p` for the same reason (reduces the leverage of a few extremely large properties (villas, fincas) on distance-based methods).
# - **`discount_amount`** → we will not use the raw amount, but`discount_pct` instead (already engineered in Section 7). A percentage is comparable across price ranges, while the absolute discount is trivially correlated with `price` itself and would duplicate that signal rather than add new information.
# - **`stars`** → we will not feed the (91%-missing) numeric variable into the models. Moreover, we computed `has_stars` as the model input.

# ## 9. Preliminary Anomaly Detection (Business Rules)
# 
# Before any unsupervised model is trained, simple domain rules are used to get a first read on which records may look suspicious. This also acts as a sanity check for the later Isolation Forest / LOF results.

# In[67]:


flags = pd.DataFrame(index=df_clean.index)

# Extreme price
q1, q3 = df_clean["price"].quantile([0.25, 0.75])
iqr = q3 - q1
flags["extreme_price_high"] = df_clean["price"] > (q3 + 3 * iqr)

# Extremely low price
flags["extreme_price_low"] = df_clean["price"] < df_clean["price"].quantile(0.01)

# Suspected placeholder sqm values
flags["suspected_placeholder_sqm"] = df_clean["sqm"] <= 2

# Extreme price per sqm
q1m, q3m = df_clean["price_per_sqm"].quantile([0.25, 0.75])
iqrm = q3m - q1m
flags["extreme_price_per_sqm"] = df_clean["price_per_sqm"] > (q3m + 3 * iqrm)

# Many rooms for very little space
flags["overcrowded_layout"] = df_clean["rooms_per_sqm"] > df_clean["rooms_per_sqm"].quantile(0.995)

# Many bathrooms relative to rooms
flags["excess_bathrooms"] = df_clean["baths_per_room"] > df_clean["baths_per_room"].quantile(0.995)

# Tiny villas / houses
large_property_types = ["Villa", "Casa o chalet"]
flags["tiny_villa"] = (
    df_clean["property_type"].isin(large_property_types) & (df_clean["sqm"] < 30)
)

# Extremely large apartments/studios
small_property_types = ["Apartamento", "Estudio"]
flags["huge_apartment"] = (
    df_clean["property_type"].isin(small_property_types) & (df_clean["sqm"] > 300)
)

flags["any_flag"] = flags.any(axis=1)

print("=" * 60)
print("BUSINESS-RULE FLAG SUMMARY")
print("=" * 60)
display(flags.drop(columns="any_flag").sum().to_frame("n_records"))
print(f"\nTotal records flagged by at least one rule: {flags['any_flag'].sum()} "
      f"({flags['any_flag'].mean() * 100:.2f}%)")


# In[68]:


df_clean = df_clean.join(flags)

display(
    df_clean[df_clean["any_flag"]]
    [["name", "property_type", "province", "sqm", "price", "price_per_sqm"]]
    .head(20)
)


# **Note**: `property_type` category values above (`"Villa"`, `"Casa o chalet"`, `"Apartamento"`, `"Estudio"`) reflect the original Spanish labels present in the `property_type` column itself (only column *names* were translated in this notebook).
# 
# - **Fix applied to `extreme_price_low`**: the original IQR-based rule (`price < q1 - 3*iqr`) never fired, because `price` is skewed enough that `q1 - 3*iqr` is negative (no price can be below 0). We have changed the rule to detect flag prices that fall within the lowest 1% of all prices (`price < price.quantile(0.01)`), which correctly identifies very low-price outliers.
# 
# - **`suspected_placeholder_sqm` is reported separately from `tiny_villa`** : several of the smallest `sqm` values in Section 2 (e.g. `sqm = 1`) sit next to prices in the thousands of euros, which looks like a scraping placeholder. We are going to keep it as its own flag to avoid mislabeling a an error as a real anomalous villa.
# 
# These rule-based flags are **not** used as model features (that would leak the target concept into training), we are just utilizing them as a demostration of some of the outliers presented in the dataset.

# ## 10. Dimensionality Reduction Preview (PCA)
# 
# A quick 2D PCA projection of the (scaled) numeric feature space, used just as an exploratory check for structure and obvious outliers before committing to Isolation Forest / LOF. This is only utilized as a descriptive model.

# In[69]:


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

pca_features = [
    "log_price", "log_sqm", "num_rooms", "num_bathrooms",
    "price_per_sqm", "sqm_per_room", "baths_per_room",
    "price_relative_to_type", "price_relative_to_province"
]

pca_df = df_clean[pca_features].dropna()
X_scaled = StandardScaler().fit_transform(pca_df)

pca = PCA(n_components=2, random_state=42)
components = pca.fit_transform(X_scaled)

plt.figure(figsize=(9, 7))
plt.scatter(components[:, 0], components[:, 1], alpha=0.15, s=8)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var.)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var.)")
plt.title("2D PCA projection of candidate anomaly-detection features")
#plt.show()

print("Explained variance ratio:", pca.explained_variance_ratio_)


# Some outliers are already shown in the graph, these may be some of the candidats that we are going to obtain later while using our trained models. The first two principal components capture 57.0% of the variance in the selected features. Most properties form a compact cluster, while a small number of observations are located far from the center of the distribution. These isolated points may represent the outliers. The PCA projection suggests that anomaly-detection methods are likely to identify a limited set of properties with markedly different characteristics from the rest of the dataset.

# ## 11. Risks and Limitations of the Dataset
# 
# These are some of the risks and limitations that are present in the dataset:
# 
# - **Information leakage**: `price_without_discount`, `discount_amount` and `price` are algebraically related. Using more than one of them as independent model features may risk the perfomance of the models. Only `price` (or`log_price`) and `discount_pct` should enter the final matrix (Section 12).
# - **Redundant / highly correlated variables**: `price_per_sqm`, `price_per_room` and `price_per_bath` are all transformations of the same underlying `price` signal. Including all of them rises the influence of price-related information relative to spatial/layout information in the methods that we are using like LOF.
# - **Geographic bias**: geographic statistics (such as means, z-scores) may be unstable for provinces with very few listings. Extreme `price_zscore_by_province` values may appear reflecting sample size and not anomalies. We should consider a minimum-count threshold before showing province-level outliers.
# - **Single-platform bias**: all data comes from one scraping source: Booking. Features such as pricing behavior, listing structure, and property mix on this platform may not generalize to other websites.
# - **Temporal bias**: the data represents a single date (`current_date`). Seasonal pricing effects (summer fees) may affect the results which will be variant depending on the scraped date when the data was obtained.
# - **Real atypical records vs. data errors**: not every extreme value is a data-entry error (a 500 sqm villa is plausible, while a listing with 1 sqm is almost certainly not). We produced section 9's business rules to show these records rather than silently dropping them.

# In[70]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd

# Solo variables numéricas continuas candidatas (sin dummies, sin identificadores)
vif_cols = ['log_price', 'log_sqm', 'num_rooms', 'num_bathrooms',
            'price_per_sqm', 'price_per_room', 'price_per_bath',
            'sqm_per_room', 'rooms_per_sqm', 'baths_per_room',
            'discount_pct', 'price_relative_to_province']

X_vif = df_clean[vif_cols].dropna()

vif_data = pd.DataFrame({
    'variable': X_vif.columns,
    'VIF': [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
}).sort_values('VIF', ascending=False)

vif_data


# **Multicollinearity check (VIF)**
# 
# With VIF we can confirm multicollinearity among price/size features. `log_sqm` (183), `log_price` (150), `num_rooms` (31), `baths_per_room` (26) and `num_bathrooms` (24) show extreme VIF, expected given that several engineered ratios (`price_per_bath`, `price_per_room`, `sqm_per_room`, `baths_per_room`) are algebraic functions of these base variables. `price_per_bath` (13.2) and `price_per_room` (10.7) also exceed the VIF > 10 threshold on their own.
# 
# We retain the base transformed variables (`log_price`, `log_sqm`, `num_rooms`, `num_bathrooms`) and the lowest derived ratio (`price_per_sqm`, VIF 2.1), while dropping `price_per_room` and `price_per_bath` from the final model matrix (Section 12.1) to reduce redundancy without losing discriminative signal.

# ## 12. Final Dataset Preparation for Isolation Forest / LOF
# 
# In this section we build the model-ready dataframe: selected variables, dropped variables, transformations, encoding and scaling.

# ### 12.1 Variable selection
# 
# **Dropped (identifiers / no discriminative value / leakage risk):** `location_id`, `room_name`, `municipality`, `locality`, `price_without_discount`, `discount_amount` (superseded by `discount_pct`), `stars` (superseded by `has_stars`), the raw business-rule flag columns from Section 9 (kept for validation, excluded from model input to avoid leakage), `price_per_room`, `price_per_bath` (multicollinearity confirmed by VIF, redundant with `log_price`, `log_sqm`, `num_rooms`, `num_bathrooms`)
# 
# **Kept: identifiers (only for traceability, not to train the models):** `id`, `room_id`, `name`,
# 
# **Kept: numeric model features:** `log_price`, `log_sqm`, `num_rooms`, `num_bathrooms`, `price_per_sqm`, `sqm_per_room`, `rooms_per_sqm`, `baths_per_room`, `discount_pct`, `price_relative_to_type`, `price_relative_to_province`, `has_stars`, `pool_flag`, `beach_flag`, `breakfast`, `cancellation`.
# 
# **Kept: categorical model features (to encode):**
# `property_type`, `region`. `province` is excluded from direct encoding due to high cardinality (its effect is
# already captured indirectly via `price_relative_to_province`).
# 
# ### 12.2 Scaling choice: RobustScaler vs. StandardScaler
# 
# **We are selecting RobustScaler over StandardScaler:**
# On one hand, StandardScaler centers on the mean and scales by standard deviation, both of which are themselves distorted by the heavy-tailed variables identified throughout this notebook (like `price`, `price_per_sqm`, etc). Moreover, a few extreme values would dominate the scaling factor and compress the bulk of "normal" observations into a narrow range, which is the opposite of what LOF requires to work well. On the other hand, RobustScaler uses the median and IQR, which are insensitive to the very outliers we ultimately want the models to detect, so the scaling doesn't get distorted.

# In[71]:


from sklearn.preprocessing import RobustScaler

numeric_model_features = [
    "log_price", "log_sqm", "num_rooms", "num_bathrooms",
    "price_per_sqm", "sqm_per_room", "rooms_per_sqm", "baths_per_room", "discount_pct",
    "price_relative_to_type", "price_relative_to_province", "has_stars",
    "pool_flag", "beach_flag", "breakfast", "cancellation"
]

categorical_model_features = ["property_type", "region"]

id_columns = ["id", "room_id", "name"]  # kept only for traceability, not modeling

model_input_cols = numeric_model_features + categorical_model_features
df_model = df_clean[id_columns + model_input_cols].copy()

print("Rows before dropping remaining NaNs in model features:", len(df_model))
missing_model_values = df_model[numeric_model_features].isna().sum()
binary_model_features = {
    "has_stars", "pool_flag", "beach_flag", "breakfast", "cancellation"
}
for column in numeric_model_features:
    if not missing_model_values[column]:
        continue
    fill_value = 0 if column in binary_model_features else df_model[column].median()
    if pd.isna(fill_value):
        fill_value = 0
    df_model[column] = df_model[column].fillna(fill_value)

print("Missing values imputed in model features:")
print(missing_model_values[missing_model_values > 0])
print("Rows after imputing model features:", len(df_model))


# In[72]:


# One-hot encode low-cardinality categoricals
df_encoded = pd.get_dummies(
    df_model,
    columns=categorical_model_features,
    prefix=categorical_model_features,
    drop_first=False
)

encoded_feature_cols = [
    c for c in df_encoded.columns if c not in id_columns
]

scaler = RobustScaler()
X_scaled = scaler.fit_transform(df_encoded[encoded_feature_cols])

df_final = pd.DataFrame(
    X_scaled,
    columns=encoded_feature_cols,
    index=df_encoded.index
)

# Reattach traceability identifiers (not used as model input)
df_final = pd.concat([df_encoded[id_columns], df_final], axis=1)

print(df_final.shape)
display(df_final.head())


# In[73 / final]:
OUTPUT_PROCESSED = ROOT_DIR / "data" / "processed"
OUTPUT_PROCESSED.mkdir(parents=True, exist_ok=True)

# Guardar datos sin escalar (con precio, sqm, features y business rules para validación posterior)
raw_features_cols = id_columns + [
    "price", "sqm", "property_type", "region", "province", 
    "num_rooms", "num_bathrooms", "log_price", "log_sqm",
    "price_per_sqm", "discount_pct", "pool_flag", "beach_flag", "breakfast",
    "price_relative_to_type", "price_relative_to_province",
    "extreme_price_high", "extreme_price_low", "suspected_placeholder_sqm",
    "extreme_price_per_sqm", "overcrowded_layout", "excess_bathrooms", 
    "tiny_villa", "huge_apartment", "any_flag"
]

# Asegurar que todas las columnas existen
raw_features_cols = [c for c in raw_features_cols if c in df_clean.columns]
df_raw_features = df_clean[raw_features_cols].copy()
df_raw_features.insert(0, "extraction_date", extraction_date)
df_raw_features.insert(1, "lookahead_period", lookahead_period)
df_raw_features.insert(2, "checkin_date", checkin_date)
df_raw_features.insert(3, "checkout_date", checkout_date)

output_raw_features = OUTPUT_PROCESSED / "dataset_raw_features.csv"
df_raw_features.to_csv(output_raw_features, sep=";", index=False, encoding="utf-8")
print(f"Saved (raw features + business rules): {output_raw_features}")

# Guardar datos escalados para el modelo
output_file = OUTPUT_PROCESSED / "dataset_anomaly_detection_ready.csv"
df_final.to_csv(output_file, sep=";", index=False, encoding="utf-8")
print(f"Saved (scaled for ML): {output_file}")


# ### Final dataset summary
# 
# - **Variables selected**: 3 identifiers + 17 numeric (transformed/engineered) + 2 one-hot encoded categoricals (`property_type`, `region`).
# - **Variables eliminated**: 2 identifiers, redundant price fields, raw `stars`, constant columns, high-cardinality `province`/`municipality`/`locality`.
# - **Transformations**: `log1p` on `price`/`sqm`; ratio features for layout (`sqm_per_room`, `rooms_per_sqm`, `baths_per_room`); segment-relative price features to control for type/region effects.
# - **Encoding**: one-hot for `property_type` and `region` (both low/medium cardinality).
# - **Scaling**: `RobustScaler`, chosen specifically because it does not let the extreme values we want Isolation Forest / LOF to detect distort the scaling itself.
# - **Output**: `df_final` / `dataset_anomaly_detection_ready.csv`, indexed with traceable `id`/`room_id`/`name` columns kept alongside (not fed to the models), so that any anomaly a model flags can be traced back to a real listing for characterization.
# 
# **We will use this dataset as the input for the next notebook: Isolation Forest and LOF
# training.**
