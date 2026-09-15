#!/usr/bin/env python
# coding: utf-8

# # Notebook 1: Model Training for Anomaly Detection
# 
# **Pipeline:** Isolation Forest (global anomalies) and Local Outlier Factor (local anomalies) on the Spanish tourist accommodation market.
# 
# **Scope of this notebook:** train and configure both models, run a sensitivity analysis of their hyperparameters and save the resulting scores/flags/config.
# 

# ## 0. Imports and configuration

# In[73]:


import sys
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import sklearn

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

try:
    display
except NameError:
    display = print

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)
plt.style.use("default")

RANDOM_STATE = 42

OUTPUT_DIR = Path("outputs")
MODELS_DIR = Path("models")
OUTPUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

print("Libraries imported successfully.")


# ## 1. Dataset loading and verification
# 
# Load the dataset already prepared for modeling and check its integrity (shape,
# absence of `NaN`/`inf`) before doing anything else.
# 

# In[74]:
import os
import pandas as pd
from pathlib import Path

# Resolver la raíz del proyecto (sube 3 niveles: src -> pipeline -> raíz)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"

data_file = DATA_PROCESSED_DIR / "dataset_anomaly_detection_ready.csv"
if not data_file.exists():
    raise FileNotFoundError(f"No se encontró el dataset procesado en: {data_file}")

# Cargar el dataset con el separador ';'
df = pd.read_csv(data_file, sep=";")

print(f"Rows:    {df.shape[0]:,}")
print(f"Columns: {df.shape[1]}")
print(df.head())


# In[75]:


nan_counts = df.isna().sum()
n_nan_total = nan_counts.sum()
print(f"Total NaN values: {n_nan_total}")
if n_nan_total > 0:
    display(nan_counts[nan_counts > 0].to_frame("n_nan"))

numeric_df = df.select_dtypes(include=[np.number])
inf_counts = np.isinf(numeric_df).sum()
n_inf_total = inf_counts.sum()
print(f"Total inf/-inf values: {n_inf_total}")
if n_inf_total > 0:
    display(inf_counts[inf_counts > 0].to_frame("n_inf"))

assert n_nan_total == 0, "The dataset contains NaN values: review preprocessing before modeling."
assert n_inf_total == 0, "The dataset contains infinite values: review preprocessing before modeling."


# In[76]:


id_cols = ['id', 'room_id', 'name']
missing_ids = [c for c in id_cols if c not in df.columns]
assert not missing_ids, f"Missing identifier columns: {missing_ids}"

ids_df = df[id_cols].copy()
feature_cols = [c for c in df.columns if c not in id_cols]

assert not (set(id_cols) & set(feature_cols)), "Identifiers must not be part of the modeling variables."

print(f"Identifiers separated: {id_cols}")
print(f"Number of candidate modeling variables: {len(feature_cols)}")


# In[77]:


print("Variables that enter the models (X):\n")
for i, c in enumerate(feature_cols, 1):
    print(f"{i:>2}. {c}")


# ## 2. Model input preparation
# 
# Build `X` using only the modeling variables (no identifiers), keeping `ids_df` separate to recover results later. **No additional preprocessing is applied**: the dataset already arrives scaled with `RobustScaler` from the data-prep pipeline.
# 
# Note on encoding: categorical variables such as `property_type` and `region` were one-hot encoded upstream. This matters specifically for LOF, since it computes distances directly on the feature vectors. the resulting binary dummy columns participate in the distance structure exactly like any numeric feature, so inconsistent categories (see the `region` check above) would distort local density estimates.
# 

# In[78]:


X = df[feature_cols].copy()
print(f"Final shape of X: {X.shape}")
display(X.head())


# In[79]:


desc = X.describe().T[['mean', 'std', 'min', '50%', 'max']].rename(columns={'50%': 'median'})
desc['abs_median'] = desc['median'].abs()
display(desc.sort_values('abs_median', ascending=False).head(10))


# Medians close to 0 and bounded dispersion are consistent with variables already scaled with RobustScaler. No additional preprocessing is performed here: the dataset arrives ready for modeling.

# ### Duplicate feature vectors
# 
# LOF is sensitive to zero-distance neighbors (duplicate rows distort local density).
# We check how many rows of `X` are exactly identical before training.
# 

# In[80]:


n_duplicated = X.duplicated().sum()
pct_duplicated = n_duplicated / len(X)
print(f"Duplicated feature vectors: {n_duplicated:,} ({pct_duplicated:.2%})")


# These duplicates are the most likely cause of sklearn's warning ('Duplicate values are leading to incorrect results') when training LOF: when several properties share the exact same feature vector, their distance is 0 and the local-density computation can become unstable for those points. Rows are not removed (they may be legitimately identical after scaling/encoding), but the phenomenon is documented and taken into account when reading LOF results, especially for small n_neighbors.

# ## 3. Isolation Forest: Baseline
# 
# Train a first model with reasonable default parameters as a starting point before
# the sensitivity analysis.
# 

# In[81]:


if_baseline = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
if_baseline.fit(X)

if_pred_baseline = if_baseline.predict(X)              # 1 = normal, -1 = anomaly
if_score_baseline = if_baseline.decision_function(X)   # raw sklearn score (lower = more anomalous)
if_score_baseline_inv = if_score_baseline * -1          # inverted: higher = more anomalous

n_if_anomalies_baseline = (if_pred_baseline == -1).sum()
print(f"Isolation Forest baseline -> anomalies detected: {n_if_anomalies_baseline} "
      f"({n_if_anomalies_baseline / len(X):.2%})")


# ## 4. Isolation Forest: Sensitivity Analysis
# 
# Instead of an exhaustive `GridSearch`, we study the effect of `contamination` and `n_estimators` on the number of anomalies, training time, and score **stability** (a statistical criterion, not a business one).
# 

# In[82]:


contamination_grid = [0.01, 0.03, 0.05, 0.10]
n_estimators_grid = [100, 200, 300, 500]

sensitivity_results = []
for c in contamination_grid:
    for n_est in n_estimators_grid:
        t0 = time.time()
        temp_if = IsolationForest(n_estimators=n_est, contamination=c, random_state=RANDOM_STATE, n_jobs=-1)
        temp_pred = temp_if.fit_predict(X)
        elapsed = time.time() - t0
        sensitivity_results.append({
            'contamination': c, 'n_estimators': n_est,
            'n_anomalies': int((temp_pred == -1).sum()),
            'threshold_offset': temp_if.offset_,
            'fit_time_s': round(elapsed, 2)
        })

sens_df = pd.DataFrame(sensitivity_results)
display(sens_df)


# In[83]:


# Stability: Spearman correlation between scores for different n_estimators (contamination fixed at 0.05)
scores_by_nest = {}
for n_est in n_estimators_grid:
    temp_if = IsolationForest(n_estimators=n_est, contamination=0.05, random_state=RANDOM_STATE, n_jobs=-1)
    temp_if.fit(X)
    scores_by_nest[n_est] = temp_if.decision_function(X)

scores_matrix = pd.DataFrame(scores_by_nest)
corr_matrix = scores_matrix.corr(method='spearman')
print("Spearman correlation between scores for different n_estimators:")
display(corr_matrix.round(4))


# In[84]:


fig, axes = plt.subplots(1, 2, figsize=(14, 5))

pivot = sens_df.pivot(index='contamination', columns='n_estimators', values='n_anomalies')
pivot.plot(marker='o', ax=axes[0])
axes[0].set_title('Number of anomalies vs contamination (by n_estimators)')
axes[0].set_xlabel('Contamination')
axes[0].set_ylabel('Number of anomalies')
axes[0].grid(True, linestyle='--', alpha=0.6)

sens_df.groupby('n_estimators')['fit_time_s'].mean().plot(kind='bar', ax=axes[1], color='steelblue')
axes[1].set_title('Average training time vs n_estimators')
axes[1].set_xlabel('n_estimators')
axes[1].set_ylabel('Time (s)')
axes[1].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
#plt.show()


# **Selection rationale:**
# 
# - `n_estimators`: the Spearman correlation is already very high between 200 and 300
#   trees (≈ 0.99), so **200 would already be enough** for a stable score ranking.
#   Keeping 300 is not justified as "the optimal value", but as a **conservative
#   trade-off**: a small extra computational cost in exchange for extra stability
#   margin, useful for future re-training on slightly different data.
# - `contamination`: **0.05 is not a claim that 5% of listings are truly
#   anomalous.** It is only the **decision threshold** the model uses to turn the
#   continuous score into the binary `if_anomaly` label. It is set to 0.05 by
#   convention (typical 1%–5% range in the absence of ground truth), checking that
#   the number of anomalies does not vary erratically between nearby grid values
#   (see left plot). The continuous score (`if_score`) is what matters for the
#   downstream analysis, not the label itself.
# 

# In[85]:


IF_PARAMS = {
    'n_estimators': 300,
    'contamination': 0.05,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}
print("Final Isolation Forest configuration:")
print(json.dumps(IF_PARAMS, indent=2))


# ## 5. LOF: Baseline
# 
# Train a first LOF model on the same `X` matrix used for Isolation Forest.
# 
# **Note on dimensionality:** LOF relies on distances/density in the full feature
# space of `X`. With `X.shape[1]` ≈ dozens of columns (numeric + one-hot encoded
# categoricals), the "curse of dimensionality" makes distances between points tend
# to look more similar as dimensionality grows, which can flatten local density
# differences and make LOF less discriminative than in low-dimensional settings. No
# dimensionality reduction is applied to compensate for this; it is simply flagged here as a known
# limitation to keep in mind when reading LOF's local anomalies in Notebook 2.
# 

# In[86]:


LOF_PARAMS_BASELINE = {'n_neighbors': 20, 'contamination': 0.05, 'n_jobs': -1}

lof_baseline = LocalOutlierFactor(**LOF_PARAMS_BASELINE)
lof_pred_baseline = lof_baseline.fit_predict(X)
lof_score_baseline = lof_baseline.negative_outlier_factor_ * -1

n_lof_anomalies_baseline = (lof_pred_baseline == -1).sum()
print(f"LOF baseline -> anomalies detected: {n_lof_anomalies_baseline} "
      f"({n_lof_anomalies_baseline / len(X):.2%})")


# ## 6. LOF: Sensitivity Analysis
# 
# We mainly study `n_neighbors` (`contamination` fixed at 0.05 to isolate its
# effect), looking at the number of anomalies, training time, and the **stability**
# of both the anomaly labels (Jaccard similarity) and the continuous scores
# (Spearman correlation).
# 

# In[87]:


n_neighbors_grid = [10, 20, 30, 50, 100]

lof_sensitivity = []
lof_preds_by_n = {}
for n in n_neighbors_grid:
    t0 = time.time()
    temp_lof = LocalOutlierFactor(n_neighbors=n, contamination=0.05, n_jobs=-1)
    temp_pred = temp_lof.fit_predict(X)
    elapsed = time.time() - t0
    lof_preds_by_n[n] = temp_pred
    lof_sensitivity.append({
        'n_neighbors': n,
        'n_anomalies': int((temp_pred == -1).sum()),
        'fit_time_s': round(elapsed, 2)
    })

lof_sens_df = pd.DataFrame(lof_sensitivity)
display(lof_sens_df)


# In[88]:


def jaccard(pred_a, pred_b):
    """Jaccard similarity between two sets of anomaly-flagged (-1) indices."""
    set_a = set(np.where(pred_a == -1)[0])
    set_b = set(np.where(pred_b == -1)[0])
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)

jaccard_matrix = pd.DataFrame(
    [[jaccard(lof_preds_by_n[i], lof_preds_by_n[j]) for j in n_neighbors_grid] for i in n_neighbors_grid],
    index=n_neighbors_grid, columns=n_neighbors_grid
)
print("Jaccard similarity between anomalies detected for each n_neighbors:")
display(jaccard_matrix.round(3))


# Jaccard only compares **binary labels**, which are sensitive to the cut-off point.
# To check whether the anomaly **ranking** is consistent even when some labels
# change, we also compare the continuous `lof_score` across configurations via
# Spearman correlation.
# 

# In[89]:


lof_scores_by_n = {}
for n in n_neighbors_grid:
    temp_lof = LocalOutlierFactor(n_neighbors=n, contamination=0.05, n_jobs=-1)
    temp_lof.fit_predict(X)
    lof_scores_by_n[n] = temp_lof.negative_outlier_factor_ * -1

lof_scores_matrix = pd.DataFrame(lof_scores_by_n)
lof_score_corr = lof_scores_matrix.corr(method='spearman')
print("Spearman correlation between lof_score for different n_neighbors:")
display(lof_score_corr.round(3))


# In[90]:


jaccard_vs_20 = jaccard_matrix[20].drop(20)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(jaccard_vs_20.index, jaccard_vs_20.values, marker='s', color='darkred', linestyle='--')
axes[0].set_title('Stability of Anomalies (Jaccard similarity vs n=20)')
axes[0].set_xlabel('n_neighbors')
axes[0].set_ylabel('Jaccard Similarity')
axes[0].set_ylim(0, 1.1)
axes[0].grid(True, linestyle='--', alpha=0.6)

axes[1].plot(lof_sens_df['n_neighbors'], lof_sens_df['fit_time_s'], marker='o', color='steelblue')
axes[1].set_title('Training time vs n_neighbors')
axes[1].set_xlabel('n_neighbors')
axes[1].set_ylabel('Time (s)')
axes[1].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
#plt.show()


# **Selection rationale**:
# 
# Unlike n_estimators for Isolation Forest, LOF is sensitive to n_neighbors. The sensitivity analysis reveals that n_neighbors=30 is the optimal trade-off:
# 
# - Stability: The Jaccard similarity plot shows that n=30 achieves the highest label consistency, whereas smaller or larger values lead to a rapid decay in stability.
# 
# - Efficiency: The training time analysis confirms that n=30 resides in the global minimum of the computational cost curve.
# 
# - Model Purpose: This value is large enough to avoid noise sensitivity (inherent in n=10), yet small enough to maintain LOF’s ability to detect local anomalies, providing a necessary counterpoint to Isolation Forest’s global view.
# 
# This sensitivity is a known limitation of density-based models. Consequently, n=30 is chosen as a robust, computationally efficient configuration.
# 

# In[91]:


LOF_PARAMS = {'n_neighbors': 30, 'contamination': 0.05, 'n_jobs': -1}
print("Final LOF configuration:")
print(json.dumps(LOF_PARAMS, indent=2))


# ## 7. Final model execution
# 
# Run both models with their final configuration and generate scores and flags.
# 

# In[92]:


if_final = IsolationForest(**IF_PARAMS)

t0 = time.time()
if_final.fit(X)
if_fit_time = time.time() - t0

if_predictions = if_final.predict(X)
if_scores_raw = if_final.decision_function(X)
if_scores = if_scores_raw * -1  # inverted: higher = more anomalous

results_if = ids_df.copy()
results_if['if_score'] = if_scores
results_if['if_anomaly'] = if_predictions

print(f"Final IF training time: {if_fit_time:.2f}s")
print(f"Anomalies detected: {(if_predictions == -1).sum()} ({(if_predictions == -1).mean():.2%})")
display(results_if.sort_values('if_score', ascending=False).head())


# In[93]:
t0 = time.time()
lof_final = LocalOutlierFactor(**LOF_PARAMS)
lof_predictions = lof_final.fit_predict(X)
lof_full_fit_time = time.time() - t0
lof_scores = lof_final.negative_outlier_factor_ * -1

model_results = ids_df.copy()
model_results['if_score'] = if_scores
model_results['if_anomaly'] = if_predictions
model_results['lof_score'] = lof_scores
model_results['lof_anomaly'] = lof_predictions

print(f"LOF training time: {lof_full_fit_time:.2f}s")
print(f"Results dataframe shape: {model_results.shape}")
print(model_results.head(10))


# In[96]:
import json
import joblib
from pathlib import Path

# Directorios de salida en la raíz
OUTPUT_DIR = ROOT_DIR / "data" / "results"
MODELS_DIR = ROOT_DIR / "pipeline" / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Cargar datos raw para enriquecer resultados (precio, sqm, business rules)
raw_features_file = ROOT_DIR / "data" / "processed" / "dataset_raw_features.csv"
if raw_features_file.exists():
    df_raw_features = pd.read_csv(raw_features_file, sep=";")
    print(f"Raw features loaded: {df_raw_features.shape}")
    
    # Merge de resultados con features raw usando 'id'
    model_results_full = model_results.merge(df_raw_features, on='id', how='left', suffixes=('', '_raw'))
    
    # Añadir también X (features escaladas) para referencia
    X_reset = X.reset_index(drop=True)
    model_results_full = pd.concat([model_results_full.reset_index(drop=True), X_reset], axis=1)
    
    print(f"Model results enriched with raw features: {model_results_full.shape}")
else:
    print(f"Warning: {raw_features_file} not found. Using only model predictions without raw features.")
    # Fallback: solo concatenar modelo + features escaladas
    model_results_reset = model_results.reset_index(drop=True)
    X_reset = X.reset_index(drop=True)
    model_results_full = pd.concat([model_results_reset, X_reset], axis=1)

# Guardar resultados
model_results.to_csv(OUTPUT_DIR / "model_results.csv", sep=";", index=False)
model_results_full.to_csv(OUTPUT_DIR / "model_results_full.csv", sep=";", index=False)

print(f"Results saved with {model_results_full.shape[1]} columns (includes price, sqm, business rules)")

software_versions = {
    'python': sys.version.split()[0],
    'pandas': pd.__version__,
    'numpy': np.__version__,
    'scikit_learn': sklearn.__version__,
}
dataset_info = {
    'file_name': 'dataset_anomaly_detection_ready.csv',
    'n_rows': len(df),
    'n_features': X.shape[1],
    'extraction_id': None,
}

final_config = {
    'software_versions': software_versions,
    'dataset': dataset_info,
    'random_state': RANDOM_STATE,
    'n_observations': len(X),
    'n_features': X.shape[1],
    'feature_columns': feature_cols,
    'isolation_forest': {
        'params': IF_PARAMS,
        'n_anomalies': int((if_predictions == -1).sum()),
        'fit_time_s': round(if_fit_time, 2),
    },
    'local_outlier_factor': {
        'params': LOF_PARAMS,
        'n_anomalies': int((lof_predictions == -1).sum()),
        'fit_time_full_dataset_s': round(lof_full_fit_time, 2),
    },
}
with open(OUTPUT_DIR / "model_config.json", "w", encoding="utf-8") as f:
    json.dump(final_config, f, indent=2, ensure_ascii=False)

# Modelos entrenados exportados a pipeline/models/
joblib.dump(if_final, MODELS_DIR / "isolation_forest_final.joblib")

with open(MODELS_DIR / "lof_final_params.json", "w", encoding="utf-8") as f:
    json.dump(LOF_PARAMS, f, indent=2)

print("=" * 60)
print("Models and outputs saved")
print("=" * 60)

expected_files = [
    OUTPUT_DIR / 'model_results.csv',
    OUTPUT_DIR / 'model_results_full.csv',
    OUTPUT_DIR / 'model_config.json',
    MODELS_DIR / 'isolation_forest_final.joblib',
    MODELS_DIR / 'lof_final_params.json',
]
for f_path in expected_files:
    print(f"Generado: {f_path}")

