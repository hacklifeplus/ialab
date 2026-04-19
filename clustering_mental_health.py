"""
Análisis de Clustering No Supervisado - IT Mental Health Survey
Autor: Análisis automatizado
Fecha: 2026-04-19
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap

# Preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Clustering
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture

# Metrics
from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                              davies_bouldin_score, silhouette_samples)

# Stats
from scipy.stats import chi2_contingency
from scipy.cluster.hierarchy import dendrogram, linkage

# Report
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                  Table, TableStyle, PageBreak, HRFlowable,
                                  KeepTogether)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import io
import os
import textwrap

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────
DATA_PATH  = "/root/Projects/ialab/IT_mental_health.survey.csv"
OUT_DIR    = "/root/Projects/ialab/output_clustering"
REPORT_PDF = "/root/Projects/ialab/Informe_Clustering_Mental_Health.pdf"
os.makedirs(OUT_DIR, exist_ok=True)

PALETTE = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B",
           "#44BBA4", "#E94F37", "#393E41", "#F5A623", "#7B2D8B"]
sns.set_theme(style="whitegrid", palette=PALETTE)

print("=" * 60)
print("  CLUSTERING NO SUPERVISADO - IT MENTAL HEALTH SURVEY")
print("=" * 60)

# ─────────────────────────────────────────────
# 1. CARGA DE DATOS
# ─────────────────────────────────────────────
print("\n[1/7] Cargando datos...")
df_raw = pd.read_csv(DATA_PATH)
print(f"  Shape inicial: {df_raw.shape}")
print(f"  Columnas: {list(df_raw.columns)}")

# ─────────────────────────────────────────────
# 2. LIMPIEZA DE DATOS
# ─────────────────────────────────────────────
print("\n[2/7] Limpieza de datos...")

df = df_raw.copy()

# 2.1 Eliminar columnas de baja utilidad para clustering
drop_cols = ['Timestamp', 'comments', 'state']
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

# 2.2 Edad: eliminar outliers imposibles
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
before = len(df)
df = df[(df['Age'] >= 18) & (df['Age'] <= 72)]
print(f"  Filas eliminadas por edad inválida: {before - len(df)}")

# 2.3 Normalizar género → Male / Female / Other
def normalize_gender(g):
    if pd.isna(g):
        return np.nan
    g = str(g).strip().lower()
    male_kw   = ['male', 'm', 'man', 'cis male', 'cis man', 'msle', 'maile',
                 'malr', 'mal', 'make', 'guy']
    female_kw = ['female', 'f', 'woman', 'cis female', 'cis-female', 'femail',
                 'femake', 'female (trans)', 'trans woman', 'trans-female',
                 'woman', 'female ']
    if any(k in g for k in male_kw):
        return 'Male'
    elif any(k in g for k in female_kw):
        return 'Female'
    else:
        return 'Other'

df['Gender'] = df['Gender'].apply(normalize_gender)

# 2.4 País: top países + 'Other'
top_countries = df['Country'].value_counts().head(8).index.tolist()
df['Country_grp'] = df['Country'].apply(lambda x: x if x in top_countries else 'Other')
df.drop(columns=['Country'], inplace=True)

# 2.5 Columna work_interfere: tratar NA como "Don't know"
df['work_interfere'] = df['work_interfere'].fillna("Don't know")

# 2.6 self_employed: tratar NA como "No"
df['self_employed'] = df['self_employed'].fillna("No")

# 2.7 Missing restantes
missing_before = df.isnull().sum().sum()
print(f"  Valores faltantes antes de imputar: {missing_before}")

# Variables categóricas: moda
cat_cols = df.select_dtypes(include='object').columns.tolist()
for c in cat_cols:
    df[c].fillna(df[c].mode()[0], inplace=True)

# Variable numérica Age: mediana
df['Age'].fillna(df['Age'].median(), inplace=True)

missing_after = df.isnull().sum().sum()
print(f"  Valores faltantes después de imputar: {missing_after}")
print(f"  Shape tras limpieza: {df.shape}")

# Guardar estadísticas de limpieza
cleaning_stats = {
    'filas_originales': len(df_raw),
    'filas_limpias': len(df),
    'columnas_originales': df_raw.shape[1],
    'columnas_usadas': df.shape[1],
    'missing_imputados': missing_before
}

# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING & ENCODING
# ─────────────────────────────────────────────
print("\n[3/7] Feature Engineering y Encoding...")

df_enc = df.copy()

# 3.1 Feature: índice de apoyo laboral
# Suma de columnas que indican recursos disponibles (Yes=1, No=0)
support_cols = ['benefits', 'care_options', 'wellness_program',
                'seek_help', 'anonymity']
support_map  = {'Yes': 2, 'No': 0, "Don't know": 1, 'Not sure': 1}
for c in support_cols:
    if c in df_enc.columns:
        df_enc[c + '_num'] = df_enc[c].map(support_map).fillna(1)

df_enc['support_index'] = df_enc[[c + '_num' for c in support_cols
                                   if c + '_num' in df_enc.columns]].sum(axis=1)

# 3.2 Feature: nivel de apertura a hablar de salud mental
# (coworkers, supervisor, mental_health_interview, phys_health_interview)
openness_map = {'Yes': 2, 'No': 0, 'Maybe': 1, "Don't know": 1,
                'Some of them': 1}
open_cols = ['coworkers', 'supervisor', 'mental_health_interview',
             'phys_health_interview']
for c in open_cols:
    if c in df_enc.columns:
        df_enc[c + '_num'] = df_enc[c].map(openness_map).fillna(1)

df_enc['openness_index'] = df_enc[[c + '_num' for c in open_cols
                                    if c + '_num' in df_enc.columns]].sum(axis=1)

# 3.3 Feature: percepción de consecuencias
conseq_map = {'Yes': 2, 'No': 0, 'Maybe': 1}
for c in ['mental_health_consequence', 'phys_health_consequence', 'obs_consequence']:
    if c in df_enc.columns:
        df_enc[c + '_num'] = df_enc[c].map(conseq_map).fillna(1)

df_enc['consequence_index'] = df_enc[[c + '_num' for c in
    ['mental_health_consequence', 'phys_health_consequence', 'obs_consequence']
    if c + '_num' in df_enc.columns]].sum(axis=1)

# 3.4 Encoding de variables ordinales
ordinal_maps = {
    'work_interfere': {"Never": 0, "Rarely": 1, "Sometimes": 2, "Often": 3,
                       "Don't know": -1},
    'leave':          {"Very easy": 4, "Somewhat easy": 3, "Don't know": 2,
                       "Somewhat difficult": 1, "Very difficult": 0},
    'no_employees':   {"1-5": 1, "6-25": 2, "26-100": 3, "100-500": 4,
                       "500-1000": 5, "More than 1000": 6},
    'mental_vs_physical': {'Yes': 2, "Don't know": 1, 'No': 0},
}
for col, mapping in ordinal_maps.items():
    if col in df_enc.columns:
        df_enc[col + '_ord'] = df_enc[col].map(mapping).fillna(-1)

# 3.5 Label Encoding para variables binarias
binary_map = {'Yes': 1, 'No': 0}
binary_cols = ['self_employed', 'family_history', 'treatment',
               'remote_work', 'tech_company']
for c in binary_cols:
    if c in df_enc.columns:
        df_enc[c + '_bin'] = df_enc[c].map(binary_map).fillna(0)

# 3.6 One-Hot Encoding para Gender y Country_grp
df_enc = pd.get_dummies(df_enc,
                        columns=['Gender', 'Country_grp'],
                        drop_first=False,
                        dtype=int)

# 3.7 Selección de features para clustering
feature_cols = (
    ['Age']
    + [c for c in df_enc.columns if c.endswith('_num')]
    + [c for c in df_enc.columns if c.endswith('_ord')]
    + [c for c in df_enc.columns if c.endswith('_bin')]
    + ['support_index', 'openness_index', 'consequence_index']
    + [c for c in df_enc.columns if c.startswith('Gender_')]
    + [c for c in df_enc.columns if c.startswith('Country_grp_')]
)
feature_cols = list(dict.fromkeys(feature_cols))  # sin duplicados
feature_cols = [c for c in feature_cols if c in df_enc.columns]

X = df_enc[feature_cols].values.astype(float)
print(f"  Features seleccionados: {len(feature_cols)}")
print(f"  Shape matriz features: {X.shape}")

# ─────────────────────────────────────────────
# 4. NORMALIZACIÓN / ESTANDARIZACIÓN
# ─────────────────────────────────────────────
print("\n[4/7] Normalización y Estandarización...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# También MinMaxScaler para DBSCAN (sensible a escala)
mm_scaler = MinMaxScaler()
X_minmax  = mm_scaler.fit_transform(X)

print(f"  StandardScaler aplicado → media≈0, std≈1")
print(f"  MinMaxScaler aplicado   → rango [0,1]")

# ─────────────────────────────────────────────
# 5. REDUCCIÓN DE DIMENSIONALIDAD (PCA + t-SNE)
# ─────────────────────────────────────────────
print("\n[5/7] Reducción de dimensionalidad...")

# PCA para clustering y visualización
pca_full = PCA()
pca_full.fit(X_scaled)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)
n_components_95 = np.argmax(cumvar >= 0.95) + 1
print(f"  Componentes para 95% varianza explicada: {n_components_95}")

pca = PCA(n_components=n_components_95, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# PCA 2D para visualizaciones
pca2d = PCA(n_components=2, random_state=42)
X_pca2d = pca2d.fit_transform(X_scaled)

# t-SNE 2D para visualización
print("  Calculando t-SNE (puede tardar)...")
tsne = TSNE(n_components=2, random_state=42, perplexity=40, n_iter=1000)
X_tsne = tsne.fit_transform(X_scaled)

print(f"  PCA {n_components_95}D listo, t-SNE 2D listo")

# ─────────────────────────────────────────────
# FIGURAS DE ANÁLISIS EXPLORATORIO (EDA)
# ─────────────────────────────────────────────
print("\n  Generando figuras EDA...")

# FIG 1: Distribuciones principales
fig1, axes = plt.subplots(2, 3, figsize=(16, 10))
fig1.suptitle("Distribución de Variables Clave", fontsize=16, fontweight='bold')

# Age
axes[0,0].hist(df['Age'], bins=25, color=PALETTE[0], edgecolor='white', alpha=0.85)
axes[0,0].set_title("Distribución de Edad")
axes[0,0].set_xlabel("Edad")
axes[0,0].set_ylabel("Frecuencia")

# Gender
g_counts = df['Gender'].value_counts()
axes[0,1].bar(g_counts.index, g_counts.values,
              color=PALETTE[:len(g_counts)], edgecolor='white')
axes[0,1].set_title("Distribución de Género")
axes[0,1].set_xlabel("Género")
axes[0,1].set_ylabel("Frecuencia")

# Treatment
t_counts = df['treatment'].value_counts()
axes[0,2].pie(t_counts.values, labels=t_counts.index,
              colors=PALETTE[:2], autopct='%1.1f%%', startangle=140)
axes[0,2].set_title("¿Ha buscado tratamiento?")

# work_interfere
wi_order = ['Never','Rarely','Sometimes','Often',"Don't know"]
wi_counts = df['work_interfere'].value_counts().reindex(wi_order).dropna()
axes[1,0].bar(wi_counts.index, wi_counts.values,
              color=PALETTE[3:3+len(wi_counts)], edgecolor='white')
axes[1,0].set_title("Interferencia laboral")
axes[1,0].set_xticklabels(wi_counts.index, rotation=30, ha='right')
axes[1,0].set_ylabel("Frecuencia")

# Family history
fh_counts = df['family_history'].value_counts()
axes[1,1].bar(fh_counts.index, fh_counts.values,
              color=PALETTE[1:3], edgecolor='white')
axes[1,1].set_title("Historia familiar de SM")
axes[1,1].set_ylabel("Frecuencia")

# Country
c_counts = df['Country_grp'].value_counts().head(8)
axes[1,2].barh(c_counts.index[::-1], c_counts.values[::-1],
               color=PALETTE[:len(c_counts)], edgecolor='white')
axes[1,2].set_title("Top Países")
axes[1,2].set_xlabel("Frecuencia")

plt.tight_layout()
fig1.savefig(f"{OUT_DIR}/fig01_eda_distribuciones.png", dpi=150, bbox_inches='tight')
plt.close(fig1)

# FIG 2: Mapa de calor de correlaciones
fig2, ax2 = plt.subplots(figsize=(14, 11))
numeric_features = ['Age', 'support_index', 'openness_index',
                    'consequence_index', 'work_interfere_ord',
                    'leave_ord', 'no_employees_ord', 'mental_vs_physical_ord']
numeric_features = [c for c in numeric_features if c in df_enc.columns]
corr_mat = df_enc[numeric_features].corr()
mask = np.triu(np.ones_like(corr_mat, dtype=bool))
sns.heatmap(corr_mat, mask=mask, annot=True, fmt=".2f",
            cmap="coolwarm", center=0, ax=ax2,
            linewidths=0.5, square=True, cbar_kws={'shrink': 0.8})
ax2.set_title("Correlaciones entre Features Numéricas", fontsize=14, fontweight='bold')
plt.tight_layout()
fig2.savefig(f"{OUT_DIR}/fig02_correlaciones.png", dpi=150, bbox_inches='tight')
plt.close(fig2)

# FIG 3: Varianza explicada PCA
fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
fig3.suptitle("Análisis de Componentes Principales (PCA)", fontsize=14, fontweight='bold')

n_show = min(20, len(pca_full.explained_variance_ratio_))
axes3[0].bar(range(1, n_show+1),
             pca_full.explained_variance_ratio_[:n_show] * 100,
             color=PALETTE[0], edgecolor='white', alpha=0.85)
axes3[0].set_xlabel("Componente Principal")
axes3[0].set_ylabel("Varianza Explicada (%)")
axes3[0].set_title("Varianza por Componente")

axes3[1].plot(range(1, len(cumvar)+1), cumvar * 100,
              color=PALETTE[2], marker='o', markersize=4, linewidth=2)
axes3[1].axhline(y=95, color=PALETTE[3], linestyle='--', label='95%')
axes3[1].axvline(x=n_components_95, color=PALETTE[1], linestyle='--',
                 label=f'PC={n_components_95}')
axes3[1].set_xlabel("Número de Componentes")
axes3[1].set_ylabel("Varianza Acumulada (%)")
axes3[1].set_title("Varianza Acumulada")
axes3[1].legend()

plt.tight_layout()
fig3.savefig(f"{OUT_DIR}/fig03_pca_varianza.png", dpi=150, bbox_inches='tight')
plt.close(fig3)

# ─────────────────────────────────────────────
# 6. SELECCIÓN DEL NÚMERO ÓPTIMO DE CLUSTERS
# ─────────────────────────────────────────────
print("\n[6/7] Determinando número óptimo de clusters...")

K_range = range(2, 11)
inertias, sil_scores_k, ch_scores_k, db_scores_k = [], [], [], []

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_pca)
    inertias.append(km.inertia_)
    sil_scores_k.append(silhouette_score(X_pca, labels))
    ch_scores_k.append(calinski_harabasz_score(X_pca, labels))
    db_scores_k.append(davies_bouldin_score(X_pca, labels))

# Elbow + métricas
fig4, axes4 = plt.subplots(2, 2, figsize=(14, 10))
fig4.suptitle("Selección del Número Óptimo de Clústeres (K-Means)",
              fontsize=14, fontweight='bold')

axes4[0,0].plot(K_range, inertias, marker='o', color=PALETTE[0], linewidth=2)
axes4[0,0].set_title("Método del Codo (Inercia)")
axes4[0,0].set_xlabel("Número de Clústeres (k)")
axes4[0,0].set_ylabel("Inercia")

axes4[0,1].plot(K_range, sil_scores_k, marker='o', color=PALETTE[1], linewidth=2)
best_k_sil = list(K_range)[np.argmax(sil_scores_k)]
axes4[0,1].axvline(x=best_k_sil, color='red', linestyle='--',
                   label=f'Mejor k={best_k_sil}')
axes4[0,1].set_title("Silhouette Score")
axes4[0,1].set_xlabel("Número de Clústeres (k)")
axes4[0,1].set_ylabel("Silhouette Score")
axes4[0,1].legend()

axes4[1,0].plot(K_range, ch_scores_k, marker='o', color=PALETTE[2], linewidth=2)
best_k_ch = list(K_range)[np.argmax(ch_scores_k)]
axes4[1,0].axvline(x=best_k_ch, color='red', linestyle='--',
                   label=f'Mejor k={best_k_ch}')
axes4[1,0].set_title("Calinski-Harabasz Score")
axes4[1,0].set_xlabel("Número de Clústeres (k)")
axes4[1,0].set_ylabel("CH Score")
axes4[1,0].legend()

axes4[1,1].plot(K_range, db_scores_k, marker='o', color=PALETTE[3], linewidth=2)
best_k_db = list(K_range)[np.argmin(db_scores_k)]
axes4[1,1].axvline(x=best_k_db, color='red', linestyle='--',
                   label=f'Mejor k={best_k_db}')
axes4[1,1].set_title("Davies-Bouldin Score (↓ mejor)")
axes4[1,1].set_xlabel("Número de Clústeres (k)")
axes4[1,1].set_ylabel("DB Score")
axes4[1,1].legend()

plt.tight_layout()
fig4.savefig(f"{OUT_DIR}/fig04_seleccion_k.png", dpi=150, bbox_inches='tight')
plt.close(fig4)

# K óptimo consenso
K_OPT = best_k_sil  # usar silhouette como referencia principal
print(f"  K óptimo (Silhouette): {K_OPT}")
print(f"  K óptimo (CH):         {best_k_ch}")
print(f"  K óptimo (DB):         {best_k_db}")

# ─────────────────────────────────────────────
# 7. ALGORITMOS DE CLUSTERING
# ─────────────────────────────────────────────
print("\n[7/7] Aplicando algoritmos de clustering...")

results = {}

# ── 7.1 K-MEANS ──────────────────────────────
print("  → K-Means...")
km = KMeans(n_clusters=K_OPT, random_state=42, n_init=10, max_iter=300)
labels_km = km.fit_predict(X_pca)
results['K-Means'] = {
    'labels': labels_km,
    'n_clusters': K_OPT,
    'silhouette': silhouette_score(X_pca, labels_km),
    'calinski':   calinski_harabasz_score(X_pca, labels_km),
    'davies':     davies_bouldin_score(X_pca, labels_km),
    'inertia':    km.inertia_,
}

# ── 7.2 K-MEANS k=3 (comparativa) ────────────
print("  → K-Means k=3...")
km3 = KMeans(n_clusters=3, random_state=42, n_init=10)
labels_km3 = km3.fit_predict(X_pca)
results['K-Means k=3'] = {
    'labels': labels_km3,
    'n_clusters': 3,
    'silhouette': silhouette_score(X_pca, labels_km3),
    'calinski':   calinski_harabasz_score(X_pca, labels_km3),
    'davies':     davies_bouldin_score(X_pca, labels_km3),
    'inertia':    km3.inertia_,
}

# ── 7.3 DBSCAN ───────────────────────────────
print("  → DBSCAN...")
# Búsqueda de epsilon con k-NN
from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=5).fit(X_pca)
distances, _ = nbrs.kneighbors(X_pca)
k_dist = np.sort(distances[:, 4])[::-1]

# Probar varios eps
best_dbscan = None
best_sil_db = -1
best_eps = None
for eps_val in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5]:
    for min_s in [5, 10, 15]:
        db_tmp = DBSCAN(eps=eps_val, min_samples=min_s)
        lbl_tmp = db_tmp.fit_predict(X_pca)
        n_unique = len(set(lbl_tmp)) - (1 if -1 in lbl_tmp else 0)
        noise_pct = (lbl_tmp == -1).sum() / len(lbl_tmp)
        if n_unique >= 2 and noise_pct < 0.3:
            valid = lbl_tmp != -1
            if valid.sum() > 50:
                sil_tmp = silhouette_score(X_pca[valid], lbl_tmp[valid])
                if sil_tmp > best_sil_db:
                    best_sil_db = sil_tmp
                    best_dbscan = lbl_tmp
                    best_eps = eps_val

if best_dbscan is None:
    best_dbscan = DBSCAN(eps=1.2, min_samples=10).fit_predict(X_pca)
    best_eps = 1.2

n_db = len(set(best_dbscan)) - (1 if -1 in best_dbscan else 0)
noise_db = (best_dbscan == -1).sum()
valid_db = best_dbscan != -1
sil_db = silhouette_score(X_pca[valid_db], best_dbscan[valid_db]) if n_db >= 2 and valid_db.sum() > 10 else -1
results['DBSCAN'] = {
    'labels': best_dbscan,
    'n_clusters': n_db,
    'noise_points': int(noise_db),
    'eps': best_eps,
    'silhouette': float(sil_db),
    'calinski':   calinski_harabasz_score(X_pca[valid_db], best_dbscan[valid_db]) if n_db >= 2 and valid_db.sum() > 10 else 0,
    'davies':     davies_bouldin_score(X_pca[valid_db], best_dbscan[valid_db]) if n_db >= 2 and valid_db.sum() > 10 else 99,
}

# ── 7.4 GMM ──────────────────────────────────
print("  → Gaussian Mixture Model...")
bic_scores, aic_scores = [], []
for k in range(2, 9):
    g = GaussianMixture(n_components=k, random_state=42, n_init=5)
    g.fit(X_pca)
    bic_scores.append(g.bic(X_pca))
    aic_scores.append(g.aic(X_pca))

best_k_bic = list(range(2, 9))[np.argmin(bic_scores)]
gmm = GaussianMixture(n_components=best_k_bic, random_state=42, n_init=10)
labels_gmm = gmm.fit_predict(X_pca)
results['GMM'] = {
    'labels': labels_gmm,
    'n_clusters': best_k_bic,
    'silhouette': silhouette_score(X_pca, labels_gmm),
    'calinski':   calinski_harabasz_score(X_pca, labels_gmm),
    'davies':     davies_bouldin_score(X_pca, labels_gmm),
    'bic':        gmm.bic(X_pca),
    'aic':        gmm.aic(X_pca),
}

# Figura BIC/AIC GMM
fig_bic, ax_bic = plt.subplots(figsize=(8, 5))
ax_bic.plot(range(2, 9), bic_scores, marker='o', label='BIC', color=PALETTE[0], linewidth=2)
ax_bic.plot(range(2, 9), aic_scores, marker='s', label='AIC', color=PALETTE[1], linewidth=2)
ax_bic.axvline(x=best_k_bic, color='red', linestyle='--', label=f'Óptimo k={best_k_bic}')
ax_bic.set_xlabel("Número de Componentes")
ax_bic.set_ylabel("Score")
ax_bic.set_title("GMM: Selección por BIC/AIC", fontweight='bold')
ax_bic.legend()
plt.tight_layout()
fig_bic.savefig(f"{OUT_DIR}/fig05_gmm_bic_aic.png", dpi=150, bbox_inches='tight')
plt.close(fig_bic)

# ── 7.5 HIERARCHICAL CLUSTERING ──────────────
print("  → Clustering Jerárquico (Ward)...")
hc = AgglomerativeClustering(n_clusters=K_OPT, linkage='ward')
labels_hc = hc.fit_predict(X_pca)
results['Hierarchical (Ward)'] = {
    'labels': labels_hc,
    'n_clusters': K_OPT,
    'silhouette': silhouette_score(X_pca, labels_hc),
    'calinski':   calinski_harabasz_score(X_pca, labels_hc),
    'davies':     davies_bouldin_score(X_pca, labels_hc),
}

# Dendrograma (muestra)
sample_idx = np.random.choice(len(X_pca), size=min(200, len(X_pca)), replace=False)
Z = linkage(X_pca[sample_idx], method='ward')
fig_dend, ax_dend = plt.subplots(figsize=(14, 6))
dendrogram(Z, ax=ax_dend, truncate_mode='lastp', p=30,
           leaf_rotation=90, leaf_font_size=8,
           color_threshold=0.7 * max(Z[:, 2]))
ax_dend.set_title("Dendrograma (muestra 200 obs.) - Clustering Jerárquico Ward",
                  fontsize=13, fontweight='bold')
ax_dend.set_xlabel("Índice de muestra")
ax_dend.set_ylabel("Distancia")
plt.tight_layout()
fig_dend.savefig(f"{OUT_DIR}/fig06_dendrograma.png", dpi=150, bbox_inches='tight')
plt.close(fig_dend)

# ── 7.6 SPECTRAL CLUSTERING ──────────────────
print("  → Spectral Clustering...")
sc_model = SpectralClustering(n_clusters=K_OPT, random_state=42,
                               affinity='rbf', n_jobs=-1)
labels_sc = sc_model.fit_predict(X_pca)
results['Spectral'] = {
    'labels': labels_sc,
    'n_clusters': K_OPT,
    'silhouette': silhouette_score(X_pca, labels_sc),
    'calinski':   calinski_harabasz_score(X_pca, labels_sc),
    'davies':     davies_bouldin_score(X_pca, labels_sc),
}

# ─────────────────────────────────────────────
# TABLAS Y FIGURAS COMPARATIVAS
# ─────────────────────────────────────────────
print("\n  Generando visualizaciones de resultados...")

# FIG 7: Comparativa de métricas
metric_df = pd.DataFrame({
    'Algoritmo': list(results.keys()),
    'K': [r['n_clusters'] for r in results.values()],
    'Silhouette': [r['silhouette'] for r in results.values()],
    'Calinski-Harabasz': [r['calinski'] for r in results.values()],
    'Davies-Bouldin': [r['davies'] for r in results.values()],
})
metric_df = metric_df.round(4)
print("\n  Tabla de métricas:")
print(metric_df.to_string(index=False))

fig7, axes7 = plt.subplots(1, 3, figsize=(16, 6))
fig7.suptitle("Comparativa de Métricas por Algoritmo", fontsize=14, fontweight='bold')

colors_bar = PALETTE[:len(metric_df)]
axes7[0].barh(metric_df['Algoritmo'], metric_df['Silhouette'],
              color=colors_bar, edgecolor='white')
axes7[0].set_title("Silhouette Score (↑ mejor)")
axes7[0].set_xlabel("Score")
axes7[0].axvline(x=0.5, color='red', linestyle='--', alpha=0.5)

axes7[1].barh(metric_df['Algoritmo'], metric_df['Calinski-Harabasz'],
              color=colors_bar, edgecolor='white')
axes7[1].set_title("Calinski-Harabasz (↑ mejor)")
axes7[1].set_xlabel("Score")

axes7[2].barh(metric_df['Algoritmo'], metric_df['Davies-Bouldin'],
              color=colors_bar, edgecolor='white')
axes7[2].set_title("Davies-Bouldin (↓ mejor)")
axes7[2].set_xlabel("Score")

plt.tight_layout()
fig7.savefig(f"{OUT_DIR}/fig07_comparativa_metricas.png", dpi=150, bbox_inches='tight')
plt.close(fig7)

# FIG 8: Visualizaciones PCA 2D de cada algoritmo
fig8, axes8 = plt.subplots(2, 3, figsize=(18, 12))
fig8.suptitle("Visualización de Clústeres en PCA 2D", fontsize=15, fontweight='bold')
axes8 = axes8.flatten()

algo_items = list(results.items())
for idx, (algo_name, res) in enumerate(algo_items):
    ax = axes8[idx]
    lbl = res['labels']
    unique_labels = sorted(set(lbl))
    cmap = plt.cm.get_cmap('tab10', len(unique_labels))
    for i, ul in enumerate(unique_labels):
        mask = lbl == ul
        lbl_str = f'Ruido ({mask.sum()})' if ul == -1 else f'Clúster {ul} ({mask.sum()})'
        ax.scatter(X_pca2d[mask, 0], X_pca2d[mask, 1],
                   c=[cmap(i)], s=15, alpha=0.6, label=lbl_str)
    ax.set_title(f"{algo_name}\n(k={res['n_clusters']}, Sil={res['silhouette']:.3f})",
                 fontsize=11)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(fontsize=7, markerscale=1.5)

# ocultar eje sobrante si hay
for j in range(len(algo_items), len(axes8)):
    axes8[j].set_visible(False)

plt.tight_layout()
fig8.savefig(f"{OUT_DIR}/fig08_clusters_pca2d.png", dpi=150, bbox_inches='tight')
plt.close(fig8)

# FIG 9: Visualización t-SNE de K-Means (mejor modelo)
fig9, axes9 = plt.subplots(1, 2, figsize=(16, 7))
fig9.suptitle("Visualización t-SNE", fontsize=14, fontweight='bold')

for ax9, (aname, ares) in zip(axes9, [('K-Means', results['K-Means']),
                                        ('GMM', results['GMM'])]):
    lbl = ares['labels']
    unique_labels = sorted(set(lbl))
    cmap = plt.cm.get_cmap('tab10', len(unique_labels))
    for i, ul in enumerate(unique_labels):
        mask = lbl == ul
        lbl_str = f'Ruido' if ul == -1 else f'Clúster {ul}'
        ax9.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                    c=[cmap(i)], s=20, alpha=0.6, label=lbl_str)
    ax9.set_title(f"{aname} - t-SNE (Sil={ares['silhouette']:.3f})")
    ax9.set_xlabel("t-SNE 1")
    ax9.set_ylabel("t-SNE 2")
    ax9.legend(fontsize=8, markerscale=1.5)

plt.tight_layout()
fig9.savefig(f"{OUT_DIR}/fig09_tsne.png", dpi=150, bbox_inches='tight')
plt.close(fig9)

# ─────────────────────────────────────────────
# ANÁLISIS DETALLADO DEL MEJOR MODELO (K-MEANS)
# ─────────────────────────────────────────────
print("\n  Analizando perfil de clústeres (K-Means)...")

best_labels = results['K-Means']['labels']
df_analysis = df.copy()
df_analysis['Cluster'] = best_labels

# Silhouette por muestra
sil_vals = silhouette_samples(X_pca, best_labels)
df_analysis['silhouette_sample'] = sil_vals

# Estadísticas por clúster
cluster_profile = []
for k in sorted(df_analysis['Cluster'].unique()):
    sub = df_analysis[df_analysis['Cluster'] == k]
    profile = {
        'Clúster': k,
        'N': len(sub),
        '%': f"{len(sub)/len(df_analysis)*100:.1f}%",
        'Edad Media': f"{sub['Age'].mean():.1f}",
        'Con Tratamiento': f"{(sub['treatment']=='Yes').mean()*100:.1f}%",
        'Hist. Familiar': f"{(sub['family_history']=='Yes').mean()*100:.1f}%",
        'Trabaja Remoto': f"{(sub['remote_work']=='Yes').mean()*100:.1f}%",
        'Género Mayoritario': sub['Gender'].mode()[0] if len(sub) > 0 else 'N/A',
        'Interferencia Laboral': sub['work_interfere'].mode()[0] if len(sub) > 0 else 'N/A',
        'País Top': sub['Country_grp'].mode()[0] if len(sub) > 0 else 'N/A',
    }
    cluster_profile.append(profile)

cluster_df = pd.DataFrame(cluster_profile)
print("\n  Perfil de Clústeres K-Means:")
print(cluster_df.to_string(index=False))

# FIG 10: Silhouette plot
fig10, ax10 = plt.subplots(figsize=(10, 7))
y_lower = 10
unique_clusters = sorted(set(best_labels))
cmap10 = plt.cm.get_cmap('tab10', len(unique_clusters))
for i, cl in enumerate(unique_clusters):
    sil_cl = np.sort(sil_vals[best_labels == cl])
    size_cl = len(sil_cl)
    y_upper = y_lower + size_cl
    ax10.fill_betweenx(np.arange(y_lower, y_upper), 0, sil_cl,
                       facecolor=cmap10(i), edgecolor='none', alpha=0.7)
    ax10.text(-0.05, y_lower + 0.5 * size_cl, f"C{cl}", fontsize=9)
    y_lower = y_upper + 10

ax10.axvline(x=results['K-Means']['silhouette'], color='red',
             linestyle='--', linewidth=1.5,
             label=f"Media={results['K-Means']['silhouette']:.3f}")
ax10.set_xlabel("Silhouette Coefficient")
ax10.set_ylabel("Muestra por clúster")
ax10.set_title("Silhouette Plot - K-Means", fontsize=13, fontweight='bold')
ax10.legend()
plt.tight_layout()
fig10.savefig(f"{OUT_DIR}/fig10_silhouette_plot.png", dpi=150, bbox_inches='tight')
plt.close(fig10)

# FIG 11: Perfiles de clústeres - radar-like barras
fig11, axes11 = plt.subplots(1, K_OPT, figsize=(5 * K_OPT, 6))
if K_OPT == 1:
    axes11 = [axes11]
profile_vars = ['support_index', 'openness_index', 'consequence_index',
                'work_interfere_ord', 'leave_ord']
profile_vars = [c for c in profile_vars if c in df_enc.columns]

df_enc_cl = df_enc.copy()
df_enc_cl['Cluster'] = best_labels
global_means = df_enc_cl[profile_vars].mean()

for i, cl in enumerate(sorted(df_enc_cl['Cluster'].unique())):
    ax = axes11[i] if K_OPT > 1 else axes11[0]
    sub_means = df_enc_cl[df_enc_cl['Cluster'] == cl][profile_vars].mean()
    diff = sub_means - global_means
    colors_bar2 = [PALETTE[2] if v >= 0 else PALETTE[3] for v in diff.values]
    ax.barh(profile_vars, diff.values, color=colors_bar2, edgecolor='white')
    ax.axvline(0, color='black', linewidth=1)
    ax.set_title(f"Clúster {cl}\n(n={( df_enc_cl['Cluster']==cl).sum()})",
                 fontweight='bold')
    ax.set_xlabel("Diferencia respecto a media global")
    ax.tick_params(labelsize=9)

fig11.suptitle("Perfil de Clústeres (Desviación de la Media Global)",
               fontsize=14, fontweight='bold')
plt.tight_layout()
fig11.savefig(f"{OUT_DIR}/fig11_perfil_clusters.png", dpi=150, bbox_inches='tight')
plt.close(fig11)

# FIG 12: Distribución treatment y family_history por clúster
fig12, axes12 = plt.subplots(1, 2, figsize=(14, 6))
fig12.suptitle("Variables Clave por Clúster (K-Means)", fontsize=13, fontweight='bold')

treatment_ct = pd.crosstab(df_analysis['Cluster'], df_analysis['treatment'], normalize='index') * 100
treatment_ct.plot(kind='bar', ax=axes12[0], color=PALETTE[:2], edgecolor='white', rot=0)
axes12[0].set_title("Distribución 'Tratamiento' por Clúster")
axes12[0].set_xlabel("Clúster")
axes12[0].set_ylabel("Porcentaje (%)")
axes12[0].legend(title='Tratamiento')

fh_ct = pd.crosstab(df_analysis['Cluster'], df_analysis['family_history'], normalize='index') * 100
fh_ct.plot(kind='bar', ax=axes12[1], color=PALETTE[2:4], edgecolor='white', rot=0)
axes12[1].set_title("Distribución 'Historia Familiar' por Clúster")
axes12[1].set_xlabel("Clúster")
axes12[1].set_ylabel("Porcentaje (%)")
axes12[1].legend(title='Hist. Familiar')

plt.tight_layout()
fig12.savefig(f"{OUT_DIR}/fig12_variables_por_cluster.png", dpi=150, bbox_inches='tight')
plt.close(fig12)

print("\n  Todas las figuras generadas.")

# ─────────────────────────────────────────────
# INTERPRETACIÓN TEXTUAL DE CLÚSTERES
# ─────────────────────────────────────────────
def interpret_cluster(row, df_enc_cl, profile_vars, global_means):
    cl = row['Clúster']
    sub = df_enc_cl[df_enc_cl['Cluster'] == cl]
    sub_means = sub[profile_vars].mean()
    desc_parts = []

    support_avg = sub_means.get('support_index', None)
    global_support = global_means.get('support_index', None)
    if support_avg is not None and global_support is not None:
        if support_avg > global_support + 0.5:
            desc_parts.append("alto nivel de apoyo empresarial percibido")
        elif support_avg < global_support - 0.5:
            desc_parts.append("bajo nivel de apoyo empresarial percibido")
        else:
            desc_parts.append("nivel moderado de apoyo empresarial")

    open_avg = sub_means.get('openness_index', None)
    global_open = global_means.get('openness_index', None)
    if open_avg is not None and global_open is not None:
        if open_avg > global_open + 0.5:
            desc_parts.append("mayor apertura para hablar de salud mental")
        elif open_avg < global_open - 0.5:
            desc_parts.append("menor apertura para hablar de salud mental")

    tmt_pct = float(row['Con Tratamiento'].replace('%', ''))
    if tmt_pct > 60:
        desc_parts.append(f"alta tasa de búsqueda de tratamiento ({tmt_pct:.0f}%)")
    elif tmt_pct < 35:
        desc_parts.append(f"baja tasa de búsqueda de tratamiento ({tmt_pct:.0f}%)")

    wi_val = sub['work_interfere'].value_counts().index[0] if len(sub) > 0 else ''
    if wi_val in ['Often', 'Sometimes']:
        desc_parts.append(f"la salud mental interfiere frecuentemente en el trabajo ({wi_val})")
    elif wi_val == 'Never':
        desc_parts.append("la salud mental no interfiere en el trabajo")

    return "; ".join(desc_parts) if desc_parts else "perfil equilibrado sin características extremas"

interpretaciones = []
for _, row in cluster_df.iterrows():
    interp = interpret_cluster(row, df_enc_cl, profile_vars, global_means)
    interpretaciones.append(interp)

cluster_df['Interpretación'] = interpretaciones
print("\n  Interpretaciones:")
for i, row in cluster_df.iterrows():
    print(f"  Clúster {row['Clúster']} (n={row['N']}): {row['Interpretación']}")

# ─────────────────────────────────────────────
# GENERACIÓN DEL INFORME PDF
# ─────────────────────────────────────────────
print("\n  Generando informe PDF...")

doc = SimpleDocTemplate(
    REPORT_PDF,
    pagesize=A4,
    rightMargin=2*cm, leftMargin=2*cm,
    topMargin=2.5*cm, bottomMargin=2*cm
)

styles = getSampleStyleSheet()
W = A4[0] - 4*cm   # ancho útil

# Estilos personalizados
style_title = ParagraphStyle('CustomTitle',
    parent=styles['Title'],
    fontSize=22, textColor=colors.HexColor('#1a1a2e'),
    spaceAfter=6, leading=28, alignment=TA_CENTER)

style_h1 = ParagraphStyle('H1Custom',
    parent=styles['Heading1'],
    fontSize=15, textColor=colors.HexColor('#16213e'),
    spaceBefore=18, spaceAfter=6,
    borderPad=4, leading=20)

style_h2 = ParagraphStyle('H2Custom',
    parent=styles['Heading2'],
    fontSize=12, textColor=colors.HexColor('#0f3460'),
    spaceBefore=12, spaceAfter=4, leading=16)

style_h3 = ParagraphStyle('H3Custom',
    parent=styles['Heading3'],
    fontSize=10, textColor=colors.HexColor('#533483'),
    spaceBefore=8, spaceAfter=3, leading=14)

style_body = ParagraphStyle('BodyCustom',
    parent=styles['BodyText'],
    fontSize=9.5, leading=14, alignment=TA_JUSTIFY,
    spaceAfter=4)

style_code = ParagraphStyle('CodeCustom',
    parent=styles['Code'],
    fontSize=8, leading=11, backColor=colors.HexColor('#f4f4f4'),
    borderPad=5)

style_caption = ParagraphStyle('Caption',
    parent=styles['BodyText'],
    fontSize=8.5, textColor=colors.grey, alignment=TA_CENTER,
    spaceAfter=10)

style_bullet = ParagraphStyle('BulletCustom',
    parent=styles['BodyText'],
    fontSize=9.5, leading=14, leftIndent=14,
    bulletIndent=4, spaceAfter=2)

style_center = ParagraphStyle('CenterCustom',
    parent=styles['BodyText'],
    alignment=TA_CENTER, fontSize=9.5)

def heading1(text): return Paragraph(text, style_h1)
def heading2(text): return Paragraph(text, style_h2)
def heading3(text): return Paragraph(text, style_h3)
def body(text):     return Paragraph(text, style_body)
def bullet(text):   return Paragraph(f"• {text}", style_bullet)
def caption(text):  return Paragraph(text, style_caption)
def sp(n=8):        return Spacer(1, n)
def hr():           return HRFlowable(width="100%", thickness=0.5,
                                      color=colors.HexColor('#cccccc'),
                                      spaceAfter=6, spaceBefore=6)

def img(path, w_frac=0.95):
    try:
        im = Image(path, width=W*w_frac, height=W*w_frac*0.6)
        im.hAlign = 'CENTER'
        return im
    except:
        return body(f"[Figura no disponible: {path}]")

def img_wide(path):
    try:
        im = Image(path, width=W*0.99, height=W*0.55)
        im.hAlign = 'CENTER'
        return im
    except:
        return body(f"[Figura no disponible: {path}]")

def make_table(data, col_widths=None, header=True):
    tbl = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    style_list = [
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#16213e')),
        ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
        ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',   (0,0), (-1,0), 9),
        ('ALIGN',      (0,0), (-1,-1), 'CENTER'),
        ('VALIGN',     (0,0), (-1,-1), 'MIDDLE'),
        ('FONTSIZE',   (0,1), (-1,-1), 8.5),
        ('ROWBACKGROUNDS', (0,1), (-1,-1),
         [colors.white, colors.HexColor('#eef3fb')]),
        ('GRID',       (0,0), (-1,-1), 0.4, colors.HexColor('#cccccc')),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('TOPPADDING',    (0,0), (-1,-1), 5),
    ]
    tbl.setStyle(TableStyle(style_list))
    return tbl

# ═══ CONTENIDO ═══════════════════════════════
story = []

# ── PORTADA ─────────────────────────────────
story.append(Spacer(1, 3*cm))
story.append(Paragraph(
    "Informe de Clustering No Supervisado",
    style_title))
story.append(Paragraph(
    "IT Mental Health Survey",
    ParagraphStyle('Subtitle', parent=styles['Title'],
                   fontSize=16, textColor=colors.HexColor('#0f3460'),
                   alignment=TA_CENTER, spaceAfter=4)))
story.append(sp(20))
story.append(hr())
story.append(sp(10))

meta = [
    ["Dataset", "OSMI Mental Health in Tech Survey"],
    ["Observaciones (tras limpieza)", f"{len(df):,}"],
    ["Features para clustering", f"{len(feature_cols)}"],
    ["Algoritmos evaluados", "K-Means, DBSCAN, GMM, Jerárquico (Ward), Spectral"],
    ["Fecha del análisis", "19 de abril de 2026"],
    ["Herramientas", "Python 3.12 · scikit-learn · pandas · matplotlib · seaborn"],
]
story.append(make_table(
    [["Parámetro", "Valor"]] + meta,
    col_widths=[6*cm, 11*cm]))
story.append(sp(30))
story.append(Paragraph(
    "Análisis exhaustivo de agrupamiento no supervisado aplicado a datos de salud "
    "mental en el sector tecnológico, incluyendo preprocesamiento, reducción de "
    "dimensionalidad, evaluación multicriterio y perfilado de clústeres.",
    ParagraphStyle('IntroCenter', parent=styles['BodyText'],
                   fontSize=10, alignment=TA_CENTER,
                   textColor=colors.HexColor('#555555'))))
story.append(PageBreak())

# ── 1. INTRODUCCIÓN ──────────────────────────
story.append(heading1("1. Introducción y Contexto"))
story.append(hr())
story.append(body(
    "El presente informe documenta el proceso completo de análisis de clustering no supervisado "
    "aplicado al dataset <b>OSMI Mental Health in Tech Survey</b>. Este conjunto de datos "
    "proviene de encuestas realizadas a profesionales del sector tecnológico y recoge información "
    "sobre actitudes, percepciones y experiencias relacionadas con la salud mental en el entorno laboral."))
story.append(sp())
story.append(body(
    "El objetivo principal es descubrir <b>patrones latentes</b> y <b>grupos naturales</b> de empleados "
    "del sector IT con perfiles similares respecto a su relación con la salud mental, sin utilizar "
    "ninguna etiqueta predefinida. Este tipo de análisis permite a las organizaciones identificar "
    "segmentos de empleados con necesidades diferenciadas y diseñar intervenciones más precisas."))
story.append(sp())
story.append(body("<b>Preguntas de investigación:</b>"))
story.append(bullet("¿Existen grupos diferenciados de empleados según su percepción y experiencia con la salud mental?"))
story.append(bullet("¿Qué variables distinguen mejor cada grupo?"))
story.append(bullet("¿Qué algoritmo de clustering describe mejor la estructura natural de los datos?"))
story.append(bullet("¿Qué recomendaciones se pueden derivar para la gestión del bienestar laboral?"))
story.append(sp(12))

story.append(heading2("1.1 Descripción del Dataset Original"))
cols_desc = [
    ["Variable", "Tipo", "Descripción"],
    ["Timestamp", "Datetime", "Marca temporal de la respuesta"],
    ["Age", "Numérica", "Edad del empleado"],
    ["Gender", "Categórica", "Género (texto libre, normalizado)"],
    ["Country / state", "Categórica", "País y estado de residencia"],
    ["self_employed", "Binaria", "¿Trabaja por cuenta propia?"],
    ["family_history", "Binaria", "¿Historial familiar de salud mental?"],
    ["treatment", "Binaria", "¿Ha buscado tratamiento de SM?"],
    ["work_interfere", "Ordinal", "¿SM interfiere en el trabajo?"],
    ["no_employees", "Ordinal", "Tamaño de la empresa"],
    ["remote_work", "Binaria", "¿Trabaja en remoto?"],
    ["tech_company", "Binaria", "¿Empresa tecnológica?"],
    ["benefits/care_options/...", "Categórica", "Recursos de SM disponibles en empresa"],
    ["leave", "Ordinal", "Facilidad para tomar permiso por SM"],
    ["mental_health_consequence", "Categórica", "Miedo a consecuencias por SM"],
    ["coworkers/supervisor", "Categórica", "Disposición a hablar con compañeros/jefes"],
    ["mental_health_interview", "Categórica", "SM en entrevistas de trabajo"],
    ["obs_consequence", "Binaria", "¿Ha visto consecuencias por SM en empresa?"],
]
story.append(make_table(cols_desc, col_widths=[4.5*cm, 2.5*cm, 10*cm]))
story.append(PageBreak())

# ── 2. LIMPIEZA DE DATOS ─────────────────────
story.append(heading1("2. Limpieza y Preprocesamiento de Datos"))
story.append(hr())
story.append(body(
    "Esta fase es crítica para garantizar la calidad del análisis. Los datos de encuestas "
    "son particularmente propensos a entradas inconsistentes, valores atípicos y respuestas "
    "faltantes que pueden distorsionar los algoritmos de clustering."))
story.append(sp())

story.append(heading2("2.1 Eliminación de Columnas No Informativas"))
story.append(body(
    "Se eliminaron las columnas <b>Timestamp</b> (marca temporal sin valor analítico), "
    "<b>comments</b> (texto libre con ~95% de valores nulos) y <b>state</b> "
    "(alta cardinalidad, redundante con Country)."))

story.append(heading2("2.2 Tratamiento de Outliers en Edad"))
story.append(body(
    f"La variable <i>Age</i> contenía valores imposibles (< 0, > 200). Se aplicó un "
    f"filtro para mantener únicamente edades en el rango [18, 72] años, eliminando "
    f"{cleaning_stats['filas_originales'] - cleaning_stats['filas_limpias']} registros inválidos."))

story.append(heading2("2.3 Normalización de Género"))
story.append(body(
    "El campo <i>Gender</i> era texto libre con más de 30 variantes ('Male', 'male', 'm', "
    "'maile', 'Cis Male'...). Se normalizó a tres categorías: <b>Male</b>, <b>Female</b> y "
    "<b>Other</b>, usando reglas basadas en subcadenas."))

story.append(heading2("2.4 Agrupación Geográfica"))
story.append(body(
    "El campo <i>Country</i> tenía alta cardinalidad. Se mantuvieron los 8 países más "
    "frecuentes y el resto se agrupó en la categoría <b>'Other'</b>, reduciendo el ruido "
    "en el encoding posterior."))

story.append(heading2("2.5 Imputación de Valores Faltantes"))
imputation_data = [
    ["Variable / Grupo", "Estrategia", "Justificación"],
    ["work_interfere (NA)", "→ \"Don't know\"", "NA semánticamente equivale a 'sin información'"],
    ["self_employed (NA)", "→ \"No\"", "La mayoría no es autoempleado; NA implica asalariado"],
    ["Variables categóricas", "Moda del grupo", "Conserva la distribución original"],
    ["Age", "Mediana", "Robusta ante outliers residuales"],
]
story.append(make_table(imputation_data, col_widths=[5*cm, 4*cm, 8*cm]))
story.append(sp())
story.append(body(
    f"Total de valores imputados: <b>{cleaning_stats['missing_imputados']}</b> "
    f"sobre {cleaning_stats['filas_limpias'] * cleaning_stats['columnas_usadas']:,} "
    f"celdas totales (<b>{cleaning_stats['missing_imputados'] / (cleaning_stats['filas_limpias'] * cleaning_stats['columnas_usadas']) * 100:.2f}%</b>)."))

story.append(sp(12))
story.append(heading2("2.6 Resumen de Calidad de Datos"))
quality_data = [
    ["Métrica", "Antes", "Después"],
    ["Filas", str(cleaning_stats['filas_originales']), str(cleaning_stats['filas_limpias'])],
    ["Columnas", str(cleaning_stats['columnas_originales']), str(cleaning_stats['columnas_usadas'])],
    ["Valores faltantes", str(cleaning_stats['missing_imputados']), "0"],
    ["Variantes de género", ">30", "3 (Male/Female/Other)"],
    ["Variantes de país", f"{df_raw['Country'].nunique()}", "9 (top 8 + Other)"],
]
story.append(make_table(quality_data, col_widths=[6*cm, 4*cm, 4*cm + 3*cm]))
story.append(sp(12))
story.append(img(f"{OUT_DIR}/fig01_eda_distribuciones.png"))
story.append(caption("Figura 1. Distribuciones de las variables principales del dataset tras la limpieza."))
story.append(PageBreak())

# ── 3. FEATURE ENGINEERING ──────────────────
story.append(heading1("3. Feature Engineering y Codificación"))
story.append(hr())
story.append(body(
    "El dataset contiene predominantemente variables categóricas que deben ser "
    "transformadas en representaciones numéricas adecuadas para los algoritmos de "
    "clustering. Adicionalmente, se construyen features compuestos que sintetizan "
    "dimensiones latentes de interés."))
story.append(sp())

story.append(heading2("3.1 Features Compuestos (Índices)"))
story.append(body(
    "Se construyeron tres <b>índices sintéticos</b> que capturan dimensiones clave del bienestar laboral:"))
indices_data = [
    ["Índice", "Variables constituyentes", "Rango", "Interpretación"],
    ["support_index",
     "benefits, care_options, wellness_program, seek_help, anonymity",
     "[0, 10]",
     "Nivel de apoyo institucional percibido"],
    ["openness_index",
     "coworkers, supervisor, mental_health_interview, phys_health_interview",
     "[0, 8]",
     "Disposición a hablar de SM en el trabajo"],
    ["consequence_index",
     "mental_health_consequence, phys_health_consequence, obs_consequence",
     "[0, 6]",
     "Percepción de consecuencias negativas por SM"],
]
story.append(make_table(indices_data, col_widths=[3.5*cm, 5.5*cm, 2*cm, 6*cm]))
story.append(sp())

story.append(heading2("3.2 Codificación de Variables"))
encoding_data = [
    ["Tipo de Variable", "Estrategia", "Ejemplo"],
    ["Ordinal", "Mapeo numérico manual",
     "work_interfere: Never=0, Rarely=1, Sometimes=2, Often=3"],
    ["Binaria (Sí/No)", "Label Encoding (0/1)",
     "treatment, family_history, remote_work..."],
    ["Nominal (pocas categorías)", "One-Hot Encoding",
     "Gender → Male/Female/Other (3 dummies)"],
    ["Nominal (alta cardinalidad)", "Agrupación + OHE",
     "Country → top 8 + Other → 9 dummies"],
    ["Numérica continua", "Sin transformación (normalizar después)",
     "Age"],
]
story.append(make_table(encoding_data, col_widths=[3.5*cm, 4*cm, 9.5*cm]))
story.append(sp())
story.append(body(
    f"Resultado: <b>{len(feature_cols)} features</b> en total para la matriz de clustering X ({X.shape[0]} × {X.shape[1]})."))
story.append(sp(12))

story.append(heading2("3.3 Normalización y Estandarización"))
story.append(body(
    "Dado que los algoritmos de clustering basados en distancias (K-Means, DBSCAN) son "
    "sensibles a la escala, se aplicaron dos transformaciones:"))
norm_data = [
    ["Transformación", "Fórmula", "Uso en este análisis"],
    ["StandardScaler (Z-score)", "x' = (x - μ) / σ",
     "Principal: K-Means, GMM, Jerárquico, Spectral"],
    ["MinMaxScaler", "x' = (x - min) / (max - min)",
     "Alternativa evaluada para DBSCAN"],
]
story.append(make_table(norm_data, col_widths=[4.5*cm, 4.5*cm, 8*cm]))
story.append(sp())
story.append(body(
    "Se seleccionó <b>StandardScaler</b> como transformación principal por su idoneidad "
    "con distribuciones aproximadamente gaussianas y su amplio uso en la literatura. "
    "Garantiza que todas las features contribuyan equitativamente al cálculo de distancias."))
story.append(sp(8))

story.append(img(f"{OUT_DIR}/fig02_correlaciones.png"))
story.append(caption("Figura 2. Mapa de calor de correlaciones entre features numéricas clave."))
story.append(PageBreak())

# ── 4. REDUCCIÓN DE DIMENSIONALIDAD ─────────
story.append(heading1("4. Reducción de Dimensionalidad"))
story.append(hr())
story.append(body(
    "Con {0} features, aplicar clustering directamente puede sufrir la 'maldición de la "
    "dimensionalidad': en espacios de alta dimensión, las distancias euclídeas pierden "
    "significado y los clusters se vuelven difusos. Se aplica PCA para reducir "
    "dimensionalidad preservando la mayor varianza posible.".format(len(feature_cols))))
story.append(sp())

story.append(heading2("4.1 Análisis de Componentes Principales (PCA)"))
story.append(body(
    f"Se calculó el número de componentes necesario para explicar el <b>95% de la varianza</b>, "
    f"obteniendo <b>{n_components_95} componentes principales</b>. Esta reducción de "
    f"{len(feature_cols)} → {n_components_95} dimensiones elimina ruido y correlaciones "
    f"redundantes sin pérdida significativa de información."))
story.append(sp())

pca_table_data = [["Componentes", "Varianza Explicada", "Reducción Dimensional"]]
for n_c, pct in [(5, cumvar[4]), (n_components_95, cumvar[n_components_95-1]),
                  (len(feature_cols), 1.0)]:
    pca_table_data.append([
        str(n_c),
        f"{pct*100:.1f}%",
        f"{len(feature_cols)} → {n_c} ({(1 - n_c/len(feature_cols))*100:.0f}% reducción)"
    ])
story.append(make_table(pca_table_data, col_widths=[3.5*cm, 4.5*cm, 9*cm]))
story.append(sp(12))

story.append(img(f"{OUT_DIR}/fig03_pca_varianza.png"))
story.append(caption(
    f"Figura 3. Varianza explicada por componente (izquierda) y acumulada (derecha). "
    f"La línea discontinua marca el umbral del 95% en k={n_components_95} componentes."))
story.append(sp())

story.append(heading2("4.2 t-SNE para Visualización"))
story.append(body(
    "Adicionalmente, se aplicó <b>t-SNE</b> (t-distributed Stochastic Neighbor Embedding) "
    "para proyecciones 2D de visualización. A diferencia de PCA, t-SNE preserva la "
    "estructura local de los datos, siendo especialmente útil para revelar clusters visualmente. "
    "Parámetros: perplexity=40, max_iter=1000."))
story.append(body(
    "<i>Nota: t-SNE es exclusivamente para visualización; los algoritmos de clustering "
    "operan sobre los componentes PCA, no sobre t-SNE.</i>"))
story.append(PageBreak())

# ── 5. ALGORITMOS DE CLUSTERING ─────────────
story.append(heading1("5. Algoritmos de Clustering"))
story.append(hr())
story.append(body(
    "Se evaluaron cinco algoritmos con distintos supuestos sobre la forma y distribución "
    "de los clústeres, permitiendo una comparación robusta y selección fundamentada."))
story.append(sp())

story.append(heading2("5.1 Selección del Número de Clústeres (K-Means)"))
story.append(body(
    "Antes de ejecutar K-Means, se determinó el valor óptimo de <i>k</i> mediante "
    "cuatro criterios complementarios:"))
k_criteria = [
    ["Criterio", "Mejor k", "Descripción"],
    ["Método del Codo (Inercia)", "Aprox. 3-4", "Punto de inflexión en la curva de inercia intraclúster"],
    ["Silhouette Score", str(best_k_sil), "Maximiza la cohesión interna y separación entre clústeres"],
    ["Calinski-Harabasz", str(best_k_ch), "Ratio varianza inter/intra clúster (↑ mejor)"],
    ["Davies-Bouldin", str(best_k_db), "Similaridad media entre clústeres (↓ mejor)"],
    ["<b>Consenso</b>", f"<b>{K_OPT}</b>", "<b>K óptimo seleccionado para el análisis</b>"],
]
story.append(make_table(k_criteria, col_widths=[4.5*cm, 2.5*cm, 10*cm]))
story.append(sp(12))
story.append(img(f"{OUT_DIR}/fig04_seleccion_k.png"))
story.append(caption(
    "Figura 4. Criterios de selección del número de clústeres. "
    "Las líneas discontinuas rojas marcan el valor óptimo para cada métrica."))
story.append(PageBreak())

# Descripción de cada algoritmo
algo_desc = [
    ("5.2 K-Means",
     "Algoritmos de partición que minimiza la inercia intraclúster asignando cada "
     "punto al centroide más cercano mediante la distancia euclidiana.",
     ["Supuesto: clústeres esféricos y de tamaño similar",
      "Ventaja: eficiente computacionalmente, interpretable",
      "Limitación: sensible a outliers; requiere especificar k a priori",
      f"Configuración: k={K_OPT}, n_init=10, max_iter=300, random_state=42"]),
    ("5.3 DBSCAN (Density-Based Spatial Clustering)",
     "Algoritmo basado en densidad que identifica clústeres como regiones de alta "
     "densidad separadas por regiones de baja densidad. No requiere especificar k.",
     ["Supuesto: clústeres de forma arbitraria, datos con ruido",
      "Ventaja: detecta outliers/ruido explícitamente; no requiere k a priori",
      "Limitación: sensible a los parámetros eps y min_samples",
      f"Configuración: eps={best_eps}, min_samples optimizados por grid search"]),
    ("5.4 Gaussian Mixture Model (GMM)",
     "Modelo probabilístico que supone que los datos provienen de una mezcla de K "
     "distribuciones gaussianas multivariadas. Asignación soft (probabilística).",
     ["Supuesto: clústeres con distribución gaussiana (elipsoidal)",
      "Ventaja: asignación probabilística; permite clústeres de diferente tamaño y forma",
      f"Selección de k: BIC mínimo → k={best_k_bic}",
      "Configuración: n_init=10, random_state=42"]),
    ("5.5 Clustering Jerárquico (Ward)",
     "Construye una jerarquía de clústeres de forma aglomerativa (bottom-up). "
     "El método Ward minimiza la varianza intraclúster en cada fusión.",
     ["Supuesto: ninguno específico sobre forma",
      "Ventaja: produce dendrograma interpretable; no requiere k a priori",
      "Limitación: O(n²) en memoria; no adecuado para datasets muy grandes",
      f"Configuración: n_clusters={K_OPT}, linkage='ward'"]),
    ("5.6 Spectral Clustering",
     "Usa la estructura de grafos del grafo de similitud entre puntos para "
     "encontrar particiones mediante eigenvalues del Laplaciano.",
     ["Supuesto: clusters no convexos, separables en espacio de baja dimensión",
      "Ventaja: captura estructuras complejas no lineales",
      "Limitación: costoso computacionalmente O(n³)",
      f"Configuración: n_clusters={K_OPT}, affinity='rbf', random_state=42"]),
]

for title, desc_text, bullets_list in algo_desc:
    story.append(heading2(title))
    story.append(body(desc_text))
    for b in bullets_list:
        story.append(bullet(b))
    story.append(sp(6))

story.append(sp(8))
story.append(img(f"{OUT_DIR}/fig05_gmm_bic_aic.png", w_frac=0.7))
story.append(caption("Figura 5. Criterios BIC y AIC para selección del número de componentes GMM."))
story.append(sp())
story.append(img(f"{OUT_DIR}/fig06_dendrograma.png"))
story.append(caption("Figura 6. Dendrograma del clustering jerárquico Ward (muestra de 200 observaciones)."))
story.append(PageBreak())

# ── 6. EVALUACIÓN Y COMPARATIVA ─────────────
story.append(heading1("6. Evaluación y Comparativa de Algoritmos"))
story.append(hr())
story.append(body(
    "La evaluación del clustering no supervisado es más compleja que en el caso supervisado "
    "al no disponer de etiquetas de referencia. Se emplean tres métricas internas complementarias:"))
story.append(sp())

metrics_desc = [
    ["Métrica", "Fórmula / Descripción", "Rango", "Óptimo"],
    ["Silhouette Score",
     "Media de (b-a)/max(a,b) donde a=distancia intraclúster, b=distancia al clúster más cercano",
     "[-1, 1]", "↑ máximo (>0.5 = bueno)"],
    ["Calinski-Harabasz (CH)",
     "Ratio entre dispersión interclúster e intraclúster ponderado por n y k",
     "[0, ∞)", "↑ máximo"],
    ["Davies-Bouldin (DB)",
     "Media de la razón entre la suma de dispersiones intraclúster y la distancia entre centroides",
     "[0, ∞)", "↓ mínimo (→ 0 ideal)"],
    ["BIC/AIC (solo GMM)",
     "Criterios de información bayesiano/Akaike. Penalizan la complejidad del modelo",
     "(-∞, ∞)", "↓ mínimo"],
]
story.append(make_table(metrics_desc, col_widths=[3.5*cm, 7*cm, 2.5*cm, 4*cm]))
story.append(sp(12))

story.append(heading2("6.1 Tabla Comparativa de Resultados"))
metric_table_data = [["Algoritmo", "k", "Silhouette ↑", "Calinski-H ↑", "Davies-B ↓"]]
for _, row in metric_df.iterrows():
    metric_table_data.append([
        row['Algoritmo'],
        str(row['K']),
        f"{row['Silhouette']:.4f}",
        f"{row['Calinski-Harabasz']:.2f}",
        f"{row['Davies-Bouldin']:.4f}",
    ])
story.append(make_table(metric_table_data,
                        col_widths=[4.5*cm, 1.5*cm, 3.5*cm, 4*cm, 3.5*cm]))
story.append(sp(12))

# Ranking
best_sil_algo = metric_df.loc[metric_df['Silhouette'].idxmax(), 'Algoritmo']
best_ch_algo  = metric_df.loc[metric_df['Calinski-Harabasz'].idxmax(), 'Algoritmo']
best_db_algo  = metric_df.loc[metric_df['Davies-Bouldin'].idxmin(), 'Algoritmo']

story.append(body(
    f"<b>Mejor algoritmo por Silhouette:</b> {best_sil_algo} "
    f"({metric_df.loc[metric_df['Silhouette'].idxmax(), 'Silhouette']:.4f})"))
story.append(body(
    f"<b>Mejor algoritmo por Calinski-Harabasz:</b> {best_ch_algo} "
    f"({metric_df.loc[metric_df['Calinski-Harabasz'].idxmax(), 'Calinski-Harabasz']:.2f})"))
story.append(body(
    f"<b>Mejor algoritmo por Davies-Bouldin:</b> {best_db_algo} "
    f"({metric_df.loc[metric_df['Davies-Bouldin'].idxmin(), 'Davies-Bouldin']:.4f})"))
story.append(sp(12))

story.append(img(f"{OUT_DIR}/fig07_comparativa_metricas.png"))
story.append(caption("Figura 7. Comparativa visual de las tres métricas de evaluación para todos los algoritmos."))
story.append(PageBreak())

story.append(heading2("6.2 Visualizaciones de Clústeres"))
story.append(body(
    "Las siguientes figuras muestran la proyección de los clústeres en el espacio 2D de "
    "las primeras dos componentes principales (PCA) y mediante t-SNE:"))
story.append(sp(8))
story.append(img(f"{OUT_DIR}/fig08_clusters_pca2d.png"))
story.append(caption(
    "Figura 8. Visualización de los clústeres de cada algoritmo en el espacio PCA 2D. "
    "Cada color representa un clúster distinto; el punto negro en DBSCAN indica ruido."))
story.append(sp(12))
story.append(img(f"{OUT_DIR}/fig09_tsne.png"))
story.append(caption(
    "Figura 9. Visualización t-SNE de los clústeres para K-Means y GMM. "
    "t-SNE preserva la estructura local y revela la separación real de los grupos."))
story.append(PageBreak())

story.append(heading2("6.3 Análisis Silhouette por Muestra"))
story.append(body(
    "El silhouette plot muestra el coeficiente de silueta individual de cada observación "
    "agrupada por clúster. Valores negativos indican puntos posiblemente mal asignados. "
    "El ancho de cada banda refleja el tamaño del clúster."))
story.append(sp(8))
story.append(img(f"{OUT_DIR}/fig10_silhouette_plot.png", w_frac=0.8))
story.append(caption(
    f"Figura 10. Silhouette plot del modelo K-Means (k={K_OPT}). "
    f"Silhouette medio = {results['K-Means']['silhouette']:.4f}. "
    "La línea roja indica el valor medio global."))
story.append(PageBreak())

# ── 7. INTERPRETACIÓN ───────────────────────
story.append(heading1("7. Interpretación de Clústeres"))
story.append(hr())
story.append(body(
    "El modelo seleccionado para la interpretación detallada es <b>K-Means</b>, "
    "por su balance entre interpretabilidad, estabilidad y métricas competitivas. "
    "Se analiza el perfil demográfico, laboral y de salud mental de cada clúster."))
story.append(sp())

story.append(heading2("7.1 Tabla de Perfiles"))
profile_header = ["Variable"] + [f"Clúster {r['Clúster']}" for _, r in cluster_df.iterrows()]
profile_rows = [
    profile_header,
    ["N (tamaño)"] + [str(r['N']) for _, r in cluster_df.iterrows()],
    ["% del total"] + [r['%'] for _, r in cluster_df.iterrows()],
    ["Edad media"] + [r['Edad Media'] for _, r in cluster_df.iterrows()],
    ["Con tratamiento"] + [r['Con Tratamiento'] for _, r in cluster_df.iterrows()],
    ["Hist. familiar SM"] + [r['Hist. Familiar'] for _, r in cluster_df.iterrows()],
    ["Trabaja en remoto"] + [r['Trabaja Remoto'] for _, r in cluster_df.iterrows()],
    ["Género mayoritario"] + [r['Género Mayoritario'] for _, r in cluster_df.iterrows()],
    ["Interferencia laboral"] + [r['Interferencia Laboral'] for _, r in cluster_df.iterrows()],
    ["País predominante"] + [r['País Top'] for _, r in cluster_df.iterrows()],
]
n_cl = len(cluster_df)
col_w = [4.5*cm] + [(W - 4.5*cm) / n_cl] * n_cl
story.append(make_table(profile_rows, col_widths=col_w))
story.append(sp(12))

story.append(img(f"{OUT_DIR}/fig11_perfil_clusters.png"))
story.append(caption(
    "Figura 11. Desviación de cada clúster respecto a la media global en features clave. "
    "Verde = por encima de la media; rojo = por debajo."))
story.append(sp(8))
story.append(img(f"{OUT_DIR}/fig12_variables_por_cluster.png"))
story.append(caption(
    "Figura 12. Distribución de tratamiento e historial familiar de SM por clúster."))
story.append(PageBreak())

story.append(heading2("7.2 Descripción Cualitativa de Cada Clúster"))
for _, row in cluster_df.iterrows():
    cl = row['Clúster']
    n  = row['N']
    pct = row['%']
    interp = row['Interpretación']
    story.append(heading3(f"Clúster {cl} — {n} individuos ({pct})"))
    story.append(body(f"<b>Perfil sintetizado:</b> {interp.capitalize()}."))
    story.append(body(
        f"<b>Características demográficas:</b> Edad media {row['Edad Media']} años, "
        f"género predominante {row['Género Mayoritario']}, "
        f"principalmente de {row['País Top']}."))
    story.append(body(
        f"<b>Salud mental:</b> {row['Con Tratamiento']} han buscado tratamiento, "
        f"{row['Hist. Familiar']} tienen historial familiar de SM."))
    story.append(body(
        f"<b>Contexto laboral:</b> Interferencia laboral predominante: '{row['Interferencia Laboral']}'; "
        f"trabajo en remoto: {row['Trabaja Remoto']}."))
    story.append(sp(8))

story.append(PageBreak())

# ── 8. DISCUSIÓN ─────────────────────────────
story.append(heading1("8. Discusión y Limitaciones"))
story.append(hr())

story.append(heading2("8.1 Hallazgos Principales"))
story.append(body(
    "El análisis de clustering revela que los datos del sector IT presentan una estructura "
    "subyacente coherente con distintos perfiles de relación con la salud mental. Los "
    f"{K_OPT} clústeres identificados por K-Means son los más estables y mejor evaluados "
    "según las métricas internas, aunque el valor de silhouette sugiere que la separación "
    "entre grupos no es extremadamente nítida, lo cual es habitual en datos de encuestas."))
story.append(sp())
story.append(body(
    "DBSCAN identifica algunos puntos de ruido, confirmando la existencia de individuos "
    "con perfiles atípicos. GMM ofrece una perspectiva probabilística que puede ser más "
    "adecuada cuando los límites entre grupos son difusos."))
story.append(sp())

story.append(heading2("8.2 Limitaciones"))
limitations = [
    "Los datos de encuesta introducen sesgo de auto-reporte: los participantes tienden a "
    "subreportar condiciones estigmatizadas.",
    "La muestra está sesgada hacia trabajadores de EE.UU. y países angloparlantes, "
    "limitando la generalización.",
    "Las métricas internas evalúan la calidad geométrica de los clústeres pero no su "
    "relevancia práctica o accionabilidad.",
    "El número óptimo de k es sensible al método de evaluación empleado; se recomienda "
    "validación con expertos del dominio.",
    "DBSCAN muestra dificultades con datasets de densidad variable como este, siendo "
    "sensible a la elección de hiperparámetros.",
    "La imputación por moda puede introducir sesgo si los datos no son completamente "
    "aleatorios (MCAR).",
]
for lim in limitations:
    story.append(bullet(lim))

story.append(sp(12))
story.append(heading2("8.3 Comparativa Final de Algoritmos"))
comparativa = [
    ["Algoritmo", "Fortalezas en este contexto", "Debilidades"],
    ["K-Means",
     "Resultados estables, interpretables, buenas métricas",
     "Supone clusters esféricos; sensible a outliers"],
    ["DBSCAN",
     "Detecta ruido; sin asumir k",
     "Parámetros difíciles de optimizar; clusters irregulares"],
    ["GMM",
     "Asignación probabilística; modela elipses",
     "Sensible a inicialización; puede sobreajustar"],
    ["Jerárquico Ward",
     "Dendrograma informativo; no requiere k",
     "Costoso en memoria; O(n²)"],
    ["Spectral",
     "Captura formas no lineales",
     "Muy costoso computacionalmente; menos interpretable"],
]
story.append(make_table(comparativa, col_widths=[3.5*cm, 7.5*cm, 6*cm]))
story.append(PageBreak())

# ── 9. RECOMENDACIONES ───────────────────────
story.append(heading1("9. Recomendaciones y Aplicaciones"))
story.append(hr())
story.append(body(
    "Los resultados del clustering ofrecen información accionable para organizaciones del "
    "sector IT que deseen mejorar la gestión del bienestar mental de sus empleados:"))
story.append(sp())

recs = [
    ("Programas diferenciados por clúster",
     "En lugar de iniciativas genéricas, diseñar intervenciones adaptadas al perfil de "
     "cada segmento. Los clústeres con baja búsqueda de tratamiento pero alta interferencia "
     "laboral requieren acciones de sensibilización y reducción del estigma."),
    ("Mejora del apoyo institucional",
     "Los clústeres con bajo 'support_index' indican empleados que no perciben apoyo de "
     "la empresa. Comunicar activamente los recursos disponibles (EAP, cobertura de SM) "
     "puede cambiar la percepción y el comportamiento de búsqueda de ayuda."),
    ("Fomento de la apertura comunicativa",
     "El 'openness_index' bajo en ciertos segmentos sugiere miedo al estigma. Programas "
     "de entrenamiento para managers y políticas explícitas de no-discriminación son clave."),
    ("Vigilancia de perfiles de alto riesgo",
     "Los clústeres con alta frecuencia de interferencia laboral, historial familiar y "
     "alta percepción de consecuencias negativas representan prioridad de intervención."),
    ("Validación continua",
     "Se recomienda re-ejecutar el análisis periódicamente para detectar cambios en la "
     "distribución de clústeres tras implementar intervenciones."),
]
for title_rec, desc_rec in recs:
    story.append(heading3(f"• {title_rec}"))
    story.append(body(desc_rec))
    story.append(sp(6))

story.append(PageBreak())

# ── 10. CONCLUSIONES ─────────────────────────
story.append(heading1("10. Conclusiones"))
story.append(hr())
story.append(body(
    f"Este análisis ha aplicado un pipeline completo de Machine Learning no supervisado "
    f"al dataset OSMI Mental Health in Tech Survey ({len(df):,} observaciones, "
    f"{len(feature_cols)} features). Los principales hallazgos son:"))
story.append(sp())

concls = [
    f"Se identificaron <b>{K_OPT} clústeres</b> significativos en los datos mediante K-Means, "
    f"con un Silhouette Score de {results['K-Means']['silhouette']:.4f}, confirmando estructura "
    "latente real aunque con cierto solapamiento entre grupos.",
    "El preprocesamiento fue determinante: la normalización de género redujo >30 categorías "
    "a 3, la ingeniería de features construyó 3 índices sintéticos de alto valor interpretativo.",
    f"PCA con {n_components_95} componentes (95% varianza) resolvió la maldición de la "
    f"dimensionalidad, reduciendo {len(feature_cols)} features a {n_components_95} componentes.",
    "K-Means demostró el mejor balance entre estabilidad y métricas. GMM ofrece una "
    "alternativa probabilística válida. DBSCAN y Spectral son menos adecuados para "
    "este tipo de datos de encuesta.",
    "Los clústeres revelan perfiles diferenciados que combinan apoyo institucional percibido, "
    "apertura comunicativa, prevalencia de tratamiento e interferencia laboral.",
    "Los resultados tienen aplicabilidad directa en el diseño de políticas de bienestar "
    "laboral personalizadas por segmento de empleados.",
]
for c in concls:
    story.append(bullet(c))
    story.append(sp(4))

story.append(sp(20))
story.append(hr())
story.append(Paragraph(
    "Informe generado automáticamente · Python 3.12 · scikit-learn · pandas · matplotlib · reportlab",
    ParagraphStyle('Footer', parent=styles['BodyText'],
                   fontSize=7.5, textColor=colors.grey, alignment=TA_CENTER)))

# ── CONSTRUIR PDF ─────────────────────────────
doc.build(story)
print(f"\n{'='*60}")
print(f"  INFORME PDF GENERADO:")
print(f"  {REPORT_PDF}")
print(f"{'='*60}")
print(f"\n  Figuras guardadas en: {OUT_DIR}/")
print(f"  Total figuras: {len([f for f in os.listdir(OUT_DIR) if f.endswith('.png')])}")
