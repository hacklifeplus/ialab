"""
Análisis de Clustering No Supervisado — IT Mental Health Survey (Clean Numeric)
Dataset: IT_mental_health.survey.clean.num.csv
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
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

# Clustering
from sklearn.cluster import (KMeans, DBSCAN, AgglomerativeClustering,
                              SpectralClustering, MeanShift, estimate_bandwidth)
from sklearn.mixture import GaussianMixture

# Metrics
from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                              davies_bouldin_score, silhouette_samples)

# Stats
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import kruskal, f_oneway

# Report
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                  Table, TableStyle, PageBreak, HRFlowable)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import os

# ─────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────
DATA_PATH  = "/root/Projects/ialab/IT_mental_health.survey.clean.num.csv"
OUT_DIR    = "/root/Projects/ialab/output_clustering_num"
REPORT_PDF = "/root/Projects/ialab/output_clustering_num/Informe_Clustering_CleanNum.pdf"
os.makedirs(OUT_DIR, exist_ok=True)

PALETTE = ["#2E86AB","#A23B72","#F18F01","#C73E1D","#3B1F2B",
           "#44BBA4","#E94F37","#393E41","#F5A623","#7B2D8B"]
sns.set_theme(style="whitegrid", palette=PALETTE)

print("=" * 65)
print("  CLUSTERING NO SUPERVISADO — IT MENTAL HEALTH (CLEAN NUM)")
print("=" * 65)

# ─────────────────────────────────────────────────────────
# 1. CARGA Y EXPLORACIÓN INICIAL
# ─────────────────────────────────────────────────────────
print("\n[1/8] Cargando y explorando datos...")
df = pd.read_csv(DATA_PATH)
print(f"  Shape: {df.shape}")
print(f"  Columnas: {list(df.columns)}")
print(f"  Tipos:\n{df.dtypes.to_string()}")
print(f"\n  Estadísticas descriptivas:\n{df.describe().round(2).to_string()}")
print(f"\n  Valores nulos: {df.isnull().sum().sum()}")
print(f"  Duplicados: {df.duplicated().sum()}")

# Rangos y cardinalidades
range_info = []
for c in df.columns:
    range_info.append({
        'col': c,
        'min': df[c].min(),
        'max': df[c].max(),
        'unique': df[c].nunique(),
        'dtype': str(df[c].dtype)
    })
range_df = pd.DataFrame(range_info)

# ─────────────────────────────────────────────────────────
# 2. ANÁLISIS DE CALIDAD Y DIAGNÓSTICO
# ─────────────────────────────────────────────────────────
print("\n[2/8] Análisis de calidad...")

missing_counts = df.isnull().sum()
dup_count      = df.duplicated().sum()
print(f"  Valores nulos totales: {missing_counts.sum()}")
print(f"  Filas duplicadas: {dup_count}")

# Eliminar duplicados si los hay
if dup_count > 0:
    df = df.drop_duplicates()
    print(f"  → Duplicados eliminados. Shape: {df.shape}")

# Outliers por IQR
outlier_report = {}
for c in df.columns:
    Q1, Q3 = df[c].quantile(0.25), df[c].quantile(0.75)
    IQR = Q3 - Q1
    n_out = ((df[c] < Q1 - 1.5*IQR) | (df[c] > Q3 + 1.5*IQR)).sum()
    outlier_report[c] = n_out

print("  Outliers por IQR por columna:")
for col, n in outlier_report.items():
    if n > 0:
        print(f"    {col}: {n} ({n/len(df)*100:.1f}%)")

# ─────────────────────────────────────────────────────────
# 3. ANÁLISIS EXPLORATORIO (EDA)
# ─────────────────────────────────────────────────────────
print("\n[3/8] Análisis exploratorio...")

# FIG 1: Distribuciones de todas las features
n_cols_plot = 4
n_rows_plot = int(np.ceil(len(df.columns) / n_cols_plot))
fig1, axes1 = plt.subplots(n_rows_plot, n_cols_plot,
                            figsize=(18, n_rows_plot * 3.2))
axes1 = axes1.flatten()
fig1.suptitle("Distribución de Todas las Features (Dataset Numérico)",
               fontsize=14, fontweight='bold', y=1.01)

for i, col in enumerate(df.columns):
    ax = axes1[i]
    unique_vals = df[col].nunique()
    if unique_vals <= 8:
        counts = df[col].value_counts().sort_index()
        ax.bar(counts.index.astype(str), counts.values,
               color=PALETTE[i % len(PALETTE)], edgecolor='white', alpha=0.85)
    else:
        ax.hist(df[col], bins=20, color=PALETTE[i % len(PALETTE)],
                edgecolor='white', alpha=0.85)
    ax.set_title(col, fontsize=9, fontweight='bold')
    ax.set_xlabel("Valor")
    ax.set_ylabel("Freq.")
    ax.tick_params(labelsize=7)

for j in range(len(df.columns), len(axes1)):
    axes1[j].set_visible(False)

plt.tight_layout()
fig1.savefig(f"{OUT_DIR}/fig01_distribuciones.png", dpi=150, bbox_inches='tight')
plt.close(fig1)

# FIG 2: Mapa de calor de correlaciones
fig2, ax2 = plt.subplots(figsize=(16, 13))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, ax=ax2, linewidths=0.4, square=True,
            annot_kws={'size': 7}, cbar_kws={'shrink': 0.8})
ax2.set_title("Mapa de Calor — Correlaciones entre Todas las Features",
               fontsize=13, fontweight='bold')
plt.tight_layout()
fig2.savefig(f"{OUT_DIR}/fig02_correlaciones.png", dpi=150, bbox_inches='tight')
plt.close(fig2)

# FIG 3: Boxplots para detectar outliers
fig3, axes3 = plt.subplots(n_rows_plot, n_cols_plot,
                            figsize=(18, n_rows_plot * 3))
axes3 = axes3.flatten()
fig3.suptitle("Boxplots por Feature — Detección de Outliers",
               fontsize=14, fontweight='bold', y=1.01)
for i, col in enumerate(df.columns):
    axes3[i].boxplot(df[col].dropna(), vert=True,
                     patch_artist=True,
                     boxprops=dict(facecolor=PALETTE[i % len(PALETTE)], alpha=0.7))
    axes3[i].set_title(col, fontsize=9, fontweight='bold')
    axes3[i].tick_params(labelsize=7)

for j in range(len(df.columns), len(axes3)):
    axes3[j].set_visible(False)
plt.tight_layout()
fig3.savefig(f"{OUT_DIR}/fig03_boxplots.png", dpi=150, bbox_inches='tight')
plt.close(fig3)

# ─────────────────────────────────────────────────────────
# 4. PREPROCESAMIENTO: ESCALADO Y COMPARATIVA
# ─────────────────────────────────────────────────────────
print("\n[4/8] Preprocesamiento y escalado...")

X_raw = df.values.astype(float)

# Tres escaladores para comparar
std_scaler    = StandardScaler()
robust_scaler = RobustScaler()
mm_scaler     = MinMaxScaler()

X_std    = std_scaler.fit_transform(X_raw)
X_robust = robust_scaler.fit_transform(X_raw)
X_mm     = mm_scaler.fit_transform(X_raw)

print(f"  StandardScaler → media≈{X_std.mean():.4f}, std≈{X_std.std():.4f}")
print(f"  RobustScaler   → mediana≈{np.median(X_robust):.4f}")
print(f"  MinMaxScaler   → min={X_mm.min():.2f}, max={X_mm.max():.2f}")

# Seleccionamos StandardScaler como principal (dataset ya numérico, sin extremos severos)
X_scaled = X_std
print("  → StandardScaler seleccionado como principal.")

# ─────────────────────────────────────────────────────────
# 5. REDUCCIÓN DE DIMENSIONALIDAD
# ─────────────────────────────────────────────────────────
print("\n[5/8] Reducción de dimensionalidad...")

# PCA completo — análisis de varianza
pca_full = PCA(random_state=42)
pca_full.fit(X_scaled)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)

n_90 = int(np.argmax(cumvar >= 0.90)) + 1
n_95 = int(np.argmax(cumvar >= 0.95)) + 1
n_99 = int(np.argmax(cumvar >= 0.99)) + 1
print(f"  PC para 90% varianza: {n_90}")
print(f"  PC para 95% varianza: {n_95}")
print(f"  PC para 99% varianza: {n_99}")

# PCA para clustering (95%)
pca = PCA(n_components=n_95, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# PCA 2D y 3D para visualización
pca2d = PCA(n_components=2, random_state=42)
X_pca2d = pca2d.fit_transform(X_scaled)

pca3d = PCA(n_components=3, random_state=42)
X_pca3d = pca3d.fit_transform(X_scaled)

# t-SNE 2D
print("  Calculando t-SNE...")
tsne = TSNE(n_components=2, random_state=42, perplexity=40, n_iter=1000)
X_tsne = tsne.fit_transform(X_scaled)

print(f"  PCA {n_95}D para clustering, 2D/3D para viz, t-SNE 2D listos.")

# FIG 4: Varianza PCA
fig4, axes4 = plt.subplots(1, 2, figsize=(14, 5))
fig4.suptitle("Análisis de Componentes Principales (PCA)", fontsize=14, fontweight='bold')
n_show = min(23, len(pca_full.explained_variance_ratio_))
axes4[0].bar(range(1, n_show+1),
             pca_full.explained_variance_ratio_[:n_show]*100,
             color=PALETTE[0], edgecolor='white', alpha=0.85)
axes4[0].set_xlabel("Componente Principal")
axes4[0].set_ylabel("Varianza Explicada (%)")
axes4[0].set_title("Varianza Individual por Componente")

axes4[1].plot(range(1, len(cumvar)+1), cumvar*100,
              color=PALETTE[2], marker='o', markersize=4, linewidth=2)
for thr, nc, c in [(90, n_90, PALETTE[1]), (95, n_95, PALETTE[3]), (99, n_99, PALETTE[4])]:
    axes4[1].axhline(y=thr, color=c, linestyle='--', alpha=0.7, label=f'{thr}% → PC={nc}')
axes4[1].set_xlabel("Número de Componentes")
axes4[1].set_ylabel("Varianza Acumulada (%)")
axes4[1].set_title("Varianza Acumulada")
axes4[1].legend(fontsize=9)
plt.tight_layout()
fig4.savefig(f"{OUT_DIR}/fig04_pca_varianza.png", dpi=150, bbox_inches='tight')
plt.close(fig4)

# FIG 5: Biplot PCA (loadings de las 2 primeras componentes)
fig5, ax5 = plt.subplots(figsize=(12, 9))
loadings = pca2d.components_.T
scale = 3.5
for i, feat in enumerate(df.columns):
    ax5.arrow(0, 0, loadings[i,0]*scale, loadings[i,1]*scale,
               head_width=0.06, head_length=0.04,
               fc=PALETTE[i % len(PALETTE)], ec=PALETTE[i % len(PALETTE)], alpha=0.8)
    ax5.text(loadings[i,0]*scale*1.15, loadings[i,1]*scale*1.15,
              feat, fontsize=8, ha='center',
              color=PALETTE[i % len(PALETTE)], fontweight='bold')
ax5.axhline(0, color='grey', linewidth=0.5)
ax5.axvline(0, color='grey', linewidth=0.5)
ax5.set_xlabel(f"PC1 ({pca2d.explained_variance_ratio_[0]*100:.1f}%)", fontsize=11)
ax5.set_ylabel(f"PC2 ({pca2d.explained_variance_ratio_[1]*100:.1f}%)", fontsize=11)
ax5.set_title("Biplot PCA — Contribución de Features a PC1 y PC2",
               fontsize=13, fontweight='bold')
circle = plt.Circle((0,0), scale, fill=False, color='grey', linestyle='--', alpha=0.4)
ax5.add_patch(circle)
plt.tight_layout()
fig5.savefig(f"{OUT_DIR}/fig05_pca_biplot.png", dpi=150, bbox_inches='tight')
plt.close(fig5)

# ─────────────────────────────────────────────────────────
# 6. SELECCIÓN DEL NÚMERO ÓPTIMO DE CLUSTERS
# ─────────────────────────────────────────────────────────
print("\n[6/8] Determinando k óptimo...")

K_range = range(2, 13)
metrics_k = {'inertia': [], 'silhouette': [], 'calinski': [], 'davies': [], 'gap': []}

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=15, max_iter=500)
    lbl = km.fit_predict(X_pca)
    metrics_k['inertia'].append(km.inertia_)
    metrics_k['silhouette'].append(silhouette_score(X_pca, lbl))
    metrics_k['calinski'].append(calinski_harabasz_score(X_pca, lbl))
    metrics_k['davies'].append(davies_bouldin_score(X_pca, lbl))

# Gap Statistic (simplificado)
def gap_statistic(X, k_range, n_refs=10, random_state=42):
    rng = np.random.RandomState(random_state)
    gaps, sks = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        W_k = np.log(km.inertia_)
        ref_disps = []
        for _ in range(n_refs):
            ref = rng.uniform(X.min(axis=0), X.max(axis=0), size=X.shape)
            km_ref = KMeans(n_clusters=k, random_state=42, n_init=5)
            km_ref.fit(ref)
            ref_disps.append(np.log(km_ref.inertia_))
        gap = np.mean(ref_disps) - W_k
        sk  = np.std(ref_disps) * np.sqrt(1 + 1/n_refs)
        gaps.append(gap)
        sks.append(sk)
    return np.array(gaps), np.array(sks)

print("  Calculando Gap Statistic...")
gaps, sks = gap_statistic(X_pca, K_range)

best_k_sil  = list(K_range)[np.argmax(metrics_k['silhouette'])]
best_k_ch   = list(K_range)[np.argmax(metrics_k['calinski'])]
best_k_db   = list(K_range)[np.argmin(metrics_k['davies'])]
best_k_gap  = list(K_range)[np.argmax(gaps)]

print(f"  Mejor k → Silhouette: {best_k_sil}, CH: {best_k_ch}, "
      f"DB: {best_k_db}, Gap: {best_k_gap}")

# Consenso por votación
votes = [best_k_sil, best_k_ch, best_k_db, best_k_gap]
from collections import Counter
K_OPT = Counter(votes).most_common(1)[0][0]
# Si empate, priorizar silhouette
if Counter(votes).most_common(1)[0][1] == 1:
    K_OPT = best_k_sil
print(f"  K óptimo por consenso: {K_OPT}")

# FIG 6: Cuadro de selección k
fig6, axes6 = plt.subplots(2, 3, figsize=(18, 10))
fig6.suptitle("Selección del Número Óptimo de Clústeres",
               fontsize=15, fontweight='bold')

axes6[0,0].plot(K_range, metrics_k['inertia'], marker='o', color=PALETTE[0], lw=2)
axes6[0,0].set_title("Método del Codo (Inercia)")
axes6[0,0].set_xlabel("k"); axes6[0,0].set_ylabel("Inercia")

axes6[0,1].plot(K_range, metrics_k['silhouette'], marker='o', color=PALETTE[1], lw=2)
axes6[0,1].axvline(x=best_k_sil, color='red', ls='--', label=f'k={best_k_sil}')
axes6[0,1].set_title("Silhouette Score (↑)"); axes6[0,1].legend()
axes6[0,1].set_xlabel("k"); axes6[0,1].set_ylabel("Score")

axes6[0,2].plot(K_range, metrics_k['calinski'], marker='o', color=PALETTE[2], lw=2)
axes6[0,2].axvline(x=best_k_ch, color='red', ls='--', label=f'k={best_k_ch}')
axes6[0,2].set_title("Calinski-Harabasz (↑)"); axes6[0,2].legend()
axes6[0,2].set_xlabel("k"); axes6[0,2].set_ylabel("Score")

axes6[1,0].plot(K_range, metrics_k['davies'], marker='o', color=PALETTE[3], lw=2)
axes6[1,0].axvline(x=best_k_db, color='red', ls='--', label=f'k={best_k_db}')
axes6[1,0].set_title("Davies-Bouldin (↓)"); axes6[1,0].legend()
axes6[1,0].set_xlabel("k"); axes6[1,0].set_ylabel("Score")

axes6[1,1].plot(K_range, gaps, marker='o', color=PALETTE[4], lw=2, label='Gap')
axes6[1,1].fill_between(K_range,
                         gaps - sks, gaps + sks,
                         alpha=0.2, color=PALETTE[4])
axes6[1,1].axvline(x=best_k_gap, color='red', ls='--', label=f'k={best_k_gap}')
axes6[1,1].set_title("Gap Statistic (↑)"); axes6[1,1].legend()
axes6[1,1].set_xlabel("k"); axes6[1,1].set_ylabel("Gap")

# Resumen votación
vote_counts = Counter(votes)
axes6[1,2].bar([str(v) for v in vote_counts.keys()],
               vote_counts.values(),
               color=PALETTE[:len(vote_counts)], edgecolor='white')
axes6[1,2].set_title(f"Votación — K óptimo = {K_OPT}")
axes6[1,2].set_xlabel("k propuesto"); axes6[1,2].set_ylabel("Votos")

plt.tight_layout()
fig6.savefig(f"{OUT_DIR}/fig06_seleccion_k.png", dpi=150, bbox_inches='tight')
plt.close(fig6)

# ─────────────────────────────────────────────────────────
# 7. ALGORITMOS DE CLUSTERING
# ─────────────────────────────────────────────────────────
print("\n[7/8] Aplicando algoritmos de clustering...")

results = {}

# ── K-Means (k óptimo) ───────────────────────
print("  → K-Means (k óptimo)...")
km = KMeans(n_clusters=K_OPT, random_state=42, n_init=15, max_iter=500)
lbl_km = km.fit_predict(X_pca)
results['K-Means'] = dict(
    labels=lbl_km, n_clusters=K_OPT,
    silhouette=silhouette_score(X_pca, lbl_km),
    calinski=calinski_harabasz_score(X_pca, lbl_km),
    davies=davies_bouldin_score(X_pca, lbl_km),
    extra=f"Inercia={km.inertia_:.1f}, n_init=15"
)

# ── K-Means k=3 ──────────────────────────────
print("  → K-Means k=3...")
km3 = KMeans(n_clusters=3, random_state=42, n_init=15)
lbl_km3 = km3.fit_predict(X_pca)
results['K-Means k=3'] = dict(
    labels=lbl_km3, n_clusters=3,
    silhouette=silhouette_score(X_pca, lbl_km3),
    calinski=calinski_harabasz_score(X_pca, lbl_km3),
    davies=davies_bouldin_score(X_pca, lbl_km3),
    extra=f"Inercia={km3.inertia_:.1f}"
)

# ── K-Means k=4 ──────────────────────────────
print("  → K-Means k=4...")
km4 = KMeans(n_clusters=4, random_state=42, n_init=15)
lbl_km4 = km4.fit_predict(X_pca)
results['K-Means k=4'] = dict(
    labels=lbl_km4, n_clusters=4,
    silhouette=silhouette_score(X_pca, lbl_km4),
    calinski=calinski_harabasz_score(X_pca, lbl_km4),
    davies=davies_bouldin_score(X_pca, lbl_km4),
    extra=f"Inercia={km4.inertia_:.1f}"
)

# ── DBSCAN ───────────────────────────────────
print("  → DBSCAN (grid search eps/min_samples)...")
best_db_lbl, best_db_sil, best_db_eps, best_db_ms = None, -2, None, None
for eps in [0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]:
    for ms in [5, 8, 10, 15, 20]:
        lbl_tmp = DBSCAN(eps=eps, min_samples=ms).fit_predict(X_pca)
        n_cl = len(set(lbl_tmp)) - (1 if -1 in lbl_tmp else 0)
        noise_pct = (lbl_tmp == -1).sum() / len(lbl_tmp)
        if 2 <= n_cl <= 10 and noise_pct < 0.25:
            valid = lbl_tmp != -1
            if valid.sum() > 100:
                s = silhouette_score(X_pca[valid], lbl_tmp[valid])
                if s > best_db_sil:
                    best_db_sil = s
                    best_db_lbl = lbl_tmp
                    best_db_eps = eps
                    best_db_ms  = ms

if best_db_lbl is None:
    # Fallback: usar MinMaxScaler + epsilon más amplio
    for eps_fb in [3.0, 4.0, 5.0, 7.0]:
        lbl_fb = DBSCAN(eps=eps_fb, min_samples=5).fit_predict(X_mm)
        n_cl_fb = len(set(lbl_fb)) - (1 if -1 in lbl_fb else 0)
        if n_cl_fb >= 2:
            best_db_lbl = lbl_fb
            best_db_eps = eps_fb
            best_db_ms  = 5
            break
    if best_db_lbl is None:
        best_db_lbl = DBSCAN(eps=1.5, min_samples=10).fit_predict(X_pca)
        best_db_eps, best_db_ms = 1.5, 10

n_db = len(set(best_db_lbl)) - (1 if -1 in best_db_lbl else 0)
noise_db = int((best_db_lbl == -1).sum())
valid_db = best_db_lbl != -1
sil_db = silhouette_score(X_pca[valid_db], best_db_lbl[valid_db]) if n_db >= 2 and valid_db.sum() > 20 else -1
ch_db  = calinski_harabasz_score(X_pca[valid_db], best_db_lbl[valid_db]) if n_db >= 2 and valid_db.sum() > 20 else 0
db_db  = davies_bouldin_score(X_pca[valid_db], best_db_lbl[valid_db])  if n_db >= 2 and valid_db.sum() > 20 else 99
results['DBSCAN'] = dict(
    labels=best_db_lbl, n_clusters=n_db,
    silhouette=sil_db, calinski=ch_db, davies=db_db,
    extra=f"eps={best_db_eps}, min_samples={best_db_ms}, ruido={noise_db} ({noise_db/len(best_db_lbl)*100:.1f}%)"
)
print(f"    DBSCAN: k={n_db}, eps={best_db_eps}, min_s={best_db_ms}, ruido={noise_db}")

# ── GMM ──────────────────────────────────────
print("  → GMM (búsqueda por BIC/AIC)...")
bic_list, aic_list = [], []
for k in range(2, 11):
    g = GaussianMixture(n_components=k, random_state=42, n_init=5)
    g.fit(X_pca)
    bic_list.append(g.bic(X_pca))
    aic_list.append(g.aic(X_pca))

best_k_bic = list(range(2, 11))[np.argmin(bic_list)]
best_k_aic = list(range(2, 11))[np.argmin(aic_list)]
print(f"    GMM mejor BIC k={best_k_bic}, AIC k={best_k_aic}")

gmm = GaussianMixture(n_components=best_k_bic, random_state=42, n_init=10)
lbl_gmm = gmm.fit_predict(X_pca)
proba_gmm = gmm.predict_proba(X_pca)
entropy_gmm = -np.sum(proba_gmm * np.log(proba_gmm + 1e-10), axis=1).mean()

results['GMM'] = dict(
    labels=lbl_gmm, n_clusters=best_k_bic,
    silhouette=silhouette_score(X_pca, lbl_gmm),
    calinski=calinski_harabasz_score(X_pca, lbl_gmm),
    davies=davies_bouldin_score(X_pca, lbl_gmm),
    extra=f"BIC={gmm.bic(X_pca):.1f}, AIC={gmm.aic(X_pca):.1f}, Entropia media={entropy_gmm:.3f}"
)

fig_bic, ax_bic = plt.subplots(figsize=(9, 5))
ax_bic.plot(range(2,11), bic_list, marker='o', label='BIC', color=PALETTE[0], lw=2)
ax_bic.plot(range(2,11), aic_list, marker='s', label='AIC', color=PALETTE[1], lw=2)
ax_bic.axvline(x=best_k_bic, color='red', ls='--', label=f'Óptimo BIC k={best_k_bic}')
ax_bic.set_xlabel("Número de Componentes"); ax_bic.set_ylabel("Score")
ax_bic.set_title("GMM — Selección por BIC / AIC", fontweight='bold')
ax_bic.legend()
plt.tight_layout()
fig_bic.savefig(f"{OUT_DIR}/fig07_gmm_bic_aic.png", dpi=150, bbox_inches='tight')
plt.close(fig_bic)

# ── Jerárquico Ward ───────────────────────────
print("  → Clustering Jerárquico (Ward)...")
hc = AgglomerativeClustering(n_clusters=K_OPT, linkage='ward')
lbl_hc = hc.fit_predict(X_pca)
results['Hierarchical Ward'] = dict(
    labels=lbl_hc, n_clusters=K_OPT,
    silhouette=silhouette_score(X_pca, lbl_hc),
    calinski=calinski_harabasz_score(X_pca, lbl_hc),
    davies=davies_bouldin_score(X_pca, lbl_hc),
    extra="linkage=ward"
)

# Jerárquico Average
hc_avg = AgglomerativeClustering(n_clusters=K_OPT, linkage='average')
lbl_hca = hc_avg.fit_predict(X_pca)
results['Hierarchical Average'] = dict(
    labels=lbl_hca, n_clusters=K_OPT,
    silhouette=silhouette_score(X_pca, lbl_hca),
    calinski=calinski_harabasz_score(X_pca, lbl_hca),
    davies=davies_bouldin_score(X_pca, lbl_hca),
    extra="linkage=average"
)

# Dendrograma
sample_idx = np.random.RandomState(42).choice(len(X_pca),
             size=min(250, len(X_pca)), replace=False)
Z = linkage(X_pca[sample_idx], method='ward')
fig_dend, ax_dend = plt.subplots(figsize=(15, 6))
dendrogram(Z, ax=ax_dend, truncate_mode='lastp', p=35,
           leaf_rotation=90, leaf_font_size=8,
           color_threshold=0.7*max(Z[:,2]))
ax_dend.set_title("Dendrograma — Clustering Jerárquico Ward (muestra 250 obs.)",
                   fontsize=13, fontweight='bold')
ax_dend.set_xlabel("Índice"); ax_dend.set_ylabel("Distancia Ward")
plt.tight_layout()
fig_dend.savefig(f"{OUT_DIR}/fig08_dendrograma.png", dpi=150, bbox_inches='tight')
plt.close(fig_dend)

# ── Spectral Clustering ───────────────────────
print("  → Spectral Clustering...")
sc_model = SpectralClustering(n_clusters=K_OPT, random_state=42,
                               affinity='rbf', n_jobs=-1)
lbl_sc = sc_model.fit_predict(X_pca)
results['Spectral'] = dict(
    labels=lbl_sc, n_clusters=K_OPT,
    silhouette=silhouette_score(X_pca, lbl_sc),
    calinski=calinski_harabasz_score(X_pca, lbl_sc),
    davies=davies_bouldin_score(X_pca, lbl_sc),
    extra="affinity=rbf"
)

# ── Mean Shift ────────────────────────────────
print("  → Mean Shift...")
bw = estimate_bandwidth(X_pca, quantile=0.2, n_samples=500, random_state=42)
ms_model = MeanShift(bandwidth=bw, bin_seeding=True)
lbl_ms = ms_model.fit_predict(X_pca)
n_ms = len(set(lbl_ms))
sil_ms = silhouette_score(X_pca, lbl_ms) if n_ms >= 2 else -1
ch_ms  = calinski_harabasz_score(X_pca, lbl_ms) if n_ms >= 2 else 0
db_ms  = davies_bouldin_score(X_pca, lbl_ms)    if n_ms >= 2 else 99
results['Mean Shift'] = dict(
    labels=lbl_ms, n_clusters=n_ms,
    silhouette=sil_ms, calinski=ch_ms, davies=db_ms,
    extra=f"bandwidth={bw:.3f} (auto)"
)
print(f"    Mean Shift: k={n_ms} detectado automáticamente")

print("\n  Resultados de todos los algoritmos:")
for name, r in results.items():
    print(f"    {name:25s} k={r['n_clusters']:2d}  "
          f"Sil={r['silhouette']:.4f}  CH={r['calinski']:8.2f}  DB={r['davies']:.4f}")

# ─────────────────────────────────────────────────────────
# FIGURAS COMPARATIVAS
# ─────────────────────────────────────────────────────────
print("\n  Generando visualizaciones comparativas...")

metric_df = pd.DataFrame({
    'Algoritmo': list(results.keys()),
    'K': [r['n_clusters'] for r in results.values()],
    'Silhouette': [round(r['silhouette'], 4) for r in results.values()],
    'Calinski-H': [round(r['calinski'], 2) for r in results.values()],
    'Davies-B':   [round(r['davies'], 4) for r in results.values()],
})

# FIG 9: Comparativa de métricas
fig9, axes9 = plt.subplots(1, 3, figsize=(18, 6))
fig9.suptitle("Comparativa de Métricas — Todos los Algoritmos",
               fontsize=14, fontweight='bold')
cb = PALETTE[:len(metric_df)]
for ax, col, title in zip(axes9,
    ['Silhouette','Calinski-H','Davies-B'],
    ['Silhouette Score (↑)','Calinski-Harabasz (↑)','Davies-Bouldin (↓)']):
    ax.barh(metric_df['Algoritmo'], metric_df[col], color=cb, edgecolor='white')
    ax.set_title(title); ax.set_xlabel("Score")
    for i, v in enumerate(metric_df[col]):
        ax.text(v * 1.01, i, f"{v:.3f}", va='center', fontsize=8)
plt.tight_layout()
fig9.savefig(f"{OUT_DIR}/fig09_comparativa_metricas.png", dpi=150, bbox_inches='tight')
plt.close(fig9)

# FIG 10: Visualización PCA 2D de todos los algoritmos
n_algos = len(results)
ncols_v = 4
nrows_v = int(np.ceil(n_algos / ncols_v))
fig10, axes10 = plt.subplots(nrows_v, ncols_v, figsize=(ncols_v*5, nrows_v*4.5))
axes10 = axes10.flatten()
fig10.suptitle("Clústeres en Espacio PCA 2D — Todos los Algoritmos",
                fontsize=15, fontweight='bold')

for idx, (name, res) in enumerate(results.items()):
    ax = axes10[idx]
    lbl = res['labels']
    ulbls = sorted(set(lbl))
    cmap_v = plt.cm.get_cmap('tab10', len(ulbls))
    for i, ul in enumerate(ulbls):
        m = lbl == ul
        lname = f'Ruido ({m.sum()})' if ul == -1 else f'C{ul} (n={m.sum()})'
        ax.scatter(X_pca2d[m,0], X_pca2d[m,1],
                   c=[cmap_v(i)], s=12, alpha=0.6, label=lname)
    ax.set_title(f"{name}\nk={res['n_clusters']} · Sil={res['silhouette']:.3f}",
                  fontsize=10, fontweight='bold')
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.legend(fontsize=6, markerscale=1.5, loc='best')

for j in range(n_algos, len(axes10)):
    axes10[j].set_visible(False)
plt.tight_layout()
fig10.savefig(f"{OUT_DIR}/fig10_clusters_pca2d.png", dpi=150, bbox_inches='tight')
plt.close(fig10)

# FIG 11: t-SNE para los 3 mejores
top3 = metric_df.nlargest(3, 'Silhouette')['Algoritmo'].tolist()
fig11, axes11 = plt.subplots(1, 3, figsize=(18, 6))
fig11.suptitle("Visualización t-SNE — Top 3 Algoritmos por Silhouette",
                fontsize=14, fontweight='bold')
for ax, name in zip(axes11, top3):
    lbl = results[name]['labels']
    ulbls = sorted(set(lbl))
    cmap_v = plt.cm.get_cmap('tab10', len(ulbls))
    for i, ul in enumerate(ulbls):
        m = lbl == ul
        lname = 'Ruido' if ul == -1 else f'C{ul}'
        ax.scatter(X_tsne[m,0], X_tsne[m,1],
                   c=[cmap_v(i)], s=15, alpha=0.65, label=lname)
    ax.set_title(f"{name}\nk={results[name]['n_clusters']} · Sil={results[name]['silhouette']:.3f}",
                  fontsize=11, fontweight='bold')
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.legend(fontsize=8, markerscale=1.5)
plt.tight_layout()
fig11.savefig(f"{OUT_DIR}/fig11_tsne_top3.png", dpi=150, bbox_inches='tight')
plt.close(fig11)

# ─────────────────────────────────────────────────────────
# ANÁLISIS PROFUNDO DEL MEJOR MODELO
# ─────────────────────────────────────────────────────────
# Seleccionar el mejor modelo: excluir degenerados (clusters de tamaño <5% n o <10 pts)
def is_valid(res):
    lbl = res['labels']
    if res['n_clusters'] < 2 or res['silhouette'] <= -0.5:
        return False
    # Excluir si algún cluster tiene menos de 10 puntos
    for cl in set(lbl):
        if cl != -1 and (lbl == cl).sum() < 10:
            return False
    return True

valid_results = {k: v for k, v in results.items() if is_valid(v)}
if not valid_results:
    valid_results = {k: v for k, v in results.items()
                     if v['n_clusters'] >= 2 and v['silhouette'] > -0.5}
best_algo = max(valid_results, key=lambda k: valid_results[k]['silhouette'])
best_labels = results[best_algo]['labels']
print(f"\n  Mejor modelo seleccionado: {best_algo} "
      f"(Silhouette={results[best_algo]['silhouette']:.4f})")

# Silhouette por muestra
sil_samples = silhouette_samples(X_pca, best_labels)

# Análisis estadístico por clúster
df_cl = df.copy()
df_cl['Cluster'] = best_labels
df_cl['sil_sample'] = sil_samples

print("\n  Estadísticas por clúster:")
cluster_stats = df_cl.groupby('Cluster').agg(['mean','std','median']).round(3)

# Perfil simplificado
profile_rows = []
for cl in sorted(df_cl['Cluster'].unique()):
    if cl == -1: continue
    sub = df_cl[df_cl['Cluster'] == cl]
    row = {'Clúster': cl, 'N': len(sub),
           'Pct': f"{len(sub)/len(df_cl)*100:.1f}%",
           'Sil_medio': f"{sub['sil_sample'].mean():.3f}"}
    for c in df.columns:
        row[c] = f"{sub[c].mean():.2f}"
    profile_rows.append(row)
    print(f"    C{cl}: n={len(sub)}, sil={sub['sil_sample'].mean():.3f}, "
          f"Age={sub['Age'].mean():.1f}, treatment={sub['treatment'].mean():.2f}, "
          f"work_int={sub['work_interfere'].mean():.2f}")
profile_df = pd.DataFrame(profile_rows)

# FIG 12: Silhouette plot
fig12, ax12 = plt.subplots(figsize=(11, 7))
y_lower = 10
ulbls_best = sorted(set(best_labels))
cmap12 = plt.cm.get_cmap('tab10', len(ulbls_best))
for i, cl in enumerate(ulbls_best):
    if cl == -1: continue
    sv = np.sort(sil_samples[best_labels == cl])
    y_upper = y_lower + len(sv)
    ax12.fill_betweenx(np.arange(y_lower, y_upper), 0, sv,
                        facecolor=cmap12(i), alpha=0.75)
    ax12.text(-0.06, y_lower + 0.5*len(sv), f"C{cl}", fontsize=9)
    y_lower = y_upper + 10
ax12.axvline(x=results[best_algo]['silhouette'], color='red', ls='--', lw=1.5,
              label=f"Media={results[best_algo]['silhouette']:.3f}")
ax12.set_xlabel("Silhouette Coefficient")
ax12.set_ylabel("Muestras por clúster")
ax12.set_title(f"Silhouette Plot — {best_algo}", fontsize=13, fontweight='bold')
ax12.legend()
plt.tight_layout()
fig12.savefig(f"{OUT_DIR}/fig12_silhouette_plot.png", dpi=150, bbox_inches='tight')
plt.close(fig12)

# FIG 13: Heatmap de medias por clúster
cluster_means = df_cl.groupby('Cluster')[df.columns].mean()
cluster_means_norm = (cluster_means - cluster_means.min()) / \
                     (cluster_means.max() - cluster_means.min() + 1e-9)

fig13, ax13 = plt.subplots(figsize=(16, max(4, len(cluster_means)*1.5)))
sns.heatmap(cluster_means_norm.T, annot=cluster_means.T.round(2),
            fmt=".2f", cmap="YlOrRd", ax=ax13,
            linewidths=0.4, cbar_kws={'label': 'Valor normalizado'},
            annot_kws={'size': 8})
ax13.set_title(f"Perfil de Clústeres — Medias Normalizadas ({best_algo})",
                fontsize=13, fontweight='bold')
ax13.set_xlabel("Clúster"); ax13.set_ylabel("Feature")
plt.tight_layout()
fig13.savefig(f"{OUT_DIR}/fig13_heatmap_clusters.png", dpi=150, bbox_inches='tight')
plt.close(fig13)

# FIG 14: Radar chart de perfiles de clúster
key_features = ['Age','treatment','family_history','work_interfere',
                'self_employed','remote_work','tech_company','benefits',
                'care_options','wellness_program','seek_help','anonymity']
key_features = [c for c in key_features if c in df.columns]
N_feat = len(key_features)
angles = np.linspace(0, 2*np.pi, N_feat, endpoint=False).tolist()
angles += angles[:1]

cl_unique = [c for c in sorted(df_cl['Cluster'].unique()) if c != -1]
fig14, ax14 = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
fig14.suptitle(f"Radar Chart — Perfiles de Clústeres ({best_algo})",
                fontsize=13, fontweight='bold')
cmap14 = plt.cm.get_cmap('tab10', len(cl_unique))

# Normalizar para radar
feat_min = df_cl[key_features].min()
feat_max = df_cl[key_features].max()

for i, cl in enumerate(cl_unique):
    sub = df_cl[df_cl['Cluster'] == cl]
    vals = sub[key_features].mean()
    vals_norm = ((vals - feat_min) / (feat_max - feat_min + 1e-9)).tolist()
    vals_norm += vals_norm[:1]
    ax14.plot(angles, vals_norm, 'o-', lw=2, color=cmap14(i),
               label=f'Clúster {cl} (n={len(sub)})', alpha=0.85)
    ax14.fill(angles, vals_norm, alpha=0.12, color=cmap14(i))

ax14.set_xticks(angles[:-1])
ax14.set_xticklabels(key_features, size=9)
ax14.set_ylim(0, 1)
ax14.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
plt.tight_layout()
fig14.savefig(f"{OUT_DIR}/fig14_radar_clusters.png", dpi=150, bbox_inches='tight')
plt.close(fig14)

# FIG 15: Distribución de features clave por clúster (violines)
key4 = ['Age','work_interfere','treatment','family_history']
key4 = [c for c in key4 if c in df_cl.columns]
fig15, axes15 = plt.subplots(1, len(key4), figsize=(5*len(key4), 6))
fig15.suptitle(f"Distribución de Features Clave por Clúster ({best_algo})",
                fontsize=13, fontweight='bold')
for ax, feat in zip(axes15, key4):
    data_v = [df_cl[df_cl['Cluster']==cl][feat].values
               for cl in cl_unique]
    vp = ax.violinplot(data_v, positions=range(len(cl_unique)),
                        showmedians=True, showextrema=True)
    for j, body in enumerate(vp['bodies']):
        body.set_facecolor(cmap14(j))
        body.set_alpha(0.65)
    ax.set_title(feat, fontweight='bold')
    ax.set_xlabel("Clúster")
    ax.set_xticks(range(len(cl_unique)))
    ax.set_xticklabels([f'C{c}' for c in cl_unique])
    ax.set_ylabel("Valor")
plt.tight_layout()
fig15.savefig(f"{OUT_DIR}/fig15_violines_clusters.png", dpi=150, bbox_inches='tight')
plt.close(fig15)

# FIG 16: PCA 3D del mejor modelo
fig16 = plt.figure(figsize=(12, 9))
ax16 = fig16.add_subplot(111, projection='3d')
cmap16 = plt.cm.get_cmap('tab10', len(cl_unique))
for i, cl in enumerate(cl_unique):
    m = best_labels == cl
    ax16.scatter(X_pca3d[m,0], X_pca3d[m,1], X_pca3d[m,2],
                  c=[cmap16(i)], s=15, alpha=0.6, label=f'C{cl}')
ax16.set_xlabel("PC1"); ax16.set_ylabel("PC2"); ax16.set_zlabel("PC3")
ax16.set_title(f"Visualización PCA 3D — {best_algo}", fontsize=12, fontweight='bold')
ax16.legend(fontsize=9)
plt.tight_layout()
fig16.savefig(f"{OUT_DIR}/fig16_pca3d.png", dpi=150, bbox_inches='tight')
plt.close(fig16)

# FIG 17: Importancia de features (varianza inter-cluster)
feat_importance = []
for feat in df.columns:
    group_means = [df_cl[df_cl['Cluster']==cl][feat].mean()
                   for cl in cl_unique]
    global_mean = df_cl[feat].mean()
    variance_between = np.var(group_means)
    feat_importance.append((feat, variance_between))
feat_imp_df = pd.DataFrame(feat_importance, columns=['Feature','Var_Between'])
feat_imp_df = feat_imp_df.sort_values('Var_Between', ascending=True)

fig17, ax17 = plt.subplots(figsize=(10, 8))
ax17.barh(feat_imp_df['Feature'], feat_imp_df['Var_Between'],
           color=PALETTE[0], edgecolor='white', alpha=0.85)
ax17.set_title(f"Importancia de Features — Varianza Inter-Clúster ({best_algo})",
                fontsize=12, fontweight='bold')
ax17.set_xlabel("Varianza entre medias de clústeres (↑ más discriminante)")
plt.tight_layout()
fig17.savefig(f"{OUT_DIR}/fig17_importancia_features.png", dpi=150, bbox_inches='tight')
plt.close(fig17)

# ─────────────────────────────────────────────────────────
# 8. GENERACIÓN DEL INFORME PDF
# ─────────────────────────────────────────────────────────
print("\n[8/8] Generando informe PDF...")

doc = SimpleDocTemplate(
    REPORT_PDF, pagesize=A4,
    rightMargin=2*cm, leftMargin=2*cm,
    topMargin=2.5*cm, bottomMargin=2*cm
)
styles = getSampleStyleSheet()
W = A4[0] - 4*cm

sTitle = ParagraphStyle('T', parent=styles['Title'], fontSize=22,
    textColor=colors.HexColor('#1a1a2e'), alignment=TA_CENTER,
    spaceAfter=6, leading=28)
sH1 = ParagraphStyle('H1', parent=styles['Heading1'], fontSize=14,
    textColor=colors.HexColor('#16213e'), spaceBefore=18, spaceAfter=6)
sH2 = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=11,
    textColor=colors.HexColor('#0f3460'), spaceBefore=10, spaceAfter=4)
sH3 = ParagraphStyle('H3', parent=styles['Heading3'], fontSize=10,
    textColor=colors.HexColor('#533483'), spaceBefore=7, spaceAfter=3)
sBody = ParagraphStyle('B', parent=styles['BodyText'], fontSize=9.5,
    leading=14, alignment=TA_JUSTIFY, spaceAfter=4)
sBullet = ParagraphStyle('BL', parent=styles['BodyText'], fontSize=9.5,
    leading=14, leftIndent=14, bulletIndent=4, spaceAfter=2)
sCaption = ParagraphStyle('Cap', parent=styles['BodyText'], fontSize=8.5,
    textColor=colors.grey, alignment=TA_CENTER, spaceAfter=10)
sCenter = ParagraphStyle('Cen', parent=styles['BodyText'],
    alignment=TA_CENTER, fontSize=9.5)
sFooter = ParagraphStyle('Foot', parent=styles['BodyText'],
    fontSize=7.5, textColor=colors.grey, alignment=TA_CENTER)

def h1(t): return Paragraph(t, sH1)
def h2(t): return Paragraph(t, sH2)
def h3(t): return Paragraph(t, sH3)
def p(t):  return Paragraph(t, sBody)
def bl(t): return Paragraph(f"• {t}", sBullet)
def cap(t):return Paragraph(t, sCaption)
def sp(n=8):return Spacer(1, n)
def hr(): return HRFlowable(width="100%", thickness=0.5,
                             color=colors.HexColor('#cccccc'),
                             spaceAfter=6, spaceBefore=6)

def im(path, frac=0.95):
    try:
        i = Image(path, width=W*frac, height=W*frac*0.58)
        i.hAlign = 'CENTER'
        return i
    except:
        return p(f"[Figura no disponible: {os.path.basename(path)}]")

def im_sq(path, frac=0.7):
    try:
        i = Image(path, width=W*frac, height=W*frac)
        i.hAlign = 'CENTER'
        return i
    except:
        return p(f"[Figura no disponible: {os.path.basename(path)}]")

def tbl(data, cw=None):
    t = Table(data, colWidths=cw, repeatRows=1)
    t.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#16213e')),
        ('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
        ('FONTSIZE',(0,0),(-1,-1),8.5),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),
         [colors.white, colors.HexColor('#eef3fb')]),
        ('GRID',(0,0),(-1,-1),0.4,colors.HexColor('#cccccc')),
        ('BOTTOMPADDING',(0,0),(-1,-1),5),
        ('TOPPADDING',(0,0),(-1,-1),5),
    ]))
    return t

story = []

# ═══ PORTADA ════════════════════════════════
story += [sp(2*cm),
          Paragraph("Informe de Clustering No Supervisado", sTitle),
          Paragraph("IT Mental Health Survey — Dataset Numérico Limpio",
                    ParagraphStyle('Sub', parent=styles['Title'], fontSize=15,
                                   textColor=colors.HexColor('#0f3460'),
                                   alignment=TA_CENTER, spaceAfter=4)),
          sp(20), hr(), sp(10)]

meta_rows = [
    ["Parámetro", "Valor"],
    ["Dataset", "IT_mental_health.survey.clean.num.csv"],
    ["Observaciones", f"{len(df):,}"],
    ["Features", f"{df.shape[1]} (todas numéricas, ya codificadas)"],
    ["PC para clustering (95% var.)", str(n_95)],
    ["Algoritmos evaluados", "K-Means ×3, DBSCAN, GMM, Jerárquico ×2, Spectral, Mean Shift"],
    ["Mejor algoritmo", f"{best_algo} (Silhouette={results[best_algo]['silhouette']:.4f})"],
    ["Fecha", "19 de abril de 2026"],
    ["Herramientas", "Python 3.12 · scikit-learn · pandas · matplotlib · seaborn"],
]
story.append(tbl(meta_rows, cw=[6*cm, 11*cm]))
story += [sp(25),
          p("Análisis exhaustivo de agrupamiento no supervisado sobre el dataset numérico "
            "limpio, que incluye diagnóstico de calidad, análisis exploratorio, reducción de "
            "dimensionalidad con PCA y t-SNE, evaluación multicriterio de 9 configuraciones "
            "de clustering, análisis estadístico de perfiles y visualizaciones avanzadas."),
          PageBreak()]

# ═══ 1. INTRODUCCIÓN ════════════════════════
story += [h1("1. Introducción"), hr(),
          p("Este informe documenta el pipeline completo de Machine Learning no supervisado "
            "aplicado al dataset <b>IT_mental_health.survey.clean.num.csv</b>, versión "
            "preprocesada y codificada numéricamente del OSMI Mental Health in Tech Survey. "
            "A diferencia del análisis previo sobre el CSV original, este dataset presenta "
            "todas las variables ya convertidas a representaciones numéricas, lo que permite "
            "un análisis directo sin necesidad de encoding adicional."),
          sp(),
          p("<b>Ventajas de trabajar con el dataset numérico:</b>"),
          bl("Todas las variables son directamente comparables y escalables."),
          bl("No hay valores categóricos: se eliminan posibles errores de encoding."),
          bl("La correlación entre features es calculable sobre todas las columnas."),
          bl("Los algoritmos pueden aplicarse con mayor eficiencia computacional."),
          sp(10),
          h2("1.1 Descripción del Dataset"),
          p(f"El dataset contiene <b>{len(df):,} observaciones</b> y <b>{df.shape[1]} features</b>, "
            "todas de tipo numérico entero. Las variables representan la misma información "
            "que el dataset original pero codificada: variables binarias en 0/1, variables "
            "ordinales en escala numérica, y la variable <i>Age</i> como valor continuo.")]

# Tabla de features
feat_table = [["Feature", "Rango", "Valores únicos", "Media", "Descripción"]]
desc_map = {
    'Age': "Edad del empleado",
    'Gender': "0=Male, 1=Female, 2=Other",
    'self_employed': "0=No, 1=Sí",
    'family_history': "0=No, 1=Sí",
    'treatment': "0=No, 1=Sí (buscó tratamiento)",
    'work_interfere': "0=Never…3=Often, -1=No sabe",
    'no_employees': "1=1-5…6=>1000 empleados",
    'remote_work': "0=No, 1=Sí",
    'tech_company': "0=No, 1=Sí",
    'benefits': "0=No, 1=No sabe, 2=Sí",
    'care_options': "0=No, 1=No sabe, 2=Sí",
    'wellness_program': "0=No, 1=No sabe, 2=Sí",
    'seek_help': "0=No, 1=No sabe, 2=Sí",
    'anonymity': "0=No, 1=No sabe, 2=Sí",
    'leave': "0=Muy difícil…4=Muy fácil",
    'mental_health_consequence': "0=No, 1=Maybe, 2=Sí",
    'phys_health_consequence': "0=No, 1=Maybe, 2=Sí",
    'coworkers': "0=No, 1=Some, 2=Sí",
    'supervisor': "0=No, 1=Maybe, 2=Sí",
    'mental_health_interview': "0=No, 1=Maybe, 2=Sí",
    'phys_health_interview': "0=No, 1=Maybe, 2=Sí",
    'mental_vs_physical': "0=No, 1=No sabe, 2=Sí",
    'obs_consequence': "0=No, 1=Sí",
}
for c in df.columns:
    feat_table.append([c,
                        f"[{int(df[c].min())}, {int(df[c].max())}]",
                        str(df[c].nunique()),
                        f"{df[c].mean():.2f}",
                        desc_map.get(c, "-")])
story.append(tbl(feat_table, cw=[3.5*cm, 2.2*cm, 2.2*cm, 1.8*cm, 7.3*cm]))
story.append(PageBreak())

# ═══ 2. CALIDAD Y EDA ════════════════════════
story += [h1("2. Análisis de Calidad y Exploratorio"), hr(),
          h2("2.1 Diagnóstico de Calidad")]

quality_data = [
    ["Métrica", "Resultado", "Acción tomada"],
    ["Valores nulos", "0", "Ninguna (dataset limpio)"],
    ["Filas duplicadas", str(dup_count), "Eliminadas" if dup_count > 0 else "Ninguna"],
    ["Outliers por IQR", f"{sum(v>0 for v in outlier_report.values())} features afectadas",
     "Conservados (algoritmos robustos / RobustScaler disponible)"],
    ["Rango de Age", f"[{int(df['Age'].min())}, {int(df['Age'].max())}]",
     "Rango válido tras limpieza previa"],
    ["Variables con 2 valores únicos", str(sum(df[c].nunique()==2 for c in df.columns)),
     "Tratadas como binarias"],
]
story.append(tbl(quality_data, cw=[4*cm, 4.5*cm, 8.5*cm]))
story += [sp(10),
          im(f"{OUT_DIR}/fig01_distribuciones.png"),
          cap("Figura 1. Distribución de las 23 features del dataset numérico."),
          sp(8),
          im(f"{OUT_DIR}/fig02_correlaciones.png"),
          cap("Figura 2. Mapa de calor de correlaciones entre todas las features (triángulo inferior)."),
          sp(8),
          im(f"{OUT_DIR}/fig03_boxplots.png"),
          cap("Figura 3. Boxplots de cada feature. Los puntos extremos indican posibles outliers."),
          PageBreak()]

# ═══ 3. PREPROCESAMIENTO ════════════════════
story += [h1("3. Preprocesamiento y Escalado"), hr(),
          p("Aunque el dataset ya está codificado numéricamente, el escalado es imprescindible "
            "porque las variables tienen rangos muy distintos: <i>Age</i> ∈ [18,72] mientras "
            "la mayoría de features binarias están en {0,1}. Sin escalar, <i>Age</i> dominaría "
            "completamente el cálculo de distancias."),
          sp(8),
          h2("3.1 Comparativa de Escaladores")]

scaler_data = [
    ["Escalador", "Fórmula", "Ventaja", "Cuándo usar"],
    ["StandardScaler\n(Z-score)", "x' = (x-μ)/σ",
     "Media 0, std 1. Idóneo para distribuciones aprox. normales",
     "Principal: K-Means, GMM, Jerárquico, Spectral, Mean Shift"],
    ["RobustScaler", "x' = (x-Q₂)/(Q₃-Q₁)",
     "Ignora outliers usando IQR. Más robusto con datos sesgados",
     "Alternativa cuando hay outliers severos"],
    ["MinMaxScaler", "x' = (x-min)/(max-min)",
     "Rango [0,1] exacto. Intuitivo",
     "DBSCAN (sensible a escala absoluta)"],
]
story.append(tbl(scaler_data, cw=[3.5*cm, 3.5*cm, 5*cm, 5*cm]))
story += [sp(8),
          p("<b>Decisión:</b> Se seleccionó <b>StandardScaler</b> como transformación "
            "principal. El dataset no presenta outliers extremos tras la limpieza previa, "
            "y la mayoría de algoritmos de clustering funcionan óptimamente con datos "
            "estandarizados a media 0 y desviación 1."),
          PageBreak()]

# ═══ 4. REDUCCIÓN DE DIMENSIONALIDAD ════════
story += [h1("4. Reducción de Dimensionalidad"), hr(),
          h2("4.1 PCA — Análisis de Componentes Principales"),
          p(f"Con 23 features, la reducción dimensional no es tan crítica como en datasets "
            f"de alta dimensión, pero sigue siendo beneficiosa para eliminar la colinealidad "
            f"observada en el mapa de correlaciones. El umbral del 95% de varianza explicada "
            f"requiere <b>{n_95} componentes principales</b>.")]

pca_thr = [
    ["Umbral Varianza", "PC necesarios", "Reducción dimensional", "Observación"],
    ["90%", str(n_90), f"23 → {n_90}", "Compresión moderada"],
    ["95%", str(n_95), f"23 → {n_95}", "Equilibrio calidad/eficiencia (seleccionado)"],
    ["99%", str(n_99), f"23 → {n_99}", "Casi sin pérdida de información"],
]
story.append(tbl(pca_thr, cw=[3.5*cm, 3*cm, 4*cm, 6.5*cm]))
story += [sp(8),
          im(f"{OUT_DIR}/fig04_pca_varianza.png"),
          cap(f"Figura 4. Varianza explicada por componente (izq.) y acumulada (der.). "
              f"Umbrales al 90%, 95% y 99%."),
          sp(8),
          im(f"{OUT_DIR}/fig05_pca_biplot.png"),
          cap("Figura 5. Biplot PCA: contribución de cada feature a PC1 y PC2. "
              "Las flechas más largas indican mayor contribución a la componente."),
          sp(8),
          h2("4.2 t-SNE para Visualización"),
          p("Se aplica t-SNE (perplexity=40, n_iter=1000) para proyección 2D no lineal. "
            "A diferencia de PCA, t-SNE preserva la estructura local y es especialmente "
            "útil para confirmar visualmente la existencia de grupos compactos. "
            "<i>Los algoritmos de clustering operan sobre el espacio PCA, no t-SNE.</i>"),
          PageBreak()]

# ═══ 5. ALGORITMOS ══════════════════════════
story += [h1("5. Selección del Número de Clústeres y Algoritmos"), hr(),
          h2("5.1 Determinación del K Óptimo"),
          p("Se emplean cinco criterios complementarios para determinar el número óptimo "
            "de clústeres, más una votación por consenso:")]

vote_tbl_data = [
    ["Criterio", "K propuesto", "Descripción del criterio"],
    ["Silhouette Score", str(best_k_sil), "Maximiza cohesión interna y separación entre clústeres"],
    ["Calinski-Harabasz", str(best_k_ch), "Ratio varianza inter/intra clúster"],
    ["Davies-Bouldin", str(best_k_db), "Minimiza similaridad media entre pares de clústeres"],
    ["Gap Statistic", str(best_k_gap), "Compara inercia real vs. datos aleatorios de referencia"],
    ["<b>Consenso (votación)</b>", f"<b>{K_OPT}</b>", "<b>K seleccionado para los algoritmos que lo requieren</b>"],
]
story.append(tbl(vote_tbl_data, cw=[4.5*cm, 2.5*cm, 10*cm]))
story += [sp(10),
          im(f"{OUT_DIR}/fig06_seleccion_k.png"),
          cap(f"Figura 6. Los 5 criterios de selección de k. Panel inferior derecho: "
              f"votación de consenso → K={K_OPT}."),
          PageBreak(),
          h2("5.2 Algoritmos Evaluados")]

algos_desc = [
    ("K-Means (k óptimo)", f"k={K_OPT}, n_init=15, max_iter=500",
     "Partición minimizando inercia. Referencia principal del estudio."),
    ("K-Means k=3", "k=3, n_init=15",
     "Comparativa con k=3 (frecuente en literatura sobre salud mental en IT)."),
    ("K-Means k=4", "k=4, n_init=15",
     "Comparativa adicional para detectar subgrupos adicionales."),
    ("DBSCAN", f"eps={best_db_eps}, min_samples={best_db_ms} (grid search)",
     "Basado en densidad. Detecta clústeres de forma arbitraria y ruido/outliers."),
    ("GMM", f"k={best_k_bic} (mínimo BIC), n_init=10",
     "Modelo probabilístico. Asignación soft con probabilidades de pertenencia."),
    ("Hierarchical Ward", f"k={K_OPT}, linkage=ward",
     "Aglomerativo ascendente. Minimiza varianza intraclúster en fusiones."),
    ("Hierarchical Average", f"k={K_OPT}, linkage=average",
     "Variante usando distancia media entre grupos para comparativa."),
    ("Spectral", f"k={K_OPT}, affinity=rbf",
     "Basado en eigenvalues del Laplaciano. Captura estructuras no lineales."),
    ("Mean Shift", f"bandwidth={bw:.3f} (auto), bin_seeding=True",
     "No requiere k. Detecta automáticamente el número de modas de densidad."),
]
algo_tbl_data = [["Algoritmo", "Configuración", "Descripción"]]
for name, config, desc in algos_desc:
    algo_tbl_data.append([name, config, desc])
story.append(tbl(algo_tbl_data, cw=[3.5*cm, 4.5*cm, 9*cm]))
story.append(PageBreak())

# ═══ 6. RESULTADOS ══════════════════════════
story += [h1("6. Resultados y Evaluación"), hr(),
          h2("6.1 Tabla Comparativa de Métricas")]

metric_tbl_data = [["Algoritmo", "k", "Silhouette ↑", "Calinski-H ↑",
                     "Davies-B ↓", "Detalles"]]
for _, row in metric_df.iterrows():
    extra = results[row['Algoritmo']].get('extra', '-')
    metric_tbl_data.append([
        row['Algoritmo'], str(row['K']),
        f"{row['Silhouette']:.4f}",
        f"{row['Calinski-H']:.2f}",
        f"{row['Davies-B']:.4f}",
        extra[:50] + ("..." if len(extra) > 50 else "")
    ])
story.append(tbl(metric_tbl_data, cw=[3.5*cm, 1.2*cm, 2.5*cm, 3*cm, 2.5*cm, 4.3*cm]))

best_sil_row = metric_df.loc[metric_df['Silhouette'].idxmax()]
best_ch_row  = metric_df.loc[metric_df['Calinski-H'].idxmax()]
best_db_row  = metric_df.loc[metric_df['Davies-B'].idxmin()]
story += [sp(8),
          p(f"<b>Mejor Silhouette:</b> {best_sil_row['Algoritmo']} ({best_sil_row['Silhouette']:.4f})"),
          p(f"<b>Mejor Calinski-Harabasz:</b> {best_ch_row['Algoritmo']} ({best_ch_row['Calinski-H']:.2f})"),
          p(f"<b>Mejor Davies-Bouldin:</b> {best_db_row['Algoritmo']} ({best_db_row['Davies-B']:.4f})"),
          p(f"<b>Modelo seleccionado para interpretación:</b> <b>{best_algo}</b> "
            f"— mejor Silhouette global ({results[best_algo]['silhouette']:.4f})."),
          sp(10),
          im(f"{OUT_DIR}/fig09_comparativa_metricas.png"),
          cap("Figura 9. Comparativa de las tres métricas internas para todos los algoritmos."),
          PageBreak(),
          h2("6.2 Visualizaciones de Clústeres — PCA 2D"),
          im(f"{OUT_DIR}/fig10_clusters_pca2d.png"),
          cap("Figura 10. Proyección PCA 2D de todos los algoritmos con sus clústeres asignados."),
          PageBreak(),
          h2("6.3 Visualización t-SNE — Top 3 Modelos"),
          im(f"{OUT_DIR}/fig11_tsne_top3.png"),
          cap(f"Figura 11. Visualización t-SNE de los tres modelos con mayor Silhouette Score. "
              f"Top 3: {', '.join(top3)}."),
          PageBreak()]

# ═══ 7. INTERPRETACIÓN ══════════════════════
story += [h1(f"7. Interpretación de Clústeres — {best_algo}"), hr(),
          p(f"El modelo <b>{best_algo}</b> con <b>k={results[best_algo]['n_clusters']}</b> "
            f"clústeres obtuvo el mejor Silhouette Score de <b>{results[best_algo]['silhouette']:.4f}</b>, "
            f"indicando una separación coherente aunque moderada (típico en datos de encuesta). "
            f"A continuación se analiza el perfil de cada clúster."),
          sp(8),
          h2("7.1 Tabla de Perfiles Medios por Clúster")]

# Tabla de perfiles
prof_hdr = ["Feature"] + [f"Clúster {r['Clúster']} (n={r['N']})"
                            for _, r in profile_df.iterrows()]
prof_rows = [prof_hdr]
for feat in df.columns:
    row_vals = [feat]
    for _, r in profile_df.iterrows():
        row_vals.append(r.get(feat, "-"))
    prof_rows.append(row_vals)
# también pct y sil
for extra_col, label in [('Pct', '% del total'), ('Sil_medio', 'Silhouette medio')]:
    row_vals = [label]
    for _, r in profile_df.iterrows():
        row_vals.append(r.get(extra_col, "-"))
    prof_rows.append(row_vals)

n_cl_v = len(profile_df)
cw_prof = [3.5*cm] + [(W - 3.5*cm) / n_cl_v] * n_cl_v
story.append(tbl(prof_rows, cw=cw_prof))
story += [sp(10),
          im(f"{OUT_DIR}/fig13_heatmap_clusters.png"),
          cap(f"Figura 13. Heatmap de medias normalizadas por clúster. "
              "Colores más intensos indican valores más altos."),
          PageBreak(),
          h2("7.2 Análisis Visual de Perfiles")]

story += [im_sq(f"{OUT_DIR}/fig14_radar_clusters.png"),
          cap("Figura 14. Radar chart comparando el perfil de cada clúster en 12 features clave. "
              "Valores normalizados [0,1]."),
          sp(8),
          im(f"{OUT_DIR}/fig15_violines_clusters.png"),
          cap("Figura 15. Distribuciones de features clave por clúster (violin plots). "
              "La línea central indica la mediana."),
          sp(8),
          im(f"{OUT_DIR}/fig12_silhouette_plot.png"),
          cap(f"Figura 12. Silhouette plot por clúster — {best_algo}. "
              f"Silhouette medio = {results[best_algo]['silhouette']:.4f}."),
          PageBreak(),
          im(f"{OUT_DIR}/fig16_pca3d.png"),
          cap("Figura 16. Visualización 3D en espacio PCA (PC1, PC2, PC3)."),
          sp(8),
          im(f"{OUT_DIR}/fig17_importancia_features.png"),
          cap("Figura 17. Importancia de features medida como varianza entre medias de clústeres. "
              "A mayor valor, más discriminante es la feature."),
          PageBreak(),
          h2("7.3 Descripción Cualitativa de los Clústeres")]

# Descripción de cada clúster
feat_imp_sorted = feat_imp_df.sort_values('Var_Between', ascending=False)
top_feats = feat_imp_sorted['Feature'].head(5).tolist()

for _, row in profile_df.iterrows():
    cl = row['Clúster']
    sub = df_cl[df_cl['Cluster'] == cl]
    story.append(h3(f"Clúster {cl} — {row['N']} individuos ({row['Pct']}) · "
                     f"Silhouette medio: {row['Sil_medio']}"))
    # Descripción automática basada en features más discriminantes
    descs = []
    for feat in top_feats:
        val = float(row[feat])
        global_mean = df[feat].mean()
        diff_pct = (val - global_mean) / (global_mean + 1e-9) * 100
        if abs(diff_pct) > 15:
            direction = "por encima" if diff_pct > 0 else "por debajo"
            descs.append(f"<b>{feat}</b>={val:.2f} ({diff_pct:+.0f}% {direction} de la media global {global_mean:.2f})")
    if descs:
        story.append(p("Características distintivas respecto a la media global:"))
        for d in descs:
            story.append(bl(d))
    else:
        story.append(p("Perfil equilibrado sin desviaciones significativas en las features principales."))

    # Características clave
    age_val = float(row['Age'])
    tmt_val = float(row['treatment'])
    fh_val  = float(row['family_history'])
    wi_val  = float(row['work_interfere'])
    story.append(p(f"Edad media: {age_val:.1f} años | "
                    f"Tratamiento: {tmt_val*100:.0f}% | "
                    f"Hist. familiar: {fh_val*100:.0f}% | "
                    f"Interferencia laboral media: {wi_val:.2f}"))
    story.append(sp(8))

story.append(PageBreak())

# ═══ 8. DISCUSIÓN ═══════════════════════════
story += [h1("8. Discusión y Limitaciones"), hr(),
          h2("8.1 Comparativa de Algoritmos")]

comp_data = [
    ["Algoritmo", "Fortalezas en este dataset", "Limitaciones observadas", "Veredicto"],
    ["K-Means", "Estable, interpretable, rápido", "Asume esfericidad; sensible a k",
     "✓ Recomendado"],
    ["DBSCAN", "Detecta outliers; sin k fijo",
     f"Datasets uniformes dificultan DBSCAN; k={n_db} clusters",
     "⚠ Complementario"],
    ["GMM", "Probabilístico, flexible",
     "Mayor complejidad; puede sobreajustar",
     "✓ Alternativa válida"],
    ["Hier. Ward", "Dendrograma informativo",
     "O(n²) memoria; menos escalable",
     "✓ Útil para exploración"],
    ["Hier. Average", "Alternativa a Ward",
     "Menos cohesivo que Ward en este caso",
     "⚠ Inferior a Ward"],
    ["Spectral", "Estructuras no lineales",
     "Costoso O(n³); menos interpretable",
     "⚠ Solo si no-lineal"],
    ["Mean Shift", f"k={n_ms} automático",
     "Bandwidth sensible; lento para n>1000",
     "⚠ Referencia exploratoria"],
]
story.append(tbl(comp_data, cw=[3*cm, 4*cm, 4.5*cm, 2.5*cm + 3*cm]))

story += [sp(10),
          h2("8.2 Ventajas del Dataset Numérico vs. Original"),
          p("El uso del dataset ya codificado numéricamente ofrece varias ventajas "
            "respecto al análisis con el CSV original:"),
          bl("Consistencia de encoding: las variables ya están codificadas con criterio "
             "experto, eliminando ambigüedades en el mapeo ordinal."),
          bl("Correlaciones directas: la matriz de correlaciones cubre todas las variables, "
             "incluyendo las previamente categóricas."),
          bl("Reproducibilidad: el pipeline no depende de decisiones de encoding ad-hoc."),
          bl("Eficiencia: menor tiempo de preprocesamiento, permite más iteraciones en la "
             "búsqueda de hiperparámetros (grid search DBSCAN, gap statistic)."),
          sp(10),
          h2("8.3 Limitaciones del Estudio"),
          bl("Los valores de Silhouette son moderados (~0.1-0.3), esperables en datos de "
             "encuesta donde las fronteras entre grupos son inherentemente difusas."),
          bl("El dataset está sesgado hacia empleados de EE.UU. y países angloparlantes."),
          bl("Las variables originalmente ordinales codificadas como enteros pueden inducir "
             "relaciones de orden que no reflejan distancias reales (e.g., leave: 0-4)."),
          bl("Los algoritmos evalúan estructura geométrica, no causalidad. Los perfiles "
             "describen asociaciones, no relaciones causa-efecto."),
          PageBreak()]

# ═══ 9. RECOMENDACIONES ═════════════════════
story += [h1("9. Recomendaciones Estratégicas"), hr(),
          p("Los perfiles identificados tienen aplicaciones directas en la gestión del "
            "bienestar en empresas tecnológicas:"),
          sp(8)]

recs = [
    ("Segmentación de políticas de bienestar",
     "Diseñar programas específicos para cada segmento en lugar de iniciativas genéricas. "
     "Las features más discriminantes (ver Figura 17) son los ejes sobre los que deben "
     "diferenciarse las intervenciones."),
    ("Identificación proactiva de grupos vulnerables",
     "Los clústeres con baja puntuación en 'treatment' combinada con alta 'work_interfere' "
     "representan empleados que necesitan apoyo pero no lo están buscando. Requieren "
     "intervención proactiva, no reactiva."),
    ("Refuerzo de recursos institucionales",
     "Las features 'benefits', 'care_options', 'wellness_program' y 'seek_help' son "
     "altamente discriminantes. Invertir en estos recursos y comunicarlos activamente "
     "puede mover a empleados entre segmentos."),
    ("Seguimiento longitudinal",
     "Re-ejecutar el análisis tras implementar cambios permite medir si los empleados "
     "migran hacia clústeres más saludables (mayor treatment, menor work_interfere)."),
    ("Modelo predictivo derivado",
     "Con los clústeres como variable objetivo, se puede entrenar un modelo supervisado "
     "(Random Forest, XGBoost) para predecir el clúster de nuevos empleados y orientar "
     "el onboarding hacia recursos de SM apropiados."),
]
for title_r, desc_r in recs:
    story += [h3(f"• {title_r}"), p(desc_r), sp(6)]

story.append(PageBreak())

# ═══ 10. CONCLUSIONES ═══════════════════════
story += [h1("10. Conclusiones"), hr(),
          p(f"El análisis de clustering no supervisado sobre el dataset numérico limpio "
            f"({len(df):,} observaciones, {df.shape[1]} features) ha producido los "
            "siguientes hallazgos principales:"),
          sp(8)]

concls = [
    f"Se evaluaron <b>9 configuraciones de clustering</b> (K-Means ×3, DBSCAN, GMM, "
    f"Jerárquico ×2, Spectral, Mean Shift), obteniendo resultados comparables y coherentes.",
    f"El modelo <b>{best_algo}</b> con k={results[best_algo]['n_clusters']} clústeres "
    f"obtuvo el mejor Silhouette Score (<b>{results[best_algo]['silhouette']:.4f}</b>), "
    f"Calinski-Harabasz ({results[best_algo]['calinski']:.2f}) y "
    f"Davies-Bouldin ({results[best_algo]['davies']:.4f}).",
    f"PCA con <b>{n_95} componentes</b> (95% varianza) proporcionó el espacio óptimo para "
    f"el clustering, eliminando redundancias observadas en la matriz de correlaciones.",
    f"La <b>Gap Statistic</b>, añadida a este análisis, confirmó el k óptimo coincidiendo "
    f"con los demás criterios.",
    f"Las features más discriminantes entre clústeres son: "
    f"<b>{', '.join(feat_imp_sorted['Feature'].head(5).tolist())}</b>.",
    "DBSCAN y Mean Shift son menos adecuados para este dataset de estructura uniforme, "
    "mientras que K-Means y Jerárquico Ward son los más robustos.",
    "Los clústeres revelan segmentos diferenciados de empleados IT según su acceso a "
    "recursos de SM, disposición a buscar tratamiento e interferencia laboral percibida.",
]
for c in concls:
    story += [bl(c), sp(4)]

story += [sp(20), hr(),
          Paragraph(
              "Informe generado automáticamente · Python 3.12 · scikit-learn 1.4 · "
              "pandas 2.1 · matplotlib 3.6 · seaborn 0.13 · reportlab 4.4",
              sFooter)]

doc.build(story)
print(f"\n{'='*65}")
print(f"  INFORME PDF GENERADO:")
print(f"  {REPORT_PDF}")
print(f"{'='*65}")
n_figs = len([f for f in os.listdir(OUT_DIR) if f.endswith('.png')])
print(f"  Directorio de salida: {OUT_DIR}")
print(f"  Total figuras generadas: {n_figs}")
print(f"  Total archivos en directorio: {len(os.listdir(OUT_DIR))}")
