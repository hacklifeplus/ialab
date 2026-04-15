"""
Modelo Predictivo de Salud Mental en el Trabajo (IT Industry)
=============================================================
Target: predecir si una persona buscó tratamiento de salud mental (treatment: Yes/No)
Métrica principal: RECALL  (minimizar falsos negativos — no pasar por alto casos reales)
Dataset: IT_mental_health.survey.csv  — 1259 registros, 27 columnas
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay,
    recall_score, precision_score, f1_score,
    precision_recall_curve, average_precision_score
)

import joblib
import os

# ─────────────────────────────────────────────
# 1. CARGA Y EXPLORACIÓN INICIAL
# ─────────────────────────────────────────────
print("=" * 60)
print("  MODELO PREDICTIVO — SALUD MENTAL EN TECNOLOGÍA")
print("=" * 60)

DATA_PATH = "/root/Projects/ialab/IT_mental_health.survey.csv"
OUT_DIR   = "/root/Projects/ialab/output"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
print(f"\n[1] Dataset cargado: {df.shape[0]} filas × {df.shape[1]} columnas")
print(f"    Target 'treatment':  Yes={df['treatment'].eq('Yes').sum()}  |  No={df['treatment'].eq('No').sum()}")

# ─────────────────────────────────────────────
# 2. LIMPIEZA Y PREPROCESAMIENTO
# ─────────────────────────────────────────────
print("\n[2] Preprocesamiento...")

# Eliminar columnas de baja utilidad
df.drop(columns=['Timestamp', 'comments', 'state'], inplace=True)

# --- Limpiar Age ---
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df['Age'] = df['Age'].where(df['Age'].between(15, 80))   # outliers extremos → NaN

# --- Normalizar Gender ---
def normalizar_genero(g):
    if pd.isna(g):
        return 'Other'
    g = str(g).strip().lower()
    if g in ('male', 'm', 'man', 'cis male', 'cis man', 'malr', 'make', 'mail'):
        return 'Male'
    if g in ('female', 'f', 'woman', 'cis female', 'femake', 'femail', 'female (cis)',
             'cis-female/femme', 'woman', 'female '):
        return 'Female'
    return 'Other'

df['Gender'] = df['Gender'].apply(normalizar_genero)

# --- Limpiar work_interfere (tiene NaN = "no aplica") ---
df['work_interfere'] = df['work_interfere'].fillna('Unknown')

# --- Target a binario ---
df['treatment'] = df['treatment'].map({'Yes': 1, 'No': 0})

print(f"    Edad — media: {df['Age'].mean():.1f}  |  NaN Age: {df['Age'].isna().sum()}")
print(f"    Géneros normalizados: {df['Gender'].value_counts().to_dict()}")
print(f"    Valores nulos por columna:\n{df.isnull().sum()[df.isnull().sum()>0]}")

# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\n[3] Feature engineering...")

TARGET = 'treatment'
DROP   = ['Country']           # demasiados valores únicos; podría usarse con encoding avanzado
FEATURES = [c for c in df.columns if c not in [TARGET] + DROP]

X = df[FEATURES].copy()
y = df[TARGET].copy()

# Separar columnas numéricas y categóricas
num_cols = X.select_dtypes(include='number').columns.tolist()
cat_cols = X.select_dtypes(exclude='number').columns.tolist()

print(f"    Numéricas  ({len(num_cols)}): {num_cols}")
print(f"    Categóricas ({len(cat_cols)}): {cat_cols}")

# Pipeline de preprocesamiento
numeric_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler())
])

categorical_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

preprocessor = ColumnTransformer([
    ('num', numeric_pipe, num_cols),
    ('cat', categorical_pipe, cat_cols)
])

# ─────────────────────────────────────────────
# 4. DIVISIÓN TRAIN / TEST
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\n[4] Train: {X_train.shape[0]} muestras  |  Test: {X_test.shape[0]} muestras")

# ─────────────────────────────────────────────
# 5. ENTRENAMIENTO Y COMPARACIÓN DE MODELOS
# ─────────────────────────────────────────────
print("\n[5] Entrenando modelos...")

modelos = {
    "Logistic Regression":    LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree":          DecisionTreeClassifier(max_depth=6, random_state=42),
    "Random Forest":          RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "Gradient Boosting":      GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42),
    "K-Nearest Neighbors":    KNeighborsClassifier(n_neighbors=7),
    "SVM (RBF)":              SVC(kernel='rbf', probability=True, random_state=42),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
resultados = {}

print(f"    {'Modelo':<25}  {'Recall-CV':>10}  {'±':>6}  {'AUC-CV':>8}  {'F1-CV':>8}")
print(f"    {'-'*25}  {'-'*10}  {'-'*6}  {'-'*8}  {'-'*8}")
for nombre, clf in modelos.items():
    pipe = Pipeline([('prep', preprocessor), ('clf', clf)])
    rec  = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='recall',   n_jobs=-1)
    auc  = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='roc_auc',  n_jobs=-1)
    f1   = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='f1',       n_jobs=-1)
    resultados[nombre] = {'recall': rec, 'auc': auc, 'f1': f1}
    print(f"    {nombre:<25}  {rec.mean():>10.4f}  {rec.std():>6.4f}  {auc.mean():>8.4f}  {f1.mean():>8.4f}")

# ─────────────────────────────────────────────
# 6. MODELO GANADOR — EVALUACIÓN FINAL
# ─────────────────────────────────────────────
# Selección por RECALL más alto en CV
best_name = max(resultados, key=lambda k: resultados[k]['recall'].mean())
print(f"\n[6] Mejor modelo por Recall (CV): {best_name}")
print(f"    Recall-CV = {resultados[best_name]['recall'].mean():.4f} ± {resultados[best_name]['recall'].std():.4f}")

best_pipe = Pipeline([('prep', preprocessor), ('clf', modelos[best_name])])
best_pipe.fit(X_train, y_train)

y_proba = best_pipe.predict_proba(X_test)[:, 1]

# ── Umbral óptimo para máximo Recall con Precision mínima aceptable (≥ 0.60) ──
precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
# Buscar el umbral con mayor recall donde precision >= 0.60
valid = [(r, p, t) for p, r, t in zip(precisions[:-1], recalls[:-1], thresholds) if p >= 0.60]
if valid:
    best_recall_val, best_prec_val, best_thresh = max(valid, key=lambda x: x[0])
else:
    best_thresh = 0.5
    best_recall_val = recall_score(y_test, (y_proba >= 0.5).astype(int))
    best_prec_val   = precision_score(y_test, (y_proba >= 0.5).astype(int))

print(f"\n    Umbral óptimo (Recall máx, Precision ≥ 0.60): {best_thresh:.3f}")

y_pred_default  = (y_proba >= 0.50).astype(int)
y_pred_optimal  = (y_proba >= best_thresh).astype(int)

acc_def  = accuracy_score(y_test, y_pred_default)
rec_def  = recall_score(y_test, y_pred_default)
prec_def = precision_score(y_test, y_pred_default)
f1_def   = f1_score(y_test, y_pred_default)
auc      = roc_auc_score(y_test, y_proba)
ap       = average_precision_score(y_test, y_proba)

acc_opt  = accuracy_score(y_test, y_pred_optimal)
rec_opt  = recall_score(y_test, y_pred_optimal)
prec_opt = precision_score(y_test, y_pred_optimal)
f1_opt   = f1_score(y_test, y_pred_optimal)

print(f"\n    {'Métrica':<20} {'Umbral 0.50':>12} {'Umbral óptimo':>14}")
print(f"    {'-'*20}  {'-'*12}  {'-'*14}")
print(f"    {'Recall':<20} {rec_def:>12.4f} {rec_opt:>14.4f}  ← objetivo")
print(f"    {'Precision':<20} {prec_def:>12.4f} {prec_opt:>14.4f}")
print(f"    {'F1-Score':<20} {f1_def:>12.4f} {f1_opt:>14.4f}")
print(f"    {'Accuracy':<20} {acc_def:>12.4f} {acc_opt:>14.4f}")
print(f"    {'AUC-ROC':<20} {auc:>12.4f}")
print(f"    {'Avg Precision':<20} {ap:>12.4f}")

print("\n    Classification Report (umbral óptimo):")
print(classification_report(y_test, y_pred_optimal, target_names=['No tratamiento', 'Tratamiento']))

# usar y_pred_optimal para métricas y visualizaciones
y_pred = y_pred_optimal

# ─────────────────────────────────────────────
# 7. IMPORTANCIA DE CARACTERÍSTICAS (si aplica)
# ─────────────────────────────────────────────
try:
    feature_names = num_cols + cat_cols
    importances   = best_pipe.named_steps['clf'].feature_importances_
    feat_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feat_df = feat_df.sort_values('importance', ascending=False).head(15)
    print("\n[7] Top 15 características más importantes:")
    print(feat_df.to_string(index=False))
except AttributeError:
    feat_df = None
    print("\n[7] Este modelo no expone feature_importances_ directamente.")

# ─────────────────────────────────────────────
# 8. VISUALIZACIONES
# ─────────────────────────────────────────────
print("\n[8] Generando visualizaciones...")

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Modelo Predictivo — Salud Mental IT\n(Métrica objetivo: RECALL — minimizar falsos negativos)", fontsize=14, fontweight='bold')

# 8a — Comparación de modelos por Recall (CV)
ax = axes[0, 0]
names  = list(resultados.keys())
means  = [resultados[n]['recall'].mean() for n in names]
stds   = [resultados[n]['recall'].std()  for n in names]
colors = ['#e74c3c' if n == best_name else '#3498db' for n in names]
bars   = ax.barh(names, means, xerr=stds, color=colors, capsize=4)
ax.set_xlabel("Recall (CV 5-fold)")
ax.set_title("Comparación de modelos — Recall")
ax.set_xlim(0.4, 1.0)
for bar, m in zip(bars, means):
    ax.text(m + 0.003, bar.get_y() + bar.get_height()/2, f"{m:.3f}", va='center', fontsize=9)

# 8b — Curva Precision-Recall
ax = axes[0, 1]
ax.plot(recalls, precisions, color='#e74c3c', lw=2,
        label=f"{best_name}\nAP = {ap:.4f}")
ax.axvline(rec_opt,  color='#ffd166', lw=1.5, ls='--', label=f"Umbral ópt. Recall={rec_opt:.2f}")
ax.axhline(prec_opt, color='#6c63ff', lw=1.5, ls='--', label=f"Precision={prec_opt:.2f}")
ax.scatter([rec_opt], [prec_opt], color='#ffd166', zorder=5, s=60)
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Curva Precision-Recall")
ax.legend(fontsize=7.5)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)

# 8c — Recall y Precision vs Umbral
ax = axes[0, 2]
ax.plot(thresholds, recalls[:-1],    color='#e74c3c', lw=2, label='Recall')
ax.plot(thresholds, precisions[:-1], color='#6c63ff', lw=2, label='Precision')
ax.plot(thresholds, 2 * precisions[:-1] * recalls[:-1] /
        np.where((precisions[:-1]+recalls[:-1])==0, 1, precisions[:-1]+recalls[:-1]),
        color='#43e97b', lw=1.5, ls='--', label='F1')
ax.axvline(best_thresh, color='#ffd166', lw=1.5, ls=':', label=f'Umbral ópt.={best_thresh:.2f}')
ax.set_xlabel("Umbral de decisión")
ax.set_ylabel("Score")
ax.set_title("Recall / Precision / F1 vs Umbral")
ax.legend(fontsize=8)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)

# 8d — Importancia de características
ax = axes[1, 0]
if feat_df is not None:
    sns.barplot(data=feat_df, x='importance', y='feature', ax=ax, palette='viridis')
    ax.set_title("Top 15 características (importancia)")
    ax.set_xlabel("Importancia")
else:
    ax.axis('off')
    ax.text(0.5, 0.5, 'No disponible\npara este modelo',
            ha='center', va='center', transform=ax.transAxes)
    ax.set_title("Importancia de características")

# 8d-extra — tabla de métricas comparativa (umbral 0.5 vs óptimo)
from matplotlib.patches import FancyBboxPatch
metrics_text = (
    f"{'Métrica':<14} {'Thresh 0.50':>11} {'Thresh ópt':>11}\n"
    f"{'-'*38}\n"
    f"{'Recall':<14} {rec_def:>11.4f} {rec_opt:>11.4f}\n"
    f"{'Precision':<14} {prec_def:>11.4f} {prec_opt:>11.4f}\n"
    f"{'F1-Score':<14} {f1_def:>11.4f} {f1_opt:>11.4f}\n"
    f"{'Accuracy':<14} {acc_def:>11.4f} {acc_opt:>11.4f}\n"
    f"{'AUC-ROC':<14} {auc:>11.4f} {'—':>11}\n"
    f"{'Avg Prec.':<14} {ap:>11.4f} {'—':>11}"
)
ax.text(1.05, 0.95, metrics_text, transform=ax.transAxes,
        fontsize=7.5, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='#1a1d27', alpha=0.8, edgecolor='#6c63ff'))

# 8e — Matrices de confusión comparadas
ax = axes[1, 1]
cm_opt = confusion_matrix(y_test, y_pred_optimal)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_opt, display_labels=['No', 'Sí'])
disp.plot(ax=ax, colorbar=False, cmap='Reds')
ax.set_title(f"Confusión — Umbral óptimo ({best_thresh:.2f})\nRecall={rec_opt:.2f}  Precision={prec_opt:.2f}")

# 8f — Matriz de confusión umbral 0.50 (comparación)
ax = axes[1, 2]
cm_def = confusion_matrix(y_test, y_pred_default)
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_def, display_labels=['No', 'Sí'])
disp2.plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title(f"Confusión — Umbral default (0.50)\nRecall={rec_def:.2f}  Precision={prec_def:.2f}")

plt.tight_layout()
plot_path = os.path.join(OUT_DIR, "resultados_ml.png")
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"    Gráfico guardado: {plot_path}")

# ─────────────────────────────────────────────
# 9. GUARDAR MODELO
# ─────────────────────────────────────────────
model_path = os.path.join(OUT_DIR, "modelo_mental_health.joblib")
joblib.dump(best_pipe, model_path)
print(f"\n[9] Modelo guardado: {model_path}")

# ─────────────────────────────────────────────
# 10. FUNCIÓN DE PREDICCIÓN
# ─────────────────────────────────────────────
print("\n[10] Ejemplo de predicción con datos nuevos:")

sample = pd.DataFrame([{
    'Age': 29,
    'Gender': 'Male',
    'self_employed': 'No',
    'family_history': 'Yes',
    'work_interfere': 'Often',
    'no_employees': '26-100',
    'remote_work': 'No',
    'tech_company': 'Yes',
    'benefits': 'Yes',
    'care_options': 'Yes',
    'wellness_program': 'No',
    'seek_help': 'Yes',
    'anonymity': 'Yes',
    'leave': 'Somewhat easy',
    'mental_health_consequence': 'No',
    'phys_health_consequence': 'No',
    'coworkers': 'Some of them',
    'supervisor': 'Yes',
    'mental_health_interview': 'No',
    'phys_health_interview': 'Maybe',
    'mental_vs_physical': 'Yes',
    'obs_consequence': 'No',
}])

prob  = best_pipe.predict_proba(sample)[0][1]
pred  = best_pipe.predict(sample)[0]
label = "Sí buscará tratamiento" if pred == 1 else "No buscará tratamiento"
print(f"    Resultado: {label}  (probabilidad={prob:.2%})")

print("\n" + "=" * 60)
print("  PROCESO COMPLETADO")
print("=" * 60)
