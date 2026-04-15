"""
Modelo Predictivo de Salud Mental en el Trabajo (IT Industry)
=============================================================
Target: predecir si una persona buscó tratamiento de salud mental (treatment: Yes/No)
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
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
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

for nombre, clf in modelos.items():
    pipe = Pipeline([('prep', preprocessor), ('clf', clf)])
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
    resultados[nombre] = scores
    print(f"    {nombre:<25}  AUC-CV = {scores.mean():.4f} ± {scores.std():.4f}")

# ─────────────────────────────────────────────
# 6. MODELO GANADOR — EVALUACIÓN FINAL
# ─────────────────────────────────────────────
best_name = max(resultados, key=lambda k: resultados[k].mean())
print(f"\n[6] Mejor modelo (CV): {best_name}")

best_pipe = Pipeline([('prep', preprocessor), ('clf', modelos[best_name])])
best_pipe.fit(X_train, y_train)

y_pred  = best_pipe.predict(X_test)
y_proba = best_pipe.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print(f"\n    Test Accuracy : {acc:.4f}")
print(f"    Test AUC-ROC  : {auc:.4f}")
print("\n    Classification Report:")
print(classification_report(y_test, y_pred, target_names=['No tratamiento', 'Tratamiento']))

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
fig.suptitle("Modelo Predictivo — Salud Mental IT\n(Target: búsqueda de tratamiento)", fontsize=14, fontweight='bold')

# 8a — Comparación de modelos (CV AUC)
ax = axes[0, 0]
names  = list(resultados.keys())
means  = [resultados[n].mean() for n in names]
stds   = [resultados[n].std()  for n in names]
colors = ['#e74c3c' if n == best_name else '#3498db' for n in names]
bars   = ax.barh(names, means, xerr=stds, color=colors, capsize=4)
ax.set_xlabel("AUC-ROC (CV 5-fold)")
ax.set_title("Comparación de modelos")
ax.set_xlim(0.5, 1.0)
for bar, m in zip(bars, means):
    ax.text(m + 0.003, bar.get_y() + bar.get_height()/2, f"{m:.3f}", va='center', fontsize=9)

# 8b — Curva ROC
ax = axes[0, 1]
fpr, tpr, _ = roc_curve(y_test, y_proba)
ax.plot(fpr, tpr, color='#e74c3c', lw=2, label=f"{best_name}\nAUC = {auc:.4f}")
ax.plot([0,1],[0,1], 'k--', lw=1)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Curva ROC — Conjunto de Test")
ax.legend(loc='lower right')

# 8c — Matriz de Confusión
ax = axes[0, 2]
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No', 'Sí'])
disp.plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title("Matriz de Confusión")

# 8d — Importancia de características
ax = axes[1, 0]
if feat_df is not None:
    sns.barplot(data=feat_df, x='importance', y='feature', ax=ax, palette='viridis')
    ax.set_title("Top 15 características")
    ax.set_xlabel("Importancia")
else:
    ax.axis('off')
    ax.text(0.5, 0.5, 'No disponible\npara este modelo',
            ha='center', va='center', transform=ax.transAxes)
    ax.set_title("Importancia de características")

# 8e — Distribución del target
ax = axes[1, 1]
target_counts = df['treatment'].value_counts()
ax.pie([target_counts[1], target_counts[0]],
       labels=['Buscó\ntratamiento', 'No buscó\ntratamiento'],
       autopct='%1.1f%%', colors=['#e74c3c','#3498db'],
       startangle=90, textprops={'fontsize':10})
ax.set_title("Distribución del Target")

# 8f — Distribución de Age por target
ax = axes[1, 2]
for label, color, name in [(1,'#e74c3c','Tratamiento'),(0,'#3498db','Sin tratamiento')]:
    subset = df[df['treatment']==label]['Age'].dropna()
    ax.hist(subset, bins=20, alpha=0.6, color=color, label=name, density=True)
ax.set_xlabel("Edad")
ax.set_ylabel("Densidad")
ax.set_title("Distribución de Edad por Target")
ax.legend()

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
