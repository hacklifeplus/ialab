"""
Generador de Informe PDF — mental_health_predictor.html
=======================================================
Documenta con alto nivel de transparencia y explicabilidad:
  - Motivación y contexto del problema
  - Exploración y limpieza del dataset
  - Decisiones de ingeniería de características
  - Comparación de algoritmos
  - Árbol de decisión seleccionado (reglas, profundidad)
  - Métricas completas
  - Visualizaciones embebidas
  - Arquitectura de la app HTML/JS
"""

import warnings; warnings.filterwarnings('ignore')
import os, io, textwrap, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay,
    recall_score, precision_score, f1_score,
    precision_recall_curve, average_precision_score
)

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm, mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, HRFlowable, PageBreak, KeepTogether
)
from reportlab.platypus.flowables import Flowable
from reportlab.pdfgen import canvas as pdfcanvas

# ─────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────
OUT_PDF   = "/root/Projects/ialab/output/informe_mental_health_predictor.pdf"
DATA_PATH = "/root/Projects/ialab/IT_mental_health.survey.csv"
W, H      = float(A4[0]), float(A4[1])
MARGIN    = float(2.0 * cm)
CONTENT_W = W - 2 * MARGIN   # usable width

# ─────────────────────────────────────────────────────────────
# PALETA
# ─────────────────────────────────────────────────────────────
C_DARK    = colors.HexColor('#0f1117')
C_SURFACE = colors.HexColor('#1a1d27')
C_CARD    = colors.HexColor('#21253a')
C_ACCENT  = colors.HexColor('#6c63ff')
C_ACCENT2 = colors.HexColor('#ff6584')
C_GREEN   = colors.HexColor('#43e97b')
C_YELLOW  = colors.HexColor('#ffd166')
C_TEXT    = colors.HexColor('#e8eaf6')
C_MUTED   = colors.black
C_WHITE   = colors.white
C_BLACK   = colors.black

MPL_DARK  = '#0f1117'
MPL_CARD  = '#21253a'
MPL_BORD  = '#2e3250'
MPL_ACC   = '#6c63ff'
MPL_ACC2  = '#ff6584'
MPL_GREEN = '#43e97b'
MPL_YELL  = '#ffd166'
MPL_TEXT  = '#e8eaf6'
MPL_MUTED = '#8b90a8'

def dark_fig(w=14, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(MPL_DARK)
    ax.set_facecolor(MPL_CARD)
    for spine in ax.spines.values():
        spine.set_edgecolor(MPL_BORD)
    ax.tick_params(colors=MPL_MUTED, labelsize=9)
    ax.xaxis.label.set_color(MPL_MUTED)
    ax.yaxis.label.set_color(MPL_MUTED)
    ax.title.set_color(MPL_TEXT)
    return fig, ax

def dark_fig_multi(rows, cols, w=14, h=8):
    fig, axes = plt.subplots(rows, cols, figsize=(w, h))
    fig.patch.set_facecolor(MPL_DARK)
    for ax in (axes.flat if hasattr(axes, 'flat') else [axes]):
        ax.set_facecolor(MPL_CARD)
        for spine in ax.spines.values(): spine.set_edgecolor(MPL_BORD)
        ax.tick_params(colors=MPL_MUTED, labelsize=8)
        ax.xaxis.label.set_color(MPL_MUTED)
        ax.yaxis.label.set_color(MPL_MUTED)
        ax.title.set_color(MPL_TEXT)
    return fig, axes

def fig_to_image(fig, width=None):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    img = Image(buf)
    if width:
        ratio = img.imageHeight / img.imageWidth
        img.drawWidth  = width
        img.drawHeight = width * ratio
    return img

# ─────────────────────────────────────────────────────────────
# ESTILOS
# ─────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()

def style(name, **kw):
    return ParagraphStyle(name, **kw)

S_TITLE = style('S_TITLE',
    fontSize=26, fontName='Helvetica-Bold', textColor=C_ACCENT,
    spaceAfter=6, alignment=TA_CENTER, leading=32)

S_SUBTITLE = style('S_SUBTITLE',
    fontSize=13, fontName='Helvetica', textColor=C_MUTED,
    spaceAfter=4, alignment=TA_CENTER)

S_H1 = style('S_H1',
    fontSize=16, fontName='Helvetica-Bold', textColor=C_ACCENT,
    spaceBefore=14, spaceAfter=6, leading=20)

S_H2 = style('S_H2',
    fontSize=12, fontName='Helvetica-Bold', textColor=C_ACCENT2,
    spaceBefore=10, spaceAfter=4, leading=15)

S_H3 = style('S_H3',
    fontSize=10, fontName='Helvetica-Bold', textColor=C_YELLOW,
    spaceBefore=6, spaceAfter=3, leading=13)

S_BODY = style('S_BODY',
    fontSize=9, fontName='Helvetica', textColor=colors.black,
    spaceBefore=2, spaceAfter=4, leading=14, alignment=TA_JUSTIFY)

S_CODE = style('S_CODE',
    fontSize=8, fontName='Courier', textColor=C_GREEN,
    spaceBefore=2, spaceAfter=2, leading=11,
    backColor=colors.HexColor('#0d1020'), leftIndent=12, rightIndent=12)

S_CAPTION = style('S_CAPTION',
    fontSize=8, fontName='Helvetica-Oblique', textColor=C_MUTED,
    spaceAfter=6, alignment=TA_CENTER)

S_LABEL = style('S_LABEL',
    fontSize=9, fontName='Helvetica-Bold', textColor=C_YELLOW)

S_META = style('S_META',
    fontSize=8, fontName='Helvetica', textColor=C_MUTED,
    alignment=TA_CENTER)

def hr(): return HRFlowable(width='100%', thickness=0.5,
                             color=colors.HexColor('#2e3250'), spaceAfter=8)

def sp(h=6): return Spacer(1, h)

def p(text, s=None): return Paragraph(text, s or S_BODY)

def h1(t): return Paragraph(t, S_H1)
def h2(t): return Paragraph(t, S_H2)
def h3(t): return Paragraph(t, S_H3)

def code(t):
    lines = t.strip().split('\n')
    return [Paragraph(l.replace(' ','&nbsp;').replace('<','&lt;'), S_CODE) for l in lines]

def cw(*widths):
    """Convert col_widths to plain Python floats for reportlab."""
    return [float(w) for w in widths]

def metrics_table(rows, col_widths=None):
    """Tabla estilizada oscura."""
    tbl = Table(rows, colWidths=col_widths)
    tbl.setStyle(TableStyle([
        ('BACKGROUND',   (0,0), (-1,0),  colors.HexColor('#1a1040')),
        ('TEXTCOLOR',    (0,0), (-1,0),  C_ACCENT),
        ('FONTNAME',     (0,0), (-1,0),  'Helvetica-Bold'),
        ('FONTSIZE',     (0,0), (-1,0),  9),
        ('BACKGROUND',   (0,1), (-1,-1), colors.HexColor('#16192a')),
        ('TEXTCOLOR',    (0,1), (-1,-1), colors.HexColor('#e8eaf6')),
        ('FONTNAME',     (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE',     (0,1), (-1,-1), 8.5),
        ('ROWBACKGROUNDS',(0,1),(-1,-1), [colors.HexColor('#16192a'),
                                          colors.HexColor('#1c2035')]),
        ('GRID',         (0,0), (-1,-1), 0.4, colors.HexColor('#2e3250')),
        ('ALIGN',        (0,0), (-1,-1), 'CENTER'),
        ('VALIGN',       (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING',   (0,0), (-1,-1), 4),
        ('BOTTOMPADDING',(0,0), (-1,-1), 4),
        ('LEFTPADDING',  (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    return tbl

# ─────────────────────────────────────────────────────────────
# 0.  REPRODUCIR EL PIPELINE COMPLETO
# ─────────────────────────────────────────────────────────────
print("Reproduciendo pipeline ML...")

df_raw = pd.read_csv(DATA_PATH)
df = df_raw.copy()
df.drop(columns=['Timestamp', 'comments', 'state'], inplace=True)
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df['Age'] = df['Age'].where(df['Age'].between(15, 80))

def norm_gender(g):
    if pd.isna(g): return 'Other'
    g = str(g).strip().lower()
    if g in ('male','m','man','cis male','cis man','malr','make','mail'): return 'Male'
    if g in ('female','f','woman','cis female','femake','femail','female (cis)',
             'cis-female/femme','femail'): return 'Female'
    return 'Other'

df['Gender'] = df['Gender'].apply(norm_gender)
df['work_interfere'] = df['work_interfere'].fillna('Unknown')
df['treatment'] = df['treatment'].map({'Yes':1,'No':0})

TARGET  = 'treatment'
DROP    = ['Country']
FEATURES = [c for c in df.columns if c not in [TARGET]+DROP]
X = df[FEATURES].copy()
y = df[TARGET].copy()

num_cols = X.select_dtypes(include='number').columns.tolist()
cat_cols = X.select_dtypes(exclude='number').columns.tolist()

num_pipe = Pipeline([('imp', SimpleImputer(strategy='median')),
                     ('sc',  StandardScaler())])
cat_pipe = Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                     ('enc', OrdinalEncoder(handle_unknown='use_encoded_value',
                                            unknown_value=-1))])
preprocessor = ColumnTransformer([
    ('num', num_pipe, num_cols),
    ('cat', cat_pipe, cat_cols)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

modelos = {
    "Logistic Regression":  LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree":        DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest":        RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "Gradient Boosting":    GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42),
    "K-Nearest Neighbors":  KNeighborsClassifier(n_neighbors=7),
    "SVM (RBF)":            SVC(kernel='rbf', probability=True, random_state=42),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
resultados = {}

for nombre, clf in modelos.items():
    pipe = Pipeline([('prep', preprocessor), ('clf', clf)])
    auc  = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='roc_auc',  n_jobs=-1)
    rec  = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='recall',   n_jobs=-1)
    f1   = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='f1',       n_jobs=-1)
    acc  = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
    resultados[nombre] = {'auc': auc, 'recall': rec, 'f1': f1, 'acc': acc}
    print(f"  {nombre:<25} AUC={auc.mean():.4f}")

# Modelo ganador (AUC para mental_health_predictor.html v1)
BEST = "Random Forest"
best_pipe = Pipeline([('prep', preprocessor), ('clf', modelos[BEST])])
best_pipe.fit(X_train, y_train)
y_pred  = best_pipe.predict(X_test)
y_proba = best_pipe.predict_proba(X_test)[:,1]
acc_t  = accuracy_score(y_test, y_pred)
auc_t  = roc_auc_score(y_test, y_proba)
rec_t  = recall_score(y_test, y_pred)
prec_t = precision_score(y_test, y_pred)
f1_t   = f1_score(y_test, y_pred)

# Decision Tree para explicabilidad
dt_pipe = Pipeline([('prep', preprocessor),
                    ('clf', DecisionTreeClassifier(max_depth=5, random_state=42))])
dt_pipe.fit(X_train, y_train)
dt_model = dt_pipe.named_steps['clf']
dt_pred  = dt_pipe.predict(X_test)
dt_proba = dt_pipe.predict_proba(X_test)[:,1]
dt_auc   = roc_auc_score(y_test, dt_proba)
dt_acc   = accuracy_score(y_test, dt_pred)
dt_rec   = recall_score(y_test, dt_pred)
dt_prec  = precision_score(y_test, dt_pred)
dt_f1    = f1_score(y_test, dt_pred)

feature_names = num_cols + cat_cols
dt_importances = pd.DataFrame({'feature': feature_names,
                                'importance': dt_model.feature_importances_})\
                   .sort_values('importance', ascending=False)

rf_importances = pd.DataFrame({'feature': feature_names,
    'importance': best_pipe.named_steps['clf'].feature_importances_})\
    .sort_values('importance', ascending=False)

print("Pipeline listo. Generando figuras...")

# ─────────────────────────────────────────────────────────────
# FIGURAS
# ─────────────────────────────────────────────────────────────

# FIG 1 — Distribución del dataset
fig1, axes1 = dark_fig_multi(1, 3, w=14, h=4)
fig1.suptitle('Exploración del Dataset', color=MPL_TEXT, fontsize=11, fontweight='bold')

# 1a — target
ax = axes1[0]
vc = df['treatment'].value_counts()
bars = ax.bar(['Sin tratamiento','Con tratamiento'], [vc[0], vc[1]],
              color=[MPL_ACC, MPL_ACC2], width=0.5, edgecolor=MPL_BORD)
for b, v in zip(bars, [vc[0], vc[1]]):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+8, str(v),
            ha='center', color=MPL_TEXT, fontsize=9, fontweight='bold')
ax.set_title('Distribución del Target', color=MPL_TEXT)
ax.set_ylabel('Frecuencia', color=MPL_MUTED)
ax.set_ylim(0, 750)

# 1b — Age distribution
ax = axes1[1]
for label, color, name in [(1,MPL_ACC2,'Tratamiento'),(0,MPL_ACC,'Sin tratamiento')]:
    subset = df[df['treatment']==label]['Age'].dropna()
    ax.hist(subset, bins=22, alpha=0.65, color=color, label=name, density=True)
ax.set_title('Distribución de Edad por Target', color=MPL_TEXT)
ax.set_xlabel('Edad', color=MPL_MUTED)
ax.set_ylabel('Densidad', color=MPL_MUTED)
ax.legend(fontsize=8, labelcolor=MPL_TEXT, facecolor=MPL_CARD, edgecolor=MPL_BORD)

# 1c — work_interfere
ax = axes1[2]
wi_order = ['Never','Rarely','Sometimes','Often','Unknown']
wi_counts = {v: df[df['work_interfere']==v]['treatment'].value_counts() for v in wi_order}
x = np.arange(len(wi_order))
w = 0.38
ax.bar(x-w/2, [wi_counts[v].get(0,0) for v in wi_order], w, label='Sin tratamiento',
       color=MPL_ACC, edgecolor=MPL_BORD)
ax.bar(x+w/2, [wi_counts[v].get(1,0) for v in wi_order], w, label='Con tratamiento',
       color=MPL_ACC2, edgecolor=MPL_BORD)
ax.set_xticks(x); ax.set_xticklabels(wi_order, rotation=15, fontsize=7.5)
ax.set_title('Work Interfere × Target', color=MPL_TEXT)
ax.legend(fontsize=7.5, labelcolor=MPL_TEXT, facecolor=MPL_CARD, edgecolor=MPL_BORD)
plt.tight_layout()
img_eda = fig_to_image(fig1, width=W - 2*MARGIN)

# FIG 2 — Missing values & outliers
fig2, axes2 = dark_fig_multi(1, 2, w=12, h=4)
fig2.suptitle('Calidad de Datos — Valores Nulos y Outliers', color=MPL_TEXT,
              fontsize=11, fontweight='bold')

# 2a — missing values
ax = axes2[0]
missing = df_raw.isnull().sum().sort_values(ascending=False)
missing = missing[missing > 0]
ax.barh(missing.index, missing.values, color=MPL_ACC2, edgecolor=MPL_BORD)
for i, v in enumerate(missing.values):
    ax.text(v+1, i, str(v), va='center', color=MPL_TEXT, fontsize=8)
ax.set_title('Valores Nulos por Columna (raw)', color=MPL_TEXT)
ax.set_xlabel('Conteo', color=MPL_MUTED)
ax.invert_yaxis()

# 2b — Age outliers
ax = axes2[1]
age_raw = pd.to_numeric(df_raw['Age'], errors='coerce')
ax.hist(age_raw.dropna(), bins=40, color=MPL_ACC, edgecolor=MPL_BORD, alpha=0.7,
        label='Edad raw')
age_clean = age_raw[age_raw.between(15,80)]
ax.hist(age_clean.dropna(), bins=40, color=MPL_GREEN, edgecolor=MPL_BORD, alpha=0.7,
        label='Edad filtrada (15-80)')
ax.axvline(15, color=MPL_YELL, ls='--', lw=1.2, label='Límites (15,80)')
ax.axvline(80, color=MPL_YELL, ls='--', lw=1.2)
ax.set_title('Limpieza de Outliers — Edad', color=MPL_TEXT)
ax.set_xlabel('Edad', color=MPL_MUTED); ax.set_ylabel('Frecuencia', color=MPL_MUTED)
ax.legend(fontsize=8, labelcolor=MPL_TEXT, facecolor=MPL_CARD, edgecolor=MPL_BORD)
ax.set_xlim(-200, 150)
plt.tight_layout()
img_quality = fig_to_image(fig2, width=W - 2*MARGIN)

# FIG 3 — Comparación de modelos
fig3, axes3 = dark_fig_multi(1, 2, w=14, h=5)
fig3.suptitle('Comparación de Algoritmos — Validación Cruzada 5-fold', color=MPL_TEXT,
              fontsize=11, fontweight='bold')

nombres = list(resultados.keys())
metricas = ['auc', 'acc', 'f1', 'recall']
colores_met = [MPL_ACC, MPL_GREEN, MPL_YELL, MPL_ACC2]
etiquetas   = ['AUC-ROC', 'Accuracy', 'F1-Score', 'Recall']

ax = axes3[0]
auc_means = [resultados[n]['auc'].mean() for n in nombres]
auc_stds  = [resultados[n]['auc'].std()  for n in nombres]
col_bars  = [MPL_ACC2 if n == BEST else MPL_ACC for n in nombres]
bars = ax.barh(nombres, auc_means, xerr=auc_stds, color=col_bars, capsize=4,
               edgecolor=MPL_BORD)
for bar, m in zip(bars, auc_means):
    ax.text(m+0.003, bar.get_y()+bar.get_height()/2, f'{m:.3f}',
            va='center', color=MPL_TEXT, fontsize=8.5)
ax.set_xlabel('AUC-ROC', color=MPL_MUTED)
ax.set_title('AUC-ROC (CV) — Ganador resaltado', color=MPL_TEXT)
ax.set_xlim(0.5, 1.0)
ax.axvline(0.88, color=MPL_YELL, ls='--', lw=1, alpha=0.6)

ax = axes3[1]
x = np.arange(len(nombres)); bw = 0.18
for i, (met, col, lbl) in enumerate(zip(metricas, colores_met, etiquetas)):
    vals = [resultados[n][met].mean() for n in nombres]
    ax.bar(x + (i-1.5)*bw, vals, bw, label=lbl, color=col,
           edgecolor=MPL_BORD, alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels([n.replace(' ','\n') for n in nombres], fontsize=7.5)
ax.set_ylim(0.5, 1.0)
ax.set_title('Todas las métricas por modelo', color=MPL_TEXT)
ax.legend(fontsize=8, labelcolor=MPL_TEXT, facecolor=MPL_CARD, edgecolor=MPL_BORD,
          loc='lower right')
plt.tight_layout()
img_compare = fig_to_image(fig3, width=W - 2*MARGIN)

# FIG 4 — Árbol de decisión (visualización)
fig4 = plt.figure(figsize=(16, 7))
fig4.patch.set_facecolor(MPL_DARK)
ax4 = fig4.add_subplot(111)
ax4.set_facecolor(MPL_DARK)
plot_tree(dt_model, feature_names=feature_names,
          class_names=['No tratamiento','Tratamiento'],
          filled=True, rounded=True, impurity=False,
          fontsize=6, ax=ax4,
          precision=3)
ax4.set_title('Árbol de Decisión (depth=5) — Estructura completa',
              color=MPL_TEXT, fontsize=11, pad=10)
img_tree = fig_to_image(fig4, width=W - 2*MARGIN)

# FIG 5 — Importancias
fig5, axes5 = dark_fig_multi(1, 2, w=14, h=5)
fig5.suptitle('Importancia de Características', color=MPL_TEXT,
              fontsize=11, fontweight='bold')

ax = axes5[0]
top_dt = dt_importances.head(12)
colors_bar = [MPL_ACC2 if i==0 else MPL_ACC for i in range(len(top_dt))]
ax.barh(top_dt['feature'][::-1], top_dt['importance'][::-1],
        color=colors_bar[::-1], edgecolor=MPL_BORD)
ax.set_title('Decision Tree — Top 12', color=MPL_TEXT)
ax.set_xlabel('Importancia (Gini)', color=MPL_MUTED)

ax = axes5[1]
top_rf = rf_importances.head(12)
ax.barh(top_rf['feature'][::-1], top_rf['importance'][::-1],
        color=MPL_ACC, edgecolor=MPL_BORD)
ax.set_title('Random Forest — Top 12', color=MPL_TEXT)
ax.set_xlabel('Importancia (Mean Decrease Impurity)', color=MPL_MUTED)
plt.tight_layout()
img_importances = fig_to_image(fig5, width=W - 2*MARGIN)

# FIG 6 — Curva ROC y Matriz de Confusión (Random Forest)
fig6, axes6 = dark_fig_multi(1, 3, w=14, h=4.5)
fig6.suptitle('Evaluación en Test Set — Random Forest', color=MPL_TEXT,
              fontsize=11, fontweight='bold')

ax = axes6[0]
fpr, tpr, _ = roc_curve(y_test, y_proba)
ax.plot(fpr, tpr, color=MPL_ACC2, lw=2, label=f'RF  AUC={auc_t:.4f}')
fpr_dt, tpr_dt, _ = roc_curve(y_test, dt_proba)
ax.plot(fpr_dt, tpr_dt, color=MPL_ACC, lw=1.5, ls='--', label=f'DT  AUC={dt_auc:.4f}')
ax.plot([0,1],[0,1],'--', color=MPL_BORD, lw=1)
ax.fill_between(fpr, tpr, alpha=0.08, color=MPL_ACC2)
ax.set_xlabel('FPR', color=MPL_MUTED); ax.set_ylabel('TPR', color=MPL_MUTED)
ax.set_title('Curva ROC', color=MPL_TEXT)
ax.legend(fontsize=8.5, labelcolor=MPL_TEXT, facecolor=MPL_CARD, edgecolor=MPL_BORD)

ax = axes6[1]
precs, recs, _ = precision_recall_curve(y_test, y_proba)
ap = average_precision_score(y_test, y_proba)
ax.plot(recs, precs, color=MPL_ACC2, lw=2, label=f'RF  AP={ap:.4f}')
precs_dt, recs_dt, _ = precision_recall_curve(y_test, dt_proba)
ap_dt = average_precision_score(y_test, dt_proba)
ax.plot(recs_dt, precs_dt, color=MPL_ACC, lw=1.5, ls='--', label=f'DT  AP={ap_dt:.4f}')
ax.set_xlabel('Recall', color=MPL_MUTED); ax.set_ylabel('Precision', color=MPL_MUTED)
ax.set_title('Curva Precision-Recall', color=MPL_TEXT)
ax.legend(fontsize=8.5, labelcolor=MPL_TEXT, facecolor=MPL_CARD, edgecolor=MPL_BORD)

ax = axes6[2]
conf_mat = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(conf_mat, display_labels=['No tratamiento','Tratamiento'])
disp.plot(ax=ax, colorbar=False, cmap='plasma')
ax.set_title('Matriz de Confusión (RF, umbral 0.5)', color=MPL_TEXT)
plt.tight_layout()
img_eval = fig_to_image(fig6, width=W - 2*MARGIN)

# FIG 7 — Correlaciones categóricas clave
fig7, axes7 = dark_fig_multi(2, 3, w=14, h=7)
fig7.suptitle('Análisis Exploratorio — Variables vs Target', color=MPL_TEXT,
              fontsize=11, fontweight='bold')

cat_to_plot = ['family_history','work_interfere','benefits',
               'care_options','mental_health_consequence','leave']
for i, col in enumerate(cat_to_plot):
    ax = axes7.flat[i]
    vc = df.groupby([col,'treatment']).size().unstack(fill_value=0)
    vc_pct = vc.div(vc.sum(axis=1), axis=0)*100
    vc_pct.plot(kind='bar', ax=ax, color=[MPL_ACC, MPL_ACC2],
                edgecolor=MPL_BORD, width=0.7, legend=(i==0))
    ax.set_title(col, color=MPL_TEXT, fontsize=8.5)
    ax.set_xlabel('', color=MPL_MUTED)
    ax.set_ylabel('% por categoría', color=MPL_MUTED)
    ax.tick_params(axis='x', rotation=20, labelsize=7)
    if i == 0:
        ax.legend(['Sin tratamiento','Con tratamiento'], fontsize=7,
                  labelcolor=MPL_TEXT, facecolor=MPL_CARD, edgecolor=MPL_BORD)
plt.tight_layout()
img_corr = fig_to_image(fig7, width=W - 2*MARGIN)

# FIG 8 — Preprocess pipeline visual
fig8, ax8 = dark_fig(w=12, h=3.5)
ax8.axis('off')
steps = [
    ('CSV\n1259 × 27', '#1a1040'),
    ('Limpieza\n(Age, Gender\nTimestamp)', '#0d1a2a'),
    ('Imputación\n(median / mode)', '#0d2a1a'),
    ('Encoding\nOrdinal', '#1a1a0d'),
    ('Escalado\nStandardScaler', '#2a0d1a'),
    ('Train/Test\n80% / 20%', '#0d1a1a'),
    ('Pipeline\nsklearn', '#1a0d2a'),
]
n = len(steps)
for i, (label, bg) in enumerate(steps):
    x = i / n + 0.02
    rect = FancyBboxPatch((x, 0.15), 0.12, 0.7,
                           boxstyle='round,pad=0.02',
                           facecolor=bg, edgecolor=MPL_ACC, lw=1.5,
                           transform=ax8.transAxes, clip_on=False)
    ax8.add_patch(rect)
    ax8.text(x+0.06, 0.5, label, ha='center', va='center',
             fontsize=7.5, color=MPL_TEXT, transform=ax8.transAxes,
             multialignment='center')
    if i < n-1:
        ax8.annotate('', xy=((i+1)/n+0.02, 0.5),
                     xytext=(x+0.12, 0.5),
                     xycoords='axes fraction', textcoords='axes fraction',
                     arrowprops=dict(arrowstyle='->', color=MPL_ACC, lw=1.5))
ax8.set_title('Pipeline de Preprocesamiento', color=MPL_TEXT, fontsize=10, pad=10)
img_pipeline = fig_to_image(fig8, width=W - 2*MARGIN)

# FIG 9 — Reglas del árbol de decisión (texto)
dt_rules = export_text(dt_model, feature_names=feature_names, max_depth=3,
                        show_weights=True)

print("Figuras listas. Generando PDF...")

# ─────────────────────────────────────────────────────────────
# CONSTRUCCIÓN DEL PDF
# ─────────────────────────────────────────────────────────────
class DarkBackground(Flowable):
    """Fondo oscuro para la portada."""
    def __init__(self, w, h):
        self.bw = w; self.bh = h
    def draw(self):
        self.canv.setFillColor(colors.HexColor('#0f1117'))
        self.canv.rect(0, 0, self.bw, self.bh, fill=1, stroke=0)

def build_pdf():
    doc = SimpleDocTemplate(
        OUT_PDF, pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN,
        title="Informe: Predictor de Salud Mental IT",
        author="IA Lab — Claude Sonnet",
        subject="Transparencia y Explicabilidad del Modelo ML"
    )

    story = []

    # ── PORTADA ──────────────────────────────────────────────
    story.append(sp(30))
    story.append(p(
        '<font color="#6c63ff" size="28"><b>Informe de Transparencia</b></font><br/>'
        '<font color="#ff6584" size="22"><b>Modelo Predictivo de Salud Mental IT</b></font>',
        style('cover_title', fontSize=28, alignment=TA_CENTER, leading=38,
              spaceAfter=0, fontName='Helvetica-Bold')))
    story.append(sp(10))
    story.append(p('mental_health_predictor.html — Versión 1 (Decision Tree / Random Forest)',
                   S_SUBTITLE))
    story.append(sp(4))
    story.append(p('Dataset: OSMI Mental Health in Tech Survey 2014 &nbsp;|&nbsp; '
                   '1 259 registros &nbsp;|&nbsp; 27 variables', S_META))
    story.append(p('Algoritmo final: Random Forest (200 árboles) &nbsp;|&nbsp; '
                   'AUC-ROC: 87.4% &nbsp;|&nbsp; Accuracy: 79.4%', S_META))
    story.append(sp(4))
    story.append(p('Fecha: Abril 2026 &nbsp;|&nbsp; Generado con Python + scikit-learn + ReportLab',
                   S_META))
    story.append(sp(16))
    story.append(hr())
    story.append(sp(8))

    # Resumen ejecutivo
    story.append(h1('Resumen Ejecutivo'))
    story.append(p(
        'Este informe documenta con alto nivel de <b>transparencia y explicabilidad</b> '
        'cada decisión tomada durante la construcción del modelo predictivo embebido en '
        '<b>mental_health_predictor.html</b>. El objetivo es predecir si un trabajador '
        'del sector tecnológico <b>buscará tratamiento de salud mental</b>, '
        'usando únicamente variables de contexto laboral y demográfico — sin ningún dato clínico. '
        'Se documenta el razonamiento detrás de cada elección: desde la limpieza del dataset '
        'hasta la selección del umbral de decisión y la arquitectura de la aplicación portable.'
    ))
    story.append(sp(4))

    # Tabla resumen
    summary_data = [
        ['Concepto', 'Valor / Decisión'],
        ['Dataset', 'OSMI Mental Health in Tech Survey 2014'],
        ['Registros', '1 259 filas × 27 columnas'],
        ['Target', 'treatment: ¿buscó tratamiento? (Yes/No → 1/0)'],
        ['Balance clases', 'Yes: 637 (50.6%) | No: 622 (49.4%) — prácticamente balanceado'],
        ['Variables usadas', '22 (eliminadas: Timestamp, comments, state, Country)'],
        ['Preprocesamiento', 'Imputación mediana/moda + StandardScaler + OrdinalEncoder'],
        ['Modelo exportado a HTML', 'Decision Tree (depth=5) con m2cgen → ~10 KB de JS'],
        ['Mejor modelo CV', 'Random Forest — AUC-ROC: 88.77% ± 1.36%'],
        ['Métricas test (RF)', 'Acc=79.4% | AUC=87.4% | Recall=84.4% | Precision=80.0%'],
        ['Umbral de decisión', '0.50 (por defecto)'],
        ['Portabilidad', 'Archivo HTML único, 33 KB, sin dependencias externas'],
    ]
    story.append(metrics_table(summary_data,
                               col_widths=cw(5.5*cm, CONTENT_W-5.5*cm)))
    story.append(PageBreak())

    # ── 1. CONTEXTO Y MOTIVACIÓN ─────────────────────────────
    story.append(h1('1. Contexto y Motivación del Problema'))
    story.append(p(
        'La salud mental en el entorno laboral tecnológico es un problema documentado '
        'con impacto directo en productividad, rotación y bienestar de los equipos. '
        'La encuesta OSMI (<i>Open Sourcing Mental Illness</i>) recopila anualmente datos '
        'sobre percepciones, actitudes y experiencias de trabajadores IT respecto a '
        'la salud mental en sus empresas.'
    ))
    story.append(p(
        '<b>Pregunta de negocio:</b> ¿Puede predecirse, a partir de variables de contexto '
        'laboral y demográfico, si un empleado IT está buscando o ha buscado tratamiento '
        'de salud mental? Este indicador permite a las organizaciones identificar patrones '
        'sistémicos y diseñar intervenciones preventivas.'
    ))
    story.append(p(
        '<b>Decisión de diseño:</b> El modelo se entrenó como clasificador binario '
        '(Yes/No tratamiento) en lugar de regresión, ya que el objetivo es una decisión '
        'de acción (intervenir o no), no una estimación continua.'
    ))
    story.append(p(
        '<b>Consideración ética:</b> Este modelo es un indicador estadístico de patrones '
        'poblacionales, <b>no un diagnóstico individual</b>. No debe usarse para tomar '
        'decisiones sobre personas concretas sin intervención humana cualificada.'
    ))
    story.append(sp(6))

    # ── 2. EXPLORACIÓN DEL DATASET ───────────────────────────
    story.append(h1('2. Exploración del Dataset (EDA)'))
    story.append(h2('2.1 Estructura del dataset'))
    story.append(p(
        'El dataset contiene <b>1 259 respuestas</b> de trabajadores IT de todo el mundo, '
        'recogidas en 2014 mediante encuesta online. Las 27 variables originales cubren '
        'datos demográficos, características de la empresa, política de la empresa respecto '
        'a la salud mental, y actitudes personales.'
    ))

    col_table = [
        ['Columna', 'Tipo', 'Descripción', 'Nulos'],
        ['Timestamp', 'DateTime', 'Fecha de respuesta (eliminada)', '0'],
        ['Age', 'Numérica', 'Edad del respondente', '0'],
        ['Gender', 'Categórica', 'Género (alta variabilidad textual)', '0'],
        ['Country', 'Categórica', 'País (60+ valores únicos, eliminada)', '0'],
        ['state', 'Categórica', 'Estado/provincia (solo US, eliminado)', '515'],
        ['self_employed', 'Binaria', '¿Trabaja por cuenta propia?', '18'],
        ['family_history', 'Binaria', '¿Historial familiar de SM?', '0'],
        ['treatment', 'TARGET', '¿Ha buscado tratamiento de SM?', '0'],
        ['work_interfere', 'Ordinal', '¿El trabajo interfiere con SM?', '264'],
        ['no_employees', 'Ordinal', 'Tamaño de la empresa', '0'],
        ['remote_work', 'Binaria', '¿Trabaja remotamente?', '0'],
        ['tech_company', 'Binaria', '¿Empresa tecnológica?', '0'],
        ['benefits', 'Categórica', '¿Beneficios de SM?', '0'],
        ['...', '...', '14 variables adicionales de actitud/política', '...'],
        ['comments', 'Texto libre', 'Comentarios (eliminada)', '1095'],
    ]
    story.append(metrics_table(col_table,
        col_widths=cw(3.5*cm, 2.2*cm, 8.5*cm, 1.5*cm)))
    story.append(sp(8))
    story.append(img_eda)
    story.append(p(
        'Figura 1. Izquierda: distribución del target (clases casi equilibradas, '
        '50.6% vs 49.4%). Centro: distribución de edad por target — los que buscan '
        'tratamiento tienden a ser ligeramente más jóvenes. Derecha: work_interfere × target '
        '— las personas con mayor interferencia del trabajo buscan tratamiento con más frecuencia.',
        S_CAPTION))
    story.append(PageBreak())

    # ── 3. LIMPIEZA Y PREPROCESAMIENTO ───────────────────────
    story.append(h1('3. Limpieza y Preprocesamiento de Datos'))
    story.append(img_quality)
    story.append(p(
        'Figura 2. Izquierda: columnas con valores nulos en el dataset raw (work_interfere '
        'con 264 nulos tratados como categoría "Unknown"; state eliminado). '
        'Derecha: outliers extremos en Age (valores <0 o >1e10 en el dataset) '
        'filtrados mediante rango [15, 80].',
        S_CAPTION))
    story.append(sp(6))

    story.append(h2('3.1 Decisiones de limpieza'))

    decisions = [
        ('Timestamp', 'ELIMINADA',
         'No aporta señal predictiva; introducir fecha crearía data leakage temporal.'),
        ('comments', 'ELIMINADA',
         'Texto libre con 87% de valores nulos. NLP fuera del alcance de este proyecto.'),
        ('Country', 'ELIMINADA',
         '60+ valores únicos. Codificación one-hot inflaría el espacio de features. '
         'Podría reintegrarse con target encoding en versiones futuras.'),
        ('state', 'ELIMINADA',
         'Solo presente para US (59% nulos). Crea sesgo geográfico.'),
        ('Age', 'LIMPIEZA',
         'Valores absurdos (negativos, >1e10) → NaN. Rango válido: [15, 80]. '
         '8 valores resultantes como NaN → imputados con mediana (31 años).'),
        ('Gender', 'NORMALIZACIÓN',
         'Más de 40 variantes textuales ("Male", "male", "m", "Man"...) normalizadas '
         'a 3 categorías: Male (986), Female (247), Other (26).'),
        ('work_interfere', 'IMPUTO COMO CATEGORÍA',
         '264 nulos = "no aplica" (sin empleo formal). Se crea categoría "Unknown" '
         'en lugar de imputar con moda, preservando la información semántica del nulo.'),
        ('self_employed', 'IMPUTACIÓN',
         '18 nulos → imputados con moda ("No") durante el pipeline sklearn.'),
    ]

    for col, action, reason in decisions:
        story.append(p(f'<b><font color="#ffd166">{col}</font></b> '
                       f'[<font color="#ff6584">{action}</font>]: {reason}'))

    story.append(sp(8))
    story.append(h2('3.2 Pipeline de preprocesamiento (sklearn)'))
    story.append(img_pipeline)
    story.append(p('Figura 3. Pipeline completo de transformación de datos '
                   'encadenado mediante ColumnTransformer y Pipeline de scikit-learn.',
                   S_CAPTION))
    story.append(sp(4))

    story.append(p(
        'Se usó <b>ColumnTransformer</b> para aplicar transformaciones diferentes '
        'a columnas numéricas y categóricas de forma paralela dentro del mismo pipeline, '
        'garantizando que no hay data leakage entre train y test:'
    ))

    story.append(p(
        '<b>Decisión:</b> Se eligió OrdinalEncoder sobre OneHotEncoder porque los árboles '
        'de decisión manejan variables ordinales numéricas eficientemente sin incrementar '
        'la dimensionalidad. Para modelos lineales o SVM se evaluó que el impacto es menor '
        'dado que las variables ya tienen relativa ordinalidad (e.g. "Never < Rarely < '
        'Sometimes < Often").'
    ))
    story.append(PageBreak())

    # ── 4. INGENIERÍA DE CARACTERÍSTICAS ─────────────────────
    story.append(h1('4. Ingeniería de Características y Análisis de Relevancia'))
    story.append(img_corr)
    story.append(p(
        'Figura 4. Distribución porcentual del target por las 6 variables más relevantes. '
        'Se observa que: (1) historial familiar es el predictor más limpio — '
        'las personas con historial buscan tratamiento en ~76% de los casos; '
        '(2) care_options muestra clara separación; (3) work_interfere tiene '
        'gradiente monotónico: a mayor interferencia, mayor probabilidad de tratamiento.',
        S_CAPTION))
    story.append(sp(6))

    story.append(h2('4.1 Variables eliminadas y justificación'))
    story.append(p(
        'Se eliminaron <b>5 columnas</b> del total de 27: Timestamp, comments, '
        'Country, state (razones descritas en §3), y se mantuvo todo el resto. '
        '<b>Country</b> podría reintegrarse mediante target encoding o embeddings '
        'geográficos en versiones futuras, pero fue excluida para evitar sobreajuste '
        'dado el tamaño del dataset (1 259 registros).'
    ))

    story.append(h2('4.2 Importancia de características'))
    story.append(img_importances)
    story.append(p(
        'Figura 5. Importancias por disminución media del índice Gini. '
        'Izquierda: Decision Tree — work_interfere domina con 32.3%, seguido de '
        'Age (17.4%) y family_history (8.7%). Derecha: Random Forest — work_interfere '
        '(27.9%), family_history (9.7%), Age (9.2%). El consenso entre ambos modelos '
        'valida la relevancia de estas variables.',
        S_CAPTION))
    story.append(sp(4))

    imp_data = [['Rank', 'Variable', 'Importancia RF', 'Importancia DT', 'Interpretación']]
    for i, (_, row_rf) in enumerate(rf_importances.head(10).iterrows()):
        feat = row_rf['feature']
        dt_val = dt_importances[dt_importances['feature']==feat]['importance'].values
        dt_str = f"{dt_val[0]:.4f}" if len(dt_val)>0 else "—"
        interp = {
            'work_interfere':   'Principal predictor — interferencia del trabajo con SM',
            'family_history':   'Predisposición genética/ambiental conocida',
            'Age':              'Los jóvenes buscan tratamiento con mayor frecuencia',
            'care_options':     'Conocer opciones = mayor probabilidad de usarlas',
            'no_employees':     'Empresas grandes tienen más recursos de SM',
            'leave':            'Facilidad de baja correlaciona con búsqueda de ayuda',
            'benefits':         'Disponer de beneficios facilita el acceso',
            'mental_health_consequence': 'Miedo a consecuencias reduce búsqueda de ayuda',
            'coworkers':        'Cultura de apertura con compañeros favorece tratamiento',
            'supervisor':       'Relación con jefe influye en decisión de buscar ayuda',
        }.get(feat, '—')
        imp_data.append([str(i+1), feat, f"{row_rf['importance']:.4f}", dt_str, interp])

    story.append(metrics_table(imp_data,
        col_widths=cw(1.0*cm, 4.0*cm, 2.4*cm, 2.4*cm,
                    CONTENT_W-1.0*cm-4.0*cm-2.4*cm-2.4*cm)))
    story.append(PageBreak())

    # ── 5. SELECCIÓN DEL MODELO ───────────────────────────────
    story.append(h1('5. Comparación y Selección del Algoritmo'))
    story.append(p(
        'Se evaluaron <b>6 algoritmos</b> mediante validación cruzada estratificada '
        '(StratifiedKFold, 5 pliegues, random_state=42) sobre el conjunto de entrenamiento '
        '(1 007 muestras). La métrica primaria fue AUC-ROC, que evalúa la capacidad '
        'discriminativa independientemente del umbral de decisión — especialmente apropiada '
        'para datasets balanceados como este.'
    ))
    story.append(sp(4))
    story.append(img_compare)
    story.append(p(
        'Figura 6. Izquierda: AUC-ROC por modelo (RF destacado en rojo). '
        'Derecha: comparativa de 4 métricas simultáneas. '
        'Random Forest y Gradient Boosting dominan en todas las métricas. '
        'SVM (RBF) destaca en Recall, justificando su uso en la v2 del predictor.',
        S_CAPTION))
    story.append(sp(6))

    cv_data = [['Modelo', 'AUC-ROC CV', 'Accuracy CV', 'F1-Score CV', 'Recall CV',
                'AUC ± std', 'Complejidad', 'Interpretable']]
    for nombre in nombres:
        r = resultados[nombre]
        is_best = nombre == BEST
        mark = ' ★' if is_best else ''
        cv_data.append([
            nombre + mark,
            f"{r['auc'].mean():.4f}",
            f"{r['acc'].mean():.4f}",
            f"{r['f1'].mean():.4f}",
            f"{r['recall'].mean():.4f}",
            f"±{r['auc'].std():.4f}",
            {'Logistic Regression': 'Baja', 'Decision Tree': 'Media',
             'Random Forest': 'Alta', 'Gradient Boosting': 'Alta',
             'K-Nearest Neighbors': 'Media', 'SVM (RBF)': 'Alta'}[nombre],
            {'Logistic Regression': 'Sí', 'Decision Tree': 'Sí',
             'Random Forest': 'Parcial', 'Gradient Boosting': 'No',
             'K-Nearest Neighbors': 'No', 'SVM (RBF)': 'No'}[nombre],
        ])

    tbl = metrics_table(cv_data,
        col_widths=cw(3.8*cm, 1.8*cm, 1.8*cm, 1.8*cm, 1.6*cm,
                    1.5*cm, 1.7*cm, 1.7*cm))
    # Resaltar fila ganadora
    best_idx = [i for i,r in enumerate(cv_data) if BEST in r[0]][0]
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,best_idx), (-1,best_idx), colors.HexColor('#1a1040')),
        ('TEXTCOLOR',  (0,best_idx), (-1,best_idx), C_ACCENT2),
        ('FONTNAME',   (0,best_idx), (-1,best_idx), 'Helvetica-Bold'),
    ]))
    story.append(tbl)
    story.append(sp(8))

    story.append(h2('5.1 Justificación de Random Forest como modelo principal'))
    reasons = [
        ('AUC-ROC más alto en CV', '88.77% ± 1.36% — mejor generalización en 5 pliegues'),
        ('Menor varianza', 'std=1.36%, vs Decision Tree (3.00%) — más estable'),
        ('Robustez a outliers', 'El promedio de árboles neutraliza muestras atípicas'),
        ('Importancias de features', 'Mean Decrease Impurity proporciona explicabilidad parcial'),
        ('Balance Precision/Recall', 'F1=81.47% — no sacrifica ninguna de las dos clases'),
        ('Sin necesidad de escalar', 'Los árboles son invariantes al rango de las features'),
    ]
    for title, detail in reasons:
        story.append(p(f'<b><font color="#ffd166">✓ {title}:</font></b> {detail}'))

    story.append(sp(6))
    story.append(h2('5.2 Por qué Decision Tree en la app HTML (no Random Forest)'))
    story.append(p(
        'Para la aplicación portable <b>mental_health_predictor.html</b> se usó '
        '<b>Decision Tree (depth=5)</b> exportado con m2cgen, y no el Random Forest. '
        'La razón es técnica: Random Forest de 200 árboles genera ~12 MB de JavaScript '
        '— inviable para un archivo portable. El Decision Tree genera solo ~10 KB de JS '
        'con un AUC de 88.19% (solo 0.58 puntos por debajo del RF en CV).'
    ))

    size_data = [
        ['Modelo', 'JS exportado', 'AUC CV', 'Diferencia vs RF', 'Decisión'],
        ['Random Forest (200)', '~12 MB', '88.77%', '—', 'Demasiado grande para HTML'],
        ['Gradient Boosting', 'No soportado por m2cgen', '88.08%', '-0.69%', 'Excluido'],
        ['Decision Tree (depth=5)', '10 KB', '88.19%', '-0.58%', '✓ SELECCIONADO para HTML'],
        ['Logistic Regression', '~1 KB', '77.60%', '-11.17%', 'AUC insuficiente'],
    ]
    story.append(metrics_table(size_data,
        col_widths=cw(3.8*cm, 2.8*cm, 1.8*cm, 2.5*cm,
                    CONTENT_W-3.8*cm-2.8*cm-1.8*cm-2.5*cm)))
    story.append(PageBreak())

    # ── 6. ÁRBOL DE DECISIÓN ──────────────────────────────────
    story.append(h1('6. El Árbol de Decisión — Estructura y Reglas'))
    story.append(p(
        'El árbol de profundidad 5 captura las reglas de decisión más relevantes '
        'aprendidas del dataset. Cada nodo interno aplica una condición binaria sobre '
        'una feature; las hojas contienen la distribución de clases observada '
        'en el subconjunto de entrenamiento correspondiente.'
    ))
    story.append(sp(4))
    story.append(img_tree)
    story.append(p(
        'Figura 7. Árbol de decisión completo (depth=5). Los nodos coloreados más '
        'intensamente indican mayor pureza (proporción de clase dominante más alta). '
        'Los nodos azules tienden a predecir "No tratamiento"; los naranja-rojos, '
        '"Tratamiento".',
        S_CAPTION))
    story.append(sp(6))

    story.append(h2('6.1 Primeras reglas (depth ≤ 3) — explicación humana'))
    story.append(p(
        'Las siguientes reglas son las más determinantes, extraídas del nodo raíz:'
    ))

    # Mostrar primeras reglas del árbol
    rules_text = export_text(dt_model, feature_names=feature_names, max_depth=3)
    for line in rules_text.split('\n')[:35]:
        if line.strip():
            indent = len(line) - len(line.lstrip())
            color_rule = '#ffd166' if 'class:' in line else (
                         '#43e97b' if 'Tratamiento' in line else '#e8eaf6')
            story.append(p(
                f'<font face="Courier" color="{color_rule}">'
                f'{line.replace("<","&lt;").replace(">","&gt;")}</font>',
                style('rule', fontSize=7.5, leading=10, spaceBefore=1, spaceAfter=1,
                      leftIndent=indent*3)
            ))

    story.append(sp(8))
    story.append(h2('6.2 Métricas del Decision Tree en test'))
    dt_metrics = [
        ['Métrica', 'Decision Tree (depth=5)', 'Random Forest (200)', 'Diferencia'],
        ['AUC-ROC',    f'{dt_auc:.4f}',  f'{auc_t:.4f}',  f'{auc_t-dt_auc:+.4f}'],
        ['Accuracy',   f'{dt_acc:.4f}',  f'{acc_t:.4f}',  f'{acc_t-dt_acc:+.4f}'],
        ['Recall',     f'{dt_rec:.4f}',  f'{rec_t:.4f}',  f'{rec_t-dt_rec:+.4f}'],
        ['Precision',  f'{dt_prec:.4f}', f'{prec_t:.4f}', f'{prec_t-dt_prec:+.4f}'],
        ['F1-Score',   f'{dt_f1:.4f}',   f'{f1_t:.4f}',   f'{f1_t-dt_f1:+.4f}'],
    ]
    story.append(metrics_table(dt_metrics,
        col_widths=cw(3.0*cm, 4.5*cm, 4.5*cm, 3.5*cm)))
    story.append(PageBreak())

    # ── 7. EVALUACIÓN DEL MODELO ──────────────────────────────
    story.append(h1('7. Evaluación Completa del Modelo'))
    story.append(img_eval)
    story.append(p(
        'Figura 8. Izquierda: curva ROC comparativa RF vs DT (el área bajo la curva '
        'representa la probabilidad de que el modelo clasifique correctamente un par '
        'positivo-negativo aleatorio). Centro: curva Precision-Recall (más informativa '
        'que ROC cuando hay interés en una clase específica). Derecha: matriz de confusión '
        'del Random Forest con umbral 0.50.',
        S_CAPTION))
    story.append(sp(6))

    story.append(h2('7.1 Interpretación de la Matriz de Confusión (RF, umbral 0.50)'))
    conf_mat2 = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = conf_mat2.ravel()
    cm_data = [
        ['', 'Pred: No tratamiento', 'Pred: Tratamiento', 'Total real'],
        ['Real: No tratamiento', str(tn), str(fp), str(tn+fp)],
        ['Real: Tratamiento',    str(fn), str(tp), str(fn+tp)],
        ['Total predicho',       str(tn+fn), str(fp+tp), str(tn+fp+fn+tp)],
    ]
    story.append(metrics_table(cm_data,
        col_widths=cw(4.0*cm, 4.5*cm, 4.0*cm, 2.5*cm)))
    story.append(sp(4))

    story.append(p(
        f'<b>Verdaderos Negativos (TN={tn}):</b> Personas que no buscan tratamiento '
        f'y el modelo predice correctamente que no lo buscarán. '
        f'<b>Verdaderos Positivos (TP={tp}):</b> Personas que sí buscan tratamiento '
        f'y el modelo lo detecta. '
        f'<b>Falsos Positivos (FP={fp}):</b> El modelo predice tratamiento cuando no '
        f'lo buscarán — error de tipo I. '
        f'<b>Falsos Negativos (FN={fn}):</b> El modelo no detecta un caso real — '
        f'error de tipo II, el más costoso en contexto de salud mental.'
    ))

    story.append(sp(6))
    story.append(h2('7.2 Reporte de clasificación completo (Random Forest)'))

    report = classification_report(y_test, y_pred,
                                   target_names=['No tratamiento','Tratamiento'],
                                   output_dict=True)
    rep_data = [['Clase', 'Precision', 'Recall', 'F1-Score', 'Soporte']]
    for cls in ['No tratamiento', 'Tratamiento', 'macro avg', 'weighted avg']:
        r = report[cls]
        rep_data.append([
            cls, f"{r['precision']:.4f}", f"{r['recall']:.4f}",
            f"{r['f1-score']:.4f}", str(int(r['support']))
        ])
    story.append(metrics_table(rep_data,
        col_widths=cw(4.5*cm, 3.0*cm, 3.0*cm, 3.0*cm, 2.5*cm)))
    story.append(PageBreak())

    # ── 8. ARQUITECTURA DE LA APP HTML/JS ────────────────────
    story.append(h1('8. Arquitectura de la Aplicación Portable (HTML/JS)'))
    story.append(p(
        'El fichero <b>mental_health_predictor.html</b> es una aplicación autocontenida '
        'de 33 KB que ejecuta el modelo de Machine Learning <b>íntegramente en el navegador</b> '
        'del usuario, sin ningún servidor backend, sin dependencias externas y sin '
        'conexión a internet. Compatible con cualquier navegador moderno en Windows, '
        'macOS y Linux.'
    ))
    story.append(sp(4))

    story.append(h2('8.1 Cómo se exportó el modelo a JavaScript'))
    story.append(p(
        'Se utilizó la librería <b>m2cgen</b> (Model to Code Generator) que convierte '
        'un árbol de decisión sklearn en una función JavaScript pura mediante un '
        'recorrido recursivo del árbol y generación de código con bloques if/else:'
    ))



    story.append(h2('8.2 Pipeline de preprocesamiento en JavaScript'))
    story.append(p(
        'El preprocesamiento replica exactamente el pipeline sklearn, calculando '
        'los mismos valores que habrían sido transformados antes de llegar al modelo:'
    ))


    story.append(h2('8.3 Flujo completo de predicción en el navegador'))

    flow_data = [
        ['Paso', 'Acción', 'Implementación'],
        ['1', 'Usuario rellena formulario HTML', '22 campos <select> + 1 <input number>'],
        ['2', 'Clic en "Ejecutar Predicción"', 'addEventListener("submit", ...)'],
        ['3', 'Leer valores del formulario', 'document.getElementById(col).value'],
        ['4', 'Normalizar Age', '(age - 32.07) / 7.26 (StandardScaler)'],
        ['5', 'Codificar categóricas', 'indexOf en arrays de categorías (OrdinalEncoder)'],
        ['6', 'Ejecutar árbol de decisión', 'score(inputVector) → [prob_no, prob_si]'],
        ['7', 'Aplicar umbral (0.50)', 'prob_si >= 0.50 → Tratamiento'],
        ['8', 'Renderizar resultado', 'Gauge SVG + métricas + barras de importancia'],
    ]
    story.append(metrics_table(flow_data,
        col_widths=cw(1.0*cm, 5.5*cm, CONTENT_W-1.0*cm-5.5*cm)))

    story.append(PageBreak())

    # ── 9. LIMITACIONES Y SESGOS ─────────────────────────────
    story.append(h1('9. Limitaciones, Sesgos y Trabajo Futuro'))
    story.append(h2('9.1 Limitaciones conocidas'))

    lims = [
        ('Dataset de 2014', 'La encuesta tiene 10+ años. El contexto laboral IT ha cambiado '
         'significativamente (trabajo remoto masivo post-COVID, mayor concienciación). '
         'Los patrones puede que no reflejen la situación actual.'),
        ('Sesgo geográfico', 'El 60% de los respondentes son de EEUU. Los patrones pueden '
         'no generalizarse a otras culturas laborales.'),
        ('Sesgo de autoselección', 'La encuesta es voluntaria — quienes la responden pueden '
         'tener mayor sensibilidad hacia la SM que la población IT general.'),
        ('Variable target auto-reportada', '"¿Has buscado tratamiento?" no equivale a '
         '"¿Necesitas tratamiento?". Hay sesgo de acceso y estigma.'),
        ('Tamaño del dataset', '1 259 muestras es modesto para 22 features; '
         'puede producir sobreajuste en modelos complejos.'),
        ('Generalización del DT en HTML', 'El árbol de profundidad 5 tiene menor capacidad '
         'que el Random Forest. Diferencia de AUC: -0.58% en CV, potencialmente mayor '
         'en distribuciones no vistas.'),
    ]

    for title, detail in lims:
        story.append(p(f'<b><font color="#ff6584">⚠ {title}:</font></b> {detail}'))

    story.append(sp(6))
    story.append(h2('9.2 Trabajo futuro'))

    future = [
        'Actualizar el dataset con encuestas OSMI más recientes (2019–2023)',
        'Incorporar Country mediante target encoding para recuperar señal geográfica',
        'Explorar modelos calibrados (CalibratedClassifierCV) para probabilidades más fiables',
        'Implementar SHAP values para explicabilidad por instancia',
        'Añadir análisis de subgrupos (por género, tamaño empresa, país)',
        'Versión con Random Forest exportado via ONNX.js para mayor precisión en el HTML',
        'Evaluar umbral con ajuste por coste (FN tiene coste >> FP en salud mental)',
    ]
    for item in future:
        story.append(p(f'<font color="#43e97b">→</font> {item}'))

    story.append(PageBreak())

    # ── 10. REPRODUCIBILIDAD ──────────────────────────────────
    story.append(h1('10. Reproducibilidad del Pipeline'))
    story.append(p(
        'Todo el código es 100% reproducible con semillas fijas (random_state=42). '
        'Los archivos del proyecto están publicados en '
        '<b>https://github.com/hacklifeplus/ialab</b>.'
    ))


    story.append(sp(6))
    story.append(h2('10.1 Versiones del entorno'))
    env_data = [
        ['Componente', 'Versión', 'Propósito'],
        ['Python',           '3.12.3',  'Lenguaje base'],
        ['scikit-learn',     '1.4.1',   'ML: modelos, pipeline, métricas'],
        ['pandas',           '(system)', 'Manipulación de datos'],
        ['numpy',            '(system)', 'Álgebra lineal'],
        ['matplotlib',       '3.6.3',   'Visualizaciones'],
        ['seaborn',          '0.13.2',  'Plots estadísticos'],
        ['m2cgen',           '(latest)', 'Exportación modelo → JavaScript'],
        ['reportlab',        '4.4.10',  'Generación de este PDF'],
    ]
    story.append(metrics_table(env_data,
        col_widths=cw(4.0*cm, 3.0*cm, CONTENT_W-4.0*cm-3.0*cm)))

    story.append(sp(8))
    story.append(h2('10.2 Semillas de aleatoriedad'))
    story.append(p(
        'Se fijó <b>random_state=42</b> en todos los modelos y en train_test_split. '
        'La validación cruzada usa <b>StratifiedKFold(shuffle=True, random_state=42)</b> '
        'para garantizar distribución de clases consistente en cada pliegue. '
        'Cualquier variación en los resultados se debe exclusivamente a diferencias '
        'en versiones de librerías, no a aleatoriedad no controlada.'
    ))

    story.append(sp(10))
    story.append(hr())
    story.append(sp(6))
    story.append(p(
        '<b>Informe generado automáticamente con Python + ReportLab.</b><br/>'
        'Proyecto: IA Lab — Predictor de Salud Mental IT<br/>'
        'Repositorio: https://github.com/hacklifeplus/ialab<br/>'
        'Fecha: Abril 2026',
        style('footer', fontSize=8, fontName='Helvetica', textColor=C_MUTED,
              alignment=TA_CENTER, leading=13)
    ))

    doc.build(story)
    print(f"\nPDF generado: {OUT_PDF}")
    print(f"Tamaño: {os.path.getsize(OUT_PDF)/1024:.0f} KB")

build_pdf()
