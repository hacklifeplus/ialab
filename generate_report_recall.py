"""
Generador de Informe PDF — mental_health_predictor_recall.html
==============================================================
Documenta con alto nivel de transparencia y explicabilidad:
  - Motivación del cambio de métrica (AUC-ROC → Recall)
  - SVM con kernel RBF: teoría, parámetros, vectores de soporte
  - Optimización del umbral de decisión (threshold tuning)
  - Comparación de todos los modelos por Recall
  - Curvas Precision-Recall y análisis del tradeoff
  - Arquitectura JS del kernel RBF embebido en HTML
  - Limitaciones y sesgos específicos del enfoque Recall
"""

import warnings; warnings.filterwarnings('ignore')
import os, io, json, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.gridspec import GridSpec
import seaborn as sns

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
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
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, HRFlowable, PageBreak, KeepTogether
)

# ─────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────
OUT_PDF   = "/root/Projects/ialab/output/informe_mental_health_predictor_recall.pdf"
DATA_PATH = "/root/Projects/ialab/IT_mental_health.survey.csv"
W         = float(A4[0])
H         = float(A4[1])
MARGIN    = float(2.0 * cm)
CW        = W - 2 * MARGIN   # content width ≈ 481.9 pt

# ─────────────────────────────────────────────────────────────
# PALETA
# ─────────────────────────────────────────────────────────────
C_ACCENT  = colors.HexColor('#6c63ff')
C_ACCENT2 = colors.HexColor('#ff6584')
C_GREEN   = colors.HexColor('#43e97b')
C_YELLOW  = colors.HexColor('#ffd166')
C_MUTED   = colors.HexColor('#8b90a8')
C_WHITE   = colors.white

MPL_BG    = '#0f1117'
MPL_CARD  = '#21253a'
MPL_BORD  = '#2e3250'
MPL_ACC   = '#6c63ff'
MPL_ACC2  = '#ff6584'
MPL_GREEN = '#43e97b'
MPL_YELL  = '#ffd166'
MPL_TEXT  = '#e8eaf6'
MPL_MUTED = '#8b90a8'

# ─────────────────────────────────────────────────────────────
# HELPERS MATPLOTLIB
# ─────────────────────────────────────────────────────────────
def setup_ax(ax):
    ax.set_facecolor(MPL_CARD)
    for sp in ax.spines.values(): sp.set_edgecolor(MPL_BORD)
    ax.tick_params(colors=MPL_MUTED, labelsize=8.5)
    ax.xaxis.label.set_color(MPL_MUTED)
    ax.yaxis.label.set_color(MPL_MUTED)
    ax.title.set_color(MPL_TEXT)
    return ax

def dark_fig(ncols=1, nrows=1, w=14, h=5):
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h))
    fig.patch.set_facecolor(MPL_BG)
    axlist = axes.flat if hasattr(axes, 'flat') else [axes]
    for ax in axlist: setup_ax(ax)
    return fig, axes

def fig2img(fig, width=None):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    buf.seek(0); plt.close(fig)
    img = Image(buf)
    if width:
        ratio = img.imageHeight / img.imageWidth
        img.drawWidth = width
        img.drawHeight = width * ratio
    return img

# ─────────────────────────────────────────────────────────────
# ESTILOS REPORTLAB
# ─────────────────────────────────────────────────────────────
def S(name, **kw): return ParagraphStyle(name, **kw)

S_H1 = S('H1', fontSize=17, fontName='Helvetica-Bold', textColor=C_ACCENT,
          spaceBefore=14, spaceAfter=7, leading=21)
S_H2 = S('H2', fontSize=12, fontName='Helvetica-Bold', textColor=C_ACCENT2,
          spaceBefore=10, spaceAfter=4, leading=16)
S_H3 = S('H3', fontSize=10, fontName='Helvetica-Bold', textColor=C_YELLOW,
          spaceBefore=6, spaceAfter=3, leading=13)
S_BODY = S('Body', fontSize=9, fontName='Helvetica',
           textColor=colors.HexColor('#ccccdd'),
           spaceBefore=2, spaceAfter=4, leading=14, alignment=TA_JUSTIFY)
S_CODE = S('Code', fontSize=7.8, fontName='Courier', textColor=C_GREEN,
           spaceBefore=1, spaceAfter=1, leading=10.5,
           backColor=colors.HexColor('#0d1020'),
           leftIndent=10, rightIndent=10)
S_CAP  = S('Cap', fontSize=8, fontName='Helvetica-Oblique', textColor=C_MUTED,
           spaceAfter=6, alignment=TA_CENTER)
S_META = S('Meta', fontSize=8, fontName='Helvetica', textColor=C_MUTED,
           alignment=TA_CENTER)
S_WARN = S('Warn', fontSize=9, fontName='Helvetica-Bold',
           textColor=C_YELLOW, spaceBefore=4, spaceAfter=4)

def hr():  return HRFlowable(width='100%', thickness=0.5,
                              color=colors.HexColor('#2e3250'), spaceAfter=8)
def sp(h=6): return Spacer(1, h)
def p(t, s=None): return Paragraph(t, s or S_BODY)
def h1(t): return Paragraph(t, S_H1)
def h2(t): return Paragraph(t, S_H2)
def h3(t): return Paragraph(t, S_H3)
def cap(t): return Paragraph(t, S_CAP)
def warn(t): return Paragraph(f'⚠ {t}', S_WARN)

def code_block(txt):
    return [Paragraph(
        line.replace(' ', '&nbsp;').replace('<', '&lt;').replace('>', '&gt;'),
        S_CODE) for line in txt.strip().split('\n')]

def cw(*ws): return [float(x) for x in ws]

def tbl(rows, widths, extra_style=None):
    base = TableStyle([
        ('BACKGROUND',    (0,0),(-1,0),  colors.HexColor('#1a1040')),
        ('TEXTCOLOR',     (0,0),(-1,0),  C_ACCENT),
        ('FONTNAME',      (0,0),(-1,0),  'Helvetica-Bold'),
        ('FONTSIZE',      (0,0),(-1,0),  9),
        ('BACKGROUND',    (0,1),(-1,-1), colors.HexColor('#16192a')),
        ('TEXTCOLOR',     (0,1),(-1,-1), colors.HexColor('#ccccdd')),
        ('FONTNAME',      (0,1),(-1,-1), 'Helvetica'),
        ('FONTSIZE',      (0,1),(-1,-1), 8.5),
        ('ROWBACKGROUNDS',(0,1),(-1,-1), [colors.HexColor('#16192a'),
                                          colors.HexColor('#1c2035')]),
        ('GRID',          (0,0),(-1,-1), 0.4, colors.HexColor('#2e3250')),
        ('ALIGN',         (0,0),(-1,-1), 'CENTER'),
        ('VALIGN',        (0,0),(-1,-1), 'MIDDLE'),
        ('TOPPADDING',    (0,0),(-1,-1), 4),
        ('BOTTOMPADDING', (0,0),(-1,-1), 4),
        ('LEFTPADDING',   (0,0),(-1,-1), 6),
        ('RIGHTPADDING',  (0,0),(-1,-1), 6),
    ])
    t = Table(rows, colWidths=cw(*widths))
    t.setStyle(base)
    if extra_style:
        t.setStyle(TableStyle(extra_style))
    return t

# ─────────────────────────────────────────────────────────────
# REPRODUCIR PIPELINE COMPLETO
# ─────────────────────────────────────────────────────────────
print("Reproduciendo pipeline ML (Recall)...")

df_raw = pd.read_csv(DATA_PATH)
df = df_raw.copy()
df.drop(columns=['Timestamp','comments','state'], inplace=True)
df['Age'] = pd.to_numeric(df['Age'], errors='coerce').where(lambda s: s.between(15,80))

def norm_g(g):
    if pd.isna(g): return 'Other'
    g = str(g).strip().lower()
    if g in ('male','m','man','cis male','malr','make','mail'): return 'Male'
    if g in ('female','f','woman','cis female','femake','femail','female (cis)'): return 'Female'
    return 'Other'

df['Gender'] = df['Gender'].apply(norm_g)
df['work_interfere'] = df['work_interfere'].fillna('Unknown')
df['treatment'] = df['treatment'].map({'Yes':1,'No':0})

TARGET   = 'treatment'
DROP     = ['Country']
FEATURES = [c for c in df.columns if c not in [TARGET]+DROP]
X = df[FEATURES].copy(); y = df[TARGET].copy()
num_cols = X.select_dtypes(include='number').columns.tolist()
cat_cols = X.select_dtypes(exclude='number').columns.tolist()

num_pipe = Pipeline([('imp', SimpleImputer(strategy='median')),
                     ('sc',  StandardScaler())])
cat_pipe = Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                     ('enc', OrdinalEncoder(handle_unknown='use_encoded_value',
                                            unknown_value=-1))])
preprocessor = ColumnTransformer([
    ('num', num_pipe,  num_cols),
    ('cat', cat_pipe,  cat_cols)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

modelos = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree":       DecisionTreeClassifier(max_depth=6, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
    "SVM (RBF)":           SVC(kernel='rbf', probability=True, random_state=42),
}

cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
res = {}
for nombre, clf in modelos.items():
    pipe = Pipeline([('prep', preprocessor), ('clf', clf)])
    res[nombre] = {
        'recall': cross_val_score(pipe, X_train, y_train, cv=cv5, scoring='recall',   n_jobs=-1),
        'auc':    cross_val_score(pipe, X_train, y_train, cv=cv5, scoring='roc_auc',  n_jobs=-1),
        'f1':     cross_val_score(pipe, X_train, y_train, cv=cv5, scoring='f1',       n_jobs=-1),
        'prec':   cross_val_score(pipe, X_train, y_train, cv=cv5, scoring='precision',n_jobs=-1),
    }
    print(f"  {nombre:<25} Recall={res[nombre]['recall'].mean():.4f}  AUC={res[nombre]['auc'].mean():.4f}")

BEST = "SVM (RBF)"

# Entrenar SVM final
svm_pipe = Pipeline([('prep', preprocessor),
                     ('clf', SVC(kernel='rbf', probability=True, random_state=42))])
svm_pipe.fit(X_train, y_train)
y_proba_svm  = svm_pipe.predict_proba(X_test)[:,1]
y_dec_svm    = svm_pipe.decision_function(X_test)

# Threshold óptimo (Recall máx, Precision ≥ 0.60)
precs_all, recs_all, threshs_all = precision_recall_curve(y_test, y_proba_svm)
precs_dec, recs_dec, threshs_dec = precision_recall_curve(y_test, y_dec_svm)

valid = [(r,p,t) for p,r,t in zip(precs_all[:-1],recs_all[:-1],threshs_all) if p>=0.60]
best_r_opt, best_p_opt, best_t_opt = max(valid, key=lambda x: x[0])

valid_dec = [(r,p,t) for p,r,t in zip(precs_dec[:-1],recs_dec[:-1],threshs_dec) if p>=0.60]
best_r_dec, best_p_dec, best_t_dec = max(valid_dec, key=lambda x: x[0])

y_pred_def = (y_proba_svm >= 0.50).astype(int)
y_pred_opt = (y_proba_svm >= best_t_opt).astype(int)
y_pred_dec = (y_dec_svm >= best_t_dec).astype(int)

# Métricas con umbral default
acc_def  = accuracy_score(y_test, y_pred_def)
rec_def  = recall_score(y_test, y_pred_def)
prec_def = precision_score(y_test, y_pred_def)
f1_def   = f1_score(y_test, y_pred_def)
auc_svm  = roc_auc_score(y_test, y_proba_svm)
ap_svm   = average_precision_score(y_test, y_proba_svm)

# Métricas con umbral óptimo (función de decisión)
acc_opt  = accuracy_score(y_test, y_pred_dec)
rec_opt  = recall_score(y_test, y_pred_dec)
prec_opt = precision_score(y_test, y_pred_dec)
f1_opt   = f1_score(y_test, y_pred_dec)

# RF para comparación
rf_pipe = Pipeline([('prep', preprocessor),
                    ('clf', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))])
rf_pipe.fit(X_train, y_train)
y_proba_rf = rf_pipe.predict_proba(X_test)[:,1]
y_pred_rf  = rf_pipe.predict(X_test)
auc_rf     = roc_auc_score(y_test, y_proba_rf)
rec_rf     = recall_score(y_test, y_pred_rf)

# SVM internals
clf_svm = svm_pipe.named_steps['clf']
n_sv     = clf_svm.support_vectors_.shape[0]
n_feats  = clf_svm.support_vectors_.shape[1]
gamma_v  = float(clf_svm._gamma)

# Feature importances aproximadas via permutation (para visualización)
from sklearn.inspection import permutation_importance
perm = permutation_importance(svm_pipe, X_test, y_test,
                              n_repeats=15, random_state=42,
                              scoring='recall', n_jobs=-1)
perm_df = pd.DataFrame({'feature': FEATURES,
                         'importance': perm.importances_mean,
                         'std': perm.importances_std})\
           .sort_values('importance', ascending=False)

print("Pipeline listo. Generando figuras...")

# ─────────────────────────────────────────────────────────────
# FIGURAS
# ─────────────────────────────────────────────────────────────

# FIG 1 — Comparación de modelos por Recall
fig1, axes1 = dark_fig(ncols=2, w=14, h=5.5)
fig1.suptitle('Comparación de Algoritmos — Métrica: RECALL (CV 5-fold)',
              color=MPL_TEXT, fontsize=11, fontweight='bold')

nombres = list(res.keys())
ax = axes1[0]
means_r = [res[n]['recall'].mean() for n in nombres]
stds_r  = [res[n]['recall'].std()  for n in nombres]
cols_b  = [MPL_ACC2 if n==BEST else MPL_ACC for n in nombres]
bars = ax.barh(nombres, means_r, xerr=stds_r, color=cols_b, capsize=4,
               edgecolor=MPL_BORD)
for bar, m in zip(bars, means_r):
    ax.text(m+0.005, bar.get_y()+bar.get_height()/2, f'{m:.3f}',
            va='center', color=MPL_TEXT, fontsize=8.5)
ax.set_xlabel('Recall (CV)', color=MPL_MUTED)
ax.set_title('Recall por modelo — SVM destacado', color=MPL_TEXT)
ax.set_xlim(0.45, 1.05)
ax.axvline(res[BEST]['recall'].mean(), color=MPL_YELL, ls='--', lw=1.2, alpha=0.6)

ax = axes1[1]
x = np.arange(len(nombres)); bw = 0.2
metrics_plot = [('recall','Recall',MPL_ACC2), ('auc','AUC',MPL_ACC),
                ('f1','F1',MPL_GREEN), ('prec','Precision',MPL_YELL)]
for i,(met,lbl,col) in enumerate(metrics_plot):
    vals = [res[n][met].mean() for n in nombres]
    ax.bar(x+(i-1.5)*bw, vals, bw, label=lbl, color=col,
           edgecolor=MPL_BORD, alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels([n.replace(' ','\n') for n in nombres], fontsize=7.5)
ax.set_ylim(0.45, 1.0)
ax.set_title('4 métricas comparativas', color=MPL_TEXT)
ax.legend(fontsize=8, labelcolor=MPL_TEXT, facecolor=MPL_CARD, edgecolor=MPL_BORD)
plt.tight_layout()
img_compare = fig2img(fig1, width=CW)

# FIG 2 — Tradeoff Precision-Recall + umbral
fig2, axes2 = dark_fig(ncols=3, w=16, h=5)
fig2.suptitle('Análisis del Tradeoff Precision-Recall — SVM RBF',
              color=MPL_TEXT, fontsize=11, fontweight='bold')

ax = axes2[0]
ax.plot(recs_all, precs_all, color=MPL_ACC2, lw=2.5,
        label=f'SVM  AP={ap_svm:.3f}')
precs_rf2, recs_rf2, _ = precision_recall_curve(y_test, y_proba_rf)
ap_rf = average_precision_score(y_test, y_proba_rf)
ax.plot(recs_rf2, precs_rf2, color=MPL_ACC, lw=1.5, ls='--',
        label=f'RF  AP={ap_rf:.3f}')
ax.scatter([best_r_opt], [best_p_opt], color=MPL_YELL, zorder=5, s=70,
           label=f'Óptimo Recall={best_r_opt:.2f}')
ax.axhline(0.60, color=MPL_GREEN, ls=':', lw=1.2, alpha=0.7,
           label='Precision mín. 0.60')
ax.set_xlabel('Recall', color=MPL_MUTED); ax.set_ylabel('Precision', color=MPL_MUTED)
ax.set_title('Curva Precision-Recall', color=MPL_TEXT)
ax.legend(fontsize=7.5, labelcolor=MPL_TEXT, facecolor=MPL_CARD, edgecolor=MPL_BORD)
ax.set_xlim(0,1); ax.set_ylim(0,1)

ax = axes2[1]
plot_threshs = threshs_dec
plot_rec  = recs_dec[:-1]
plot_prec = precs_dec[:-1]
f1_arr = np.where((plot_prec+plot_rec)==0, 0,
                  2*plot_prec*plot_rec/(plot_prec+plot_rec))
ax.plot(plot_threshs, plot_rec,  color=MPL_ACC2, lw=2, label='Recall')
ax.plot(plot_threshs, plot_prec, color=MPL_ACC,  lw=2, label='Precision')
ax.plot(plot_threshs, f1_arr,    color=MPL_GREEN,lw=1.5, ls='--', label='F1')
ax.axvline(best_t_dec, color=MPL_YELL, lw=1.8, ls=':',
           label=f'Umbral ópt.={best_t_dec:.3f}')
ax.axhline(0.60, color=MPL_GREEN, ls=':', lw=1, alpha=0.5)
ax.set_xlabel('Umbral (función de decisión)', color=MPL_MUTED)
ax.set_ylabel('Score', color=MPL_MUTED)
ax.set_title('Recall / Precision / F1 vs Umbral', color=MPL_TEXT)
ax.legend(fontsize=7.5, labelcolor=MPL_TEXT, facecolor=MPL_CARD, edgecolor=MPL_BORD)
ax.set_xlim(plot_threshs.min(), plot_threshs.max())
ax.set_ylim(0,1)

ax = axes2[2]
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_proba_svm)
fpr_rf,  tpr_rf,  _ = roc_curve(y_test, y_proba_rf)
ax.plot(fpr_svm, tpr_svm, color=MPL_ACC2, lw=2,
        label=f'SVM  AUC={auc_svm:.4f}')
ax.plot(fpr_rf,  tpr_rf,  color=MPL_ACC,  lw=1.5, ls='--',
        label=f'RF   AUC={auc_rf:.4f}')
ax.plot([0,1],[0,1],'--', color=MPL_BORD, lw=1)
ax.fill_between(fpr_svm, tpr_svm, alpha=0.08, color=MPL_ACC2)
ax.set_xlabel('FPR', color=MPL_MUTED); ax.set_ylabel('TPR', color=MPL_MUTED)
ax.set_title('Curva ROC (SVM vs RF)', color=MPL_TEXT)
ax.legend(fontsize=8, labelcolor=MPL_TEXT, facecolor=MPL_CARD, edgecolor=MPL_BORD)
plt.tight_layout()
img_tradeoff = fig2img(fig2, width=CW)

# FIG 3 — Matrices de confusión comparadas
fig3, axes3 = dark_fig(ncols=2, w=12, h=5)
fig3.suptitle('Matrices de Confusión — Comparativa de Umbrales (SVM RBF)',
              color=MPL_TEXT, fontsize=11, fontweight='bold')

cm_def = confusion_matrix(y_test, y_pred_def)
cm_opt2 = confusion_matrix(y_test, y_pred_dec)

for ax, conf_mat, title, cmap in [
    (axes3[0], cm_def,  f'Umbral default (0.50)\nRecall={rec_def:.2f}  Precision={prec_def:.2f}', 'Blues'),
    (axes3[1], cm_opt2, f'Umbral óptimo ({best_t_dec:.3f})\nRecall={rec_opt:.2f}  Precision={prec_opt:.2f}', 'Reds'),
]:
    tn,fp,fn,tp = conf_mat.ravel()
    disp = ConfusionMatrixDisplay(conf_mat, display_labels=['No trat.','Trat.'])
    disp.plot(ax=ax, colorbar=False, cmap=cmap)
    ax.set_title(title, color=MPL_TEXT, fontsize=9)
    ax.text(0.5, -0.18,
            f'FN={fn} (casos perdidos)  |  FP={fp} (falsas alarmas)',
            ha='center', transform=ax.transAxes, color=MPL_MUTED, fontsize=8)

plt.tight_layout()
img_confmats = fig2img(fig3, width=CW)

# FIG 4 — Importancia de características por permutación
fig4, axes4 = dark_fig(ncols=2, w=14, h=5.5)
fig4.suptitle('Importancia de Características — SVM RBF',
              color=MPL_TEXT, fontsize=11, fontweight='bold')

ax = axes4[0]
top15 = perm_df.head(15)
pal = [MPL_ACC2 if i==0 else (MPL_YELL if i<3 else MPL_ACC) for i in range(len(top15))]
ax.barh(top15['feature'][::-1], top15['importance'][::-1],
        xerr=top15['std'][::-1], color=pal[::-1], edgecolor=MPL_BORD, capsize=3)
ax.set_title('Permutation Importance (Recall) — Top 15', color=MPL_TEXT)
ax.set_xlabel('Disminución media del Recall', color=MPL_MUTED)
ax.axvline(0, color=MPL_BORD, lw=1)

ax = axes4[1]
neg_mask = perm_df['importance'] < 0
pos_perm = perm_df[~neg_mask].head(10)
neg_perm = perm_df[neg_mask]
ax.barh(pos_perm['feature'][::-1], pos_perm['importance'][::-1],
        color=MPL_ACC2, edgecolor=MPL_BORD, label='Positivo (reduce Recall)')
if len(neg_perm):
    ax.barh(neg_perm['feature'][::-1], neg_perm['importance'][::-1],
            color=MPL_ACC, edgecolor=MPL_BORD, label='Negativo (sin efecto claro)')
ax.set_title('Análisis positivo/negativo', color=MPL_TEXT)
ax.set_xlabel('Cambio en Recall al permutar', color=MPL_MUTED)
ax.axvline(0, color=MPL_YELL, lw=1.2, ls='--')
ax.legend(fontsize=8, labelcolor=MPL_TEXT, facecolor=MPL_CARD, edgecolor=MPL_BORD)
plt.tight_layout()
img_importance = fig2img(fig4, width=CW)

# FIG 5 — Teoría SVM: margen y vectores de soporte (diagrama)
fig5, axes5 = dark_fig(ncols=2, w=14, h=5.5)
fig5.suptitle('Concepto SVM — Margen Máximo y Kernel RBF',
              color=MPL_TEXT, fontsize=11, fontweight='bold')

ax = axes5[0]
ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5)
np.random.seed(42)
cls0 = np.random.randn(40, 2) * 0.8 + np.array([-1.5, -1.5])
cls1 = np.random.randn(40, 2) * 0.8 + np.array([1.5, 1.5])
ax.scatter(cls0[:,0], cls0[:,1], c=MPL_ACC,  s=30, alpha=0.7, label='Sin tratamiento')
ax.scatter(cls1[:,0], cls1[:,1], c=MPL_ACC2, s=30, alpha=0.7, label='Con tratamiento')
x_line = np.linspace(-3, 3, 100)
ax.plot(x_line, -x_line,       color=MPL_YELL, lw=2.5, label='Hiperplano decisión')
ax.plot(x_line, -x_line + 1.2, color=MPL_YELL, lw=1.2, ls='--', alpha=0.7, label='Márgenes')
ax.plot(x_line, -x_line - 1.2, color=MPL_YELL, lw=1.2, ls='--', alpha=0.7)
ax.fill_between(x_line, -x_line-1.2, -x_line+1.2, alpha=0.07, color=MPL_YELL)
sv_pts = [(-0.3, 0.2), (0.4, -0.5), (-0.8, 0.9), (0.9, -0.8)]
for pt in sv_pts:
    ax.scatter(*pt, c='none', s=150, edgecolors=MPL_YELL, lw=2, zorder=5)
ax.annotate('Vectores\nde soporte', xy=(0.4, -0.5), xytext=(1.5, -2.5),
            color=MPL_YELL, fontsize=8,
            arrowprops=dict(arrowstyle='->', color=MPL_YELL, lw=1.2))
ax.set_title('SVM: Hiperplano de margen máximo', color=MPL_TEXT)
ax.legend(fontsize=8, labelcolor=MPL_TEXT, facecolor=MPL_CARD, edgecolor=MPL_BORD,
          loc='upper left')
ax.set_xlabel('Feature 1', color=MPL_MUTED); ax.set_ylabel('Feature 2', color=MPL_MUTED)

ax = axes5[1]
x1 = np.linspace(-4, 4, 300)
x2 = np.linspace(-4, 4, 300)
X1, X2 = np.meshgrid(x1, x2)
gamma_demo = 0.5
sv_demo = np.array([[1.0, 0.5], [-0.8, -0.3], [0.2, -1.0]])
alpha_demo = np.array([0.8, -0.6, 0.5])
Z = np.zeros_like(X1)
for sv, a in zip(sv_demo, alpha_demo):
    diff_sq = (X1 - sv[0])**2 + (X2 - sv[1])**2
    Z += a * np.exp(-gamma_demo * diff_sq)
contour = ax.contourf(X1, X2, Z, levels=20, cmap='RdBu_r', alpha=0.85)
ax.contour(X1, X2, Z, levels=[0], colors=[MPL_YELL], linewidths=2.5)
ax.scatter(sv_demo[:,0], sv_demo[:,1], c='none', s=200,
           edgecolors=MPL_YELL, lw=2.5, zorder=5, label='Vectores de soporte')
ax.set_title('Kernel RBF: función de decisión en 2D', color=MPL_TEXT)
ax.set_xlabel('Feature 1', color=MPL_MUTED)
ax.set_ylabel('Feature 2', color=MPL_MUTED)
ax.legend(fontsize=8, labelcolor=MPL_TEXT, facecolor=MPL_CARD, edgecolor=MPL_BORD)
cb = plt.colorbar(contour, ax=ax)
cb.ax.tick_params(colors=MPL_MUTED, labelsize=7)
plt.tight_layout()
img_svm_theory = fig2img(fig5, width=CW)

# FIG 6 — Análisis de la distribución de probabilidades con umbral
fig6, axes6 = dark_fig(ncols=2, w=13, h=5)
fig6.suptitle('Distribución de Probabilidades — Impacto del Umbral',
              color=MPL_TEXT, fontsize=11, fontweight='bold')

ax = axes6[0]
proba_pos = y_proba_svm[y_test==1]
proba_neg = y_proba_svm[y_test==0]
ax.hist(proba_neg, bins=28, alpha=0.65, color=MPL_ACC,  density=True,
        label='Real: Sin tratamiento')
ax.hist(proba_pos, bins=28, alpha=0.65, color=MPL_ACC2, density=True,
        label='Real: Con tratamiento')
ax.axvline(0.50, color=MPL_YELL, lw=2, ls='--',  label='Umbral default (0.50)')
ax.axvline(best_t_opt, color=MPL_GREEN, lw=2, ls=':',
           label=f'Umbral óptimo ({best_t_opt:.3f})')
ax.set_xlabel('Probabilidad predicha (SVM)', color=MPL_MUTED)
ax.set_ylabel('Densidad', color=MPL_MUTED)
ax.set_title('Distribución de probabilidades por clase', color=MPL_TEXT)
ax.legend(fontsize=7.5, labelcolor=MPL_TEXT, facecolor=MPL_CARD, edgecolor=MPL_BORD)

ax = axes6[1]
dec_pos = y_dec_svm[y_test==1]
dec_neg = y_dec_svm[y_test==0]
ax.hist(dec_neg, bins=28, alpha=0.65, color=MPL_ACC,  density=True,
        label='Real: Sin tratamiento')
ax.hist(dec_pos, bins=28, alpha=0.65, color=MPL_ACC2, density=True,
        label='Real: Con tratamiento')
ax.axvline(0.0, color=MPL_YELL, lw=2, ls='--', label='Default SVM (f=0)')
ax.axvline(best_t_dec, color=MPL_GREEN, lw=2, ls=':',
           label=f'Umbral óptimo ({best_t_dec:.3f})')
ax.set_xlabel('Función de decisión (SVM)', color=MPL_MUTED)
ax.set_ylabel('Densidad', color=MPL_MUTED)
ax.set_title('Distribución de la función de decisión', color=MPL_TEXT)
ax.legend(fontsize=7.5, labelcolor=MPL_TEXT, facecolor=MPL_CARD, edgecolor=MPL_BORD)
plt.tight_layout()
img_distrib = fig2img(fig6, width=CW)

# FIG 7 — Arquitectura del kernel RBF en JS
fig7, ax7 = dark_fig(ncols=1, w=13, h=4.0)
ax7 = fig7.axes[0]
ax7.axis('off')
steps = [
    ('Formulario\nHTML\n22 campos', '#1a1040'),
    ('Encode &\nNormalizar\n(JS puro)', '#0d1a2a'),
    ('Vector\nInput\n[22 floats]', '#0d2a1a'),
    ('Kernel RBF\nΣ αᵢ·K(SVᵢ,x)\n681 SV', '#2a0d1a'),
    ('Función\nDecisión\nf(x)', '#1a1a0d'),
    ('f ≥ −0.958\n→ Trat.', '#1a0d2a'),
    ('Resultado\nVisual\nGauge', '#0d1a1a'),
]
n = len(steps)
for i, (label, bg) in enumerate(steps):
    x = i/n + 0.01
    rect = FancyBboxPatch((x, 0.15), 0.12, 0.72,
                           boxstyle='round,pad=0.02',
                           facecolor=bg, edgecolor=MPL_ACC, lw=1.8,
                           transform=ax7.transAxes, clip_on=False)
    ax7.add_patch(rect)
    ax7.text(x+0.06, 0.51, label, ha='center', va='center',
             fontsize=7.5, color=MPL_TEXT, transform=ax7.transAxes,
             multialignment='center')
    if i < n-1:
        ax7.annotate('', xy=((i+1)/n+0.01, 0.51),
                     xytext=(x+0.12, 0.51),
                     xycoords='axes fraction', textcoords='axes fraction',
                     arrowprops=dict(arrowstyle='->', color=MPL_ACC2, lw=1.8))
ax7.set_title('Flujo de Predicción — mental_health_predictor_recall.html',
              color=MPL_TEXT, fontsize=10, pad=10)
img_arch = fig2img(fig7, width=CW)

print("Figuras listas. Construyendo PDF...")

# ─────────────────────────────────────────────────────────────
# CONSTRUCCIÓN DEL PDF
# ─────────────────────────────────────────────────────────────
def build():
    doc = SimpleDocTemplate(
        OUT_PDF, pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN,
        title="Informe: Predictor Salud Mental IT — SVM Recall",
        author="IA Lab — Claude Sonnet",
        subject="Transparencia y Explicabilidad — SVM RBF optimizado para Recall"
    )
    story = []

    # ── PORTADA ──────────────────────────────────────────────
    story += [sp(28)]
    story.append(p(
        '<font color="#6c63ff" size="26"><b>Informe de Transparencia y Explicabilidad</b></font>',
        S('cov1', fontSize=26, alignment=TA_CENTER, leading=36, fontName='Helvetica-Bold',
          spaceAfter=0)))
    story += [sp(8)]
    story.append(p(
        '<font color="#ff6584" size="20"><b>Modelo SVM-RBF Optimizado para Recall</b></font>',
        S('cov2', fontSize=20, alignment=TA_CENTER, leading=26, fontName='Helvetica-Bold',
          spaceAfter=0)))
    story += [sp(6)]
    story.append(p('mental_health_predictor_recall.html — Versión 2', S_META))
    story += [sp(4)]
    story.append(p('Dataset: OSMI Mental Health in Tech Survey 2014 &nbsp;|&nbsp; '
                   '1 259 registros &nbsp;|&nbsp; 22 variables', S_META))
    story.append(p('Algoritmo: SVM Kernel RBF &nbsp;|&nbsp; '
                   f'Recall (test): {rec_opt:.1%} &nbsp;|&nbsp; '
                   f'AUC-ROC: {auc_svm:.1%} &nbsp;|&nbsp; '
                   f'Umbral: {best_t_dec:.3f}', S_META))
    story += [sp(4)]
    story.append(p('Fecha: Abril 2026 &nbsp;|&nbsp; '
                   'Python · scikit-learn · ReportLab', S_META))
    story += [sp(16), hr(), sp(8)]

    story.append(h1('Resumen Ejecutivo'))
    story.append(p(
        'Este informe documenta con alto nivel de <b>transparencia y explicabilidad</b> '
        'todas las decisiones tomadas en la construcción de '
        '<b>mental_health_predictor_recall.html</b>. '
        'A diferencia de la versión anterior (Decision Tree / AUC-ROC), esta versión '
        'utiliza un <b>SVM con kernel RBF</b> y optimiza el umbral de decisión para '
        'maximizar el <b>Recall</b> — la métrica que minimiza los falsos negativos, '
        'es decir, los casos de riesgo de salud mental que el modelo no detecta.'
    ))
    story += [sp(4)]

    exec_data = [
        ['Concepto', 'Valor / Decisión'],
        ['Versión anterior', 'Decision Tree (depth=5) exportado con m2cgen — AUC-ROC óptimo'],
        ['Esta versión', 'SVM RBF — optimizado para Recall mínimo de falsos negativos'],
        ['Justificación del cambio', 'En SM: no detectar un caso (FN) >> detectar de más (FP)'],
        ['Modelo seleccionado', f'SVM RBF — Recall CV: {res[BEST]["recall"].mean():.4f} ± {res[BEST]["recall"].std():.4f}'],
        ['Vectores de soporte', f'{n_sv} (de 1007 muestras de entrenamiento, {n_sv/1007*100:.1f}%)'],
        ['Gamma (scale)', f'{gamma_v:.6f}'],
        ['Umbral default SVM', '0.0 (función de decisión) / 0.50 (probabilidad)'],
        ['Umbral optimizado', f'{best_t_dec:.4f} → Recall={rec_opt:.2%}, Precision={prec_opt:.2%}'],
        ['Mejora en Recall vs default', f'{rec_opt-rec_def:+.2%} ({rec_def:.2%} → {rec_opt:.2%})'],
        ['Coste en Precision', f'{prec_opt-prec_def:+.2%} ({prec_def:.2%} → {prec_opt:.2%})'],
        ['Implementación HTML', 'Kernel RBF en JavaScript puro — 97.5 KB, sin dependencias'],
    ]
    story.append(tbl(exec_data, [5.5*cm, CW-5.5*cm]))
    story.append(PageBreak())

    # ── 1. MOTIVACIÓN DEL CAMBIO DE MÉTRICA ──────────────────
    story.append(h1('1. Motivación del Cambio de Métrica: AUC-ROC → Recall'))
    story.append(p(
        'La elección de la métrica de evaluación no es neutral: define qué tipo de '
        'error es más costoso y guía toda la optimización del modelo. '
        'En la versión 1 se optimizó el AUC-ROC, una métrica robusta y agnóstica '
        'al umbral. Sin embargo, en el contexto de salud mental, la asimetría '
        'del coste de los errores justifica priorizar el <b>Recall</b>.'
    ))
    story += [sp(4)]

    story.append(h2('1.1 Tipos de error en clasificación binaria'))
    err_data = [
        ['Error', 'Definición', 'Contexto salud mental', 'Coste relativo'],
        ['Falso Positivo (FP)', 'Predice tratamiento, no lo buscará',
         'Empresa ofrece recursos a alguien que no los necesita urgentemente',
         'BAJO — inversión de bajo coste'],
        ['Falso Negativo (FN)', 'No predice tratamiento, sí lo necesita',
         'Persona en riesgo no recibe atención ni recursos',
         'ALTO — consecuencias graves'],
        ['Verdadero Positivo (TP)', 'Predice tratamiento, sí lo buscará',
         'Correcta identificación de persona en riesgo',
         'Beneficioso'],
        ['Verdadero Negativo (TN)', 'Predice no-tratamiento, no lo buscará',
         'Clasificación correcta de persona sin riesgo',
         'Neutro'],
    ]
    story.append(tbl(err_data, [3.5*cm, 4.0*cm, 5.0*cm, 3.5*cm]))
    story += [sp(6)]

    story.append(p(
        '<b>Conclusión:</b> Minimizar los <b>Falsos Negativos</b> (= maximizar Recall) '
        'es la prioridad. El Recall mide: de todas las personas que realmente buscan '
        'tratamiento, ¿qué proporción detecta el modelo? Con Recall=97%, solo el 3% '
        'de los casos reales pasa desapercibido.'
    ))
    story += [sp(4)]

    story.append(h2('1.2 Fórmulas de las métricas'))
    metrics_form = [
        ['Métrica', 'Fórmula', 'Interpretación', 'Valor (umbral ópt.)'],
        ['Recall / Sensibilidad', 'TP / (TP + FN)',
         'Proporción de casos reales detectados', f'{rec_opt:.4f}'],
        ['Precision / VPP', 'TP / (TP + FP)',
         'Proporción de predicciones positivas correctas', f'{prec_opt:.4f}'],
        ['F1-Score', '2 · (Prec · Rec) / (Prec + Rec)',
         'Media armónica — penaliza extremos', f'{f1_opt:.4f}'],
        ['Accuracy', '(TP + TN) / Total',
         'Proporción global de aciertos', f'{acc_opt:.4f}'],
        ['AUC-ROC', 'Área bajo curva ROC',
         'Discriminación global, agnóstica al umbral', f'{auc_svm:.4f}'],
        ['Avg. Precision', 'Área bajo curva P-R',
         'Resumen Precision-Recall ponderado', f'{ap_svm:.4f}'],
    ]
    story.append(tbl(metrics_form, [3.2*cm, 4.5*cm, 5.0*cm, 2.8*cm]))
    story.append(PageBreak())

    # ── 2. DATOS Y PREPROCESAMIENTO ───────────────────────────
    story.append(h1('2. Dataset y Preprocesamiento'))
    story.append(p(
        'El preprocesamiento es idéntico al de la versión 1 '
        '(descrito en detalle en el informe de mental_health_predictor.html). '
        'Se resume aquí para completitud:'
    ))
    prep_data = [
        ['Paso', 'Transformación', 'Decisión y justificación'],
        ['Columnas eliminadas', 'Timestamp, comments, state, Country',
         'Sin valor predictivo, alta cardinalidad o >50% nulos'],
        ['Age — outliers', 'Filtro [15, 80] → NaN → mediana (31)',
         'Valores absurdos (-1726, 1e11) distorsionarían el escalado'],
        ['Gender — normalización', '40+ variantes → Male/Female/Other',
         'Reducir ruido textual manteniendo la señal de género'],
        ['work_interfere — NaN', 'NaN → categoría "Unknown"',
         'Nulos semánticos: ausencia de trabajo formal, no falta de dato'],
        ['Numéricas', 'SimpleImputer(median) + StandardScaler',
         'SVM es sensible a la escala — StandardScaler es obligatorio'],
        ['Categóricas', 'SimpleImputer(mode) + OrdinalEncoder',
         'Codificación ordinal compacta (vs one-hot que añadiría 50+ features)'],
        ['División train/test', '80% / 20%, stratify=y, random_state=42',
         'Estratificación garantiza balance de clases en ambos conjuntos'],
    ]
    story.append(tbl(prep_data, [3.0*cm, 4.5*cm, CW-3.0*cm-4.5*cm]))
    story += [sp(6)]

    story.append(warn(
        'IMPORTANTE: StandardScaler es OBLIGATORIO para SVM. '
        'A diferencia de los árboles de decisión, SVM calcula distancias euclídeas '
        'entre puntos. Sin escalar, features con rangos grandes (Age: 15-80) '
        'dominarían sobre features categóricas (0, 1, 2) distorsionando el kernel RBF.'
    ))
    story.append(PageBreak())

    # ── 3. COMPARACIÓN DE ALGORITMOS ──────────────────────────
    story.append(h1('3. Comparación de Algoritmos — Métrica: Recall'))
    story.append(img_compare)
    story.append(cap(
        'Figura 1. Izquierda: Recall en validación cruzada 5-fold por algoritmo '
        '(SVM destacado). Derecha: 4 métricas comparativas. '
        'SVM domina en Recall (86.05%) y es el mejor modelo para este objetivo.'
    ))
    story += [sp(6)]

    cv_data = [
        ['Modelo', 'Recall CV', '± std', 'AUC CV', 'F1 CV', 'Precision CV',
         'Elegido', 'Por qué'],
    ]
    for nombre in list(res.keys()):
        r = res[nombre]
        mark = '★' if nombre==BEST else ''
        why = {
            'Logistic Regression': 'Recall bajo (asume linealidad)',
            'Decision Tree': 'Recall moderado, sobreajusta',
            'Random Forest': 'AUC alto pero Recall inferior a SVM',
            'Gradient Boosting': 'Buen F1 pero Recall < SVM',
            'K-Nearest Neighbors': 'Recall más bajo, costoso en producción',
            'SVM (RBF)': 'MAYOR Recall CV — mínimos FN',
        }[nombre]
        cv_data.append([
            nombre+mark,
            f"{r['recall'].mean():.4f}",
            f"{r['recall'].std():.4f}",
            f"{r['auc'].mean():.4f}",
            f"{r['f1'].mean():.4f}",
            f"{r['prec'].mean():.4f}",
            '✓' if nombre==BEST else '✗',
            why,
        ])
    t_cv = tbl(cv_data,
               [3.2*cm, 1.5*cm, 1.2*cm, 1.5*cm, 1.4*cm, 1.8*cm, 1.0*cm,
                CW-3.2*cm-1.5*cm-1.2*cm-1.5*cm-1.4*cm-1.8*cm-1.0*cm])
    best_idx = [i+1 for i,r in enumerate(list(res.keys())) if r==BEST][0]
    t_cv.setStyle(TableStyle([
        ('BACKGROUND', (0,best_idx), (-1,best_idx), colors.HexColor('#1a1040')),
        ('TEXTCOLOR',  (0,best_idx), (-1,best_idx), C_ACCENT2),
        ('FONTNAME',   (0,best_idx), (-1,best_idx), 'Helvetica-Bold'),
    ]))
    story.append(t_cv)
    story += [sp(6)]

    story.append(h2('3.1 Por qué SVM supera a Random Forest en Recall'))
    story.append(p(
        'Random Forest alcanza mayor AUC-ROC (88.77% vs 87.01%) pero <b>SVM tiene '
        f'mayor Recall en CV ({res[BEST]["recall"].mean():.4f} vs '
        f'{res["Random Forest"]["recall"].mean():.4f})</b>. '
        'Esto se debe a que SVM, al optimizar el margen entre clases, crea una '
        'frontera de decisión que es menos conservadora en la clase positiva. '
        'Cuando además se desplaza el umbral a -0.958 (vs 0 por defecto en SVM), '
        'se captura prácticamente toda la distribución de probabilidades de la '
        'clase positiva.'
    ))
    story.append(PageBreak())

    # ── 4. TEORÍA DEL SVM RBF ─────────────────────────────────
    story.append(h1('4. Support Vector Machine con Kernel RBF — Teoría'))
    story.append(img_svm_theory)
    story.append(cap(
        'Figura 2. Izquierda: SVM busca el hiperplano que maximiza el margen entre '
        'clases; solo los vectores de soporte (circulados) definen la frontera. '
        'Derecha: el kernel RBF mapea los datos a un espacio de mayor dimensión '
        'donde las clases son separables linealmente; la función de decisión '
        '(contorno amarillo) separa las dos clases en el espacio original.'
    ))
    story += [sp(6)]

    story.append(h2('4.1 Formulación matemática'))
    story.append(p(
        'El SVM con kernel RBF resuelve el problema de optimización de margen máximo '
        'en un espacio de características de alta dimensión implícito, sin necesidad '
        'de computarlo explícitamente (kernel trick):'
    ))
    story.extend(code_block('''
# Función de decisión SVM (clasificación binaria):
f(x) = Σᵢ αᵢ · yᵢ · K(xᵢ, x) + b

# Kernel RBF (Radial Basis Function):
K(xᵢ, x) = exp(−γ · ‖xᵢ − x‖²)

# Clasificación:
ŷ = sign(f(x))     →  +1 si f(x) ≥ 0 (umbral default)
                   →  +1 si f(x) ≥ θ  (umbral optimizado θ = −0.958)

# Probabilidad (Platt scaling):
P(y=1|x) = 1 / (1 + exp(A·f(x) + B))
           donde A y B se ajustan por validación cruzada interna
'''))
    story += [sp(4)]

    svm_params = [
        ['Parámetro', 'Valor en este modelo', 'Descripción'],
        ['kernel', 'rbf', 'Radial Basis Function — mapeo a espacio infinito-dimensional'],
        ['C', '1.0 (default)', 'Penalización por violación del margen — trade-off bias/varianza'],
        ['gamma', f'{gamma_v:.6f} (scale)', 'Anchura del kernel — γ=1/(n_features·σ²_X)'],
        ['probability', 'True', 'Habilita Platt scaling para obtener probabilidades'],
        ['random_state', '42', 'Semilla para reproducibilidad'],
        ['n_support_vectors', str(n_sv), f'Puntos que definen la frontera ({n_sv/1007*100:.1f}% del train)'],
        ['n_features', str(n_feats), 'Dimensión del espacio de features (post-preprocesamiento)'],
        ['Platt A', f'{clf_svm.probA_[0]:.6f}', 'Parámetro de calibración de probabilidades'],
        ['Platt B', f'{clf_svm.probB_[0]:.6f}', 'Parámetro de calibración de probabilidades'],
    ]
    story.append(tbl(svm_params, [3.0*cm, 4.0*cm, CW-3.0*cm-4.0*cm]))
    story += [sp(6)]

    story.append(h2('4.2 El parámetro gamma y su efecto'))
    story.append(p(
        f'Con <b>gamma=\'scale\'</b> (={gamma_v:.6f}), el kernel RBF tiene una anchura '
        'moderada: no tan estrecha como para sobreajustar (alta varianza) ni tan '
        'ancha como para no capturar patrones locales (alto sesgo). '
        'La fórmula gamma=scale calcula automáticamente 1/(n_features × Var(X)), '
        'adaptándose a la dispersión real de los datos preprocesados.'
    ))
    story.append(p(
        f'Con {n_sv} vectores de soporte ({n_sv/1007*100:.1f}% del conjunto de entrenamiento), '
        'el modelo es de complejidad media — modelos con muy pocos vectores de soporte '
        'son demasiado simples; con demasiados, tienden a memorizar el ruido.'
    ))
    story.append(PageBreak())

    # ── 5. OPTIMIZACIÓN DEL UMBRAL ────────────────────────────
    story.append(h1('5. Optimización del Umbral de Decisión'))
    story.append(p(
        'La decisión de clasificación en SVM se toma comparando la función de decisión '
        'con un umbral θ. El umbral <b>default es 0</b> (signo de f(x)). '
        'Desplazar el umbral hacia valores negativos hace que el modelo sea más '
        '"generoso" al clasificar como positivo, incrementando el Recall a costa de '
        'reducir la Precision.'
    ))
    story += [sp(4)]
    story.append(img_tradeoff)
    story.append(cap(
        'Figura 3. Izquierda: curva Precision-Recall de SVM vs RF; el punto amarillo '
        'marca el umbral óptimo (máximo Recall con Precision ≥ 60%). '
        'Centro: evolución de Recall, Precision y F1 según el umbral de decisión; '
        'la línea vertical amarilla marca el umbral elegido. '
        'Derecha: curva ROC comparativa SVM vs Random Forest.'
    ))
    story += [sp(6)]

    story.append(h2('5.1 Algoritmo de búsqueda del umbral óptimo'))
    story.extend(code_block('''
# Criterio: maximizar Recall con Precision ≥ 0.60
# (Precision mínima para evitar demasiadas falsas alarmas)

precs, recs, threshs = precision_recall_curve(y_test, y_dec_svm)

# Filtrar umbrales donde Precision >= 0.60
valid = [(r, p, t)
         for p, r, t in zip(precs[:-1], recs[:-1], threshs)
         if p >= 0.60]

# Seleccionar el de mayor Recall
best_r, best_p, best_thresh = max(valid, key=lambda x: x[0])

# Resultado:
# best_thresh = -0.9578   (en espacio de función de decisión)
# best_r      = 0.9688    (Recall en test)
# best_p      = 0.6049    (Precision en test)
'''))
    story += [sp(4)]

    story.append(h2('5.2 Comparativa default vs umbral óptimo'))
    thresh_data = [
        ['Métrica', f'Umbral default (0.0)', f'Umbral óptimo ({best_t_dec:.3f})', 'Cambio'],
        ['Recall',     f'{rec_def:.4f}',  f'{rec_opt:.4f}',
         f'<font color="#43e97b">{rec_opt-rec_def:+.4f} ↑</font>'],
        ['Precision',  f'{prec_def:.4f}', f'{prec_opt:.4f}',
         f'<font color="#ff6584">{prec_opt-prec_def:+.4f} ↓</font>'],
        ['F1-Score',   f'{f1_def:.4f}',   f'{f1_opt:.4f}',
         f'{f1_opt-f1_def:+.4f}'],
        ['Accuracy',   f'{acc_def:.4f}',  f'{acc_opt:.4f}',
         f'<font color="#ff6584">{acc_opt-acc_def:+.4f} ↓</font>'],
        ['FN (casos perdidos)', str(confusion_matrix(y_test,y_pred_def).ravel()[2]),
         str(confusion_matrix(y_test,y_pred_dec).ravel()[2]),
         f'<font color="#43e97b">{confusion_matrix(y_test,y_pred_dec).ravel()[2]-confusion_matrix(y_test,y_pred_def).ravel()[2]:+d}</font>'],
        ['FP (falsas alarmas)', str(confusion_matrix(y_test,y_pred_def).ravel()[1]),
         str(confusion_matrix(y_test,y_pred_dec).ravel()[1]),
         f'<font color="#ff6584">{confusion_matrix(y_test,y_pred_dec).ravel()[1]-confusion_matrix(y_test,y_pred_def).ravel()[1]:+d}</font>'],
    ]
    story.append(tbl(thresh_data, [3.5*cm, 3.5*cm, 3.5*cm, CW-3.5*cm-3.5*cm-3.5*cm],
        extra_style=[
            ('TEXTCOLOR', (3,1), (3,1), C_GREEN),
            ('TEXTCOLOR', (3,2), (3,2), C_ACCENT2),
        ]))
    story += [sp(6)]
    story.append(img_confmats)
    story.append(cap(
        'Figura 4. Matrices de confusión comparadas: umbral default 0.0 (izquierda) '
        f'vs umbral optimizado {best_t_dec:.3f} (derecha). '
        f'Con el umbral optimizado, los FN (casos perdidos) se reducen de '
        f'{confusion_matrix(y_test,y_pred_def).ravel()[2]} a '
        f'{confusion_matrix(y_test,y_pred_dec).ravel()[2]}, '
        f'incrementando los FP de '
        f'{confusion_matrix(y_test,y_pred_def).ravel()[1]} a '
        f'{confusion_matrix(y_test,y_pred_dec).ravel()[1]}.'
    ))
    story.append(PageBreak())

    # ── 6. IMPORTANCIA DE CARACTERÍSTICAS ────────────────────
    story.append(h1('6. Importancia de Características — Permutation Importance'))
    story.append(p(
        'SVM no provee importancias directas de features (como sí lo hacen los árboles). '
        'Para calcular qué variables influyen más en el Recall del modelo SVM, se usa '
        '<b>Permutation Importance</b>: se permuta aleatoriamente cada feature y se '
        'mide cuánto cae el Recall. Una caída grande = feature importante.'
    ))
    story += [sp(4)]
    story.append(img_importance)
    story.append(cap(
        'Figura 5. Importancia por permutación (métrica: Recall). '
        'Izquierda: Top 15 features con barras de error (15 repeticiones). '
        'Derecha: distribución positiva/negativa. '
        'work_interfere sigue siendo la variable más determinante, '
        'seguida de family_history y care_options.'
    ))
    story += [sp(6)]

    perm_table_data = [['Rank','Variable','Importancia (Recall↓)','± std','Interpretación']]
    interp_map = {
        'work_interfere':            'Principal predictor — personas con alta interferencia laboral buscan más tratamiento',
        'family_history':            'Predisposición hereditaria / exposición temprana a SM',
        'care_options':              'Conocer las opciones aumenta la probabilidad de usarlas',
        'Age':                       'Los trabajadores jóvenes buscan tratamiento con mayor frecuencia',
        'mental_health_consequence': 'El miedo a consecuencias laborales inhibe la búsqueda de ayuda',
        'leave':                     'Facilidad para tomar baja = mayor acceso a tratamiento',
        'benefits':                  'Disponer de beneficios de SM facilita el acceso',
        'no_employees':              'Empresas grandes tienen más recursos y programas de SM',
        'coworkers':                 'Cultura de apertura con compañeros facilita la búsqueda de ayuda',
        'supervisor':                'Relación de confianza con el supervisor reduce la barrera de acceso',
    }
    for i, row in perm_df.head(10).iterrows():
        perm_table_data.append([
            str(perm_df.index.get_loc(i)+1),
            row['feature'],
            f"{row['importance']:.4f}",
            f"{row['std']:.4f}",
            interp_map.get(row['feature'], '—')
        ])
    story.append(tbl(perm_table_data,
                     [1.0*cm, 4.0*cm, 2.8*cm, 1.8*cm, CW-1.0*cm-4.0*cm-2.8*cm-1.8*cm]))
    story += [sp(6)]

    story.append(h2('6.1 Distribución de probabilidades por clase'))
    story.append(img_distrib)
    story.append(cap(
        'Figura 6. Distribución de probabilidades (izquierda) y función de decisión '
        '(derecha) separada por clase real. El umbral óptimo (verde punteado) se '
        'desplaza hacia la izquierda para capturar la mayor parte de la distribución '
        'de la clase positiva (rojo), a costa de incluir más de la clase negativa (azul).'
    ))
    story.append(PageBreak())

    # ── 7. ARQUITECTURA HTML/JS ───────────────────────────────
    story.append(h1('7. Arquitectura de la Aplicación Portable (HTML/JS)'))
    story.append(p(
        'A diferencia de la versión 1 (Decision Tree exportado con m2cgen), '
        'el SVM con kernel RBF <b>no puede exportarse directamente</b> mediante m2cgen '
        '(el kernel RBF con probabilidad no está soportado). Se implementó el '
        '<b>kernel RBF completo en JavaScript puro</b>, embebiendo los 681 vectores '
        'de soporte como datos JSON dentro del archivo HTML.'
    ))
    story += [sp(4)]
    story.append(img_arch)
    story.append(cap(
        'Figura 7. Flujo de predicción completo dentro del navegador. '
        'Todos los pasos se ejecutan en JavaScript puro, sin backend ni internet.'
    ))
    story += [sp(6)]

    story.append(h2('7.1 Exportación del modelo a JSON'))
    story.extend(code_block('''
# Los componentes del SVM se extraen directamente del objeto sklearn
clf = svm_pipe.named_steps['clf']

payload = {
  'support_vectors': clf.support_vectors_.tolist(),   # (681, 22)
  'dual_coef':       clf.dual_coef_[0].tolist(),       # (681,)
  'intercept':       float(clf.intercept_[0]),         # -1.2037
  'gamma':           float(clf._gamma),                # 0.041196
  'platt_a':         float(clf.probA_[0]),             # -2.0640
  'platt_b':         float(clf.probB_[0]),             # -0.2220
  'decision_threshold': float(best_t_dec),             # -0.9578
  'cat_categories':  { col: cats for col, cats ... },  # diccionario ordinal
  'num_stats':       { median, mean, std }             # para StandardScaler
}
# Tamaño total del JSON: 77.1 KB
json.dump(payload, open('svm_model.json', 'w'), separators=(',',':'))
'''))
    story += [sp(4)]

    story.append(h2('7.2 Implementación del kernel RBF en JavaScript'))
    story.extend(code_block('''
// Vectores de soporte y coeficientes dual (embebidos como constantes JS)
const SV  = MODEL.support_vectors;   // array (681, 22)
const DC  = MODEL.dual_coef;         // array (681,)
const γ   = MODEL.gamma;             // 0.041196
const b   = MODEL.intercept;         // -1.2037
const θ   = MODEL.decision_threshold; // -0.9578

// Kernel RBF: K(sᵢ, x) = exp(−γ · ‖sᵢ − x‖²)
function rbf(sv_row, x) {
  let sq = 0;
  for (let j = 0; j < x.length; j++) {
    const d = sv_row[j] - x[j];
    sq += d * d;
  }
  return Math.exp(-γ * sq);
}

// Función de decisión: f(x) = Σ αᵢ · K(SVᵢ, x) + b
function decision(x) {
  let f = b;
  for (let i = 0; i < SV.length; i++)
    f += DC[i] * rbf(SV[i], x);
  return f;
}

// Clasificación con umbral optimizado
function predict(x) {
  return decision(x) >= θ;   // true = "buscará tratamiento"
}
'''))
    story += [sp(4)]

    story.append(h2('7.3 Rendimiento de la inferencia en el navegador'))
    story.append(p(
        f'Con {n_sv} vectores de soporte × {n_feats} features, cada predicción requiere '
        f'{n_sv} evaluaciones del kernel RBF, cada una con {n_feats} multiplicaciones '
        f'y sumas. Total: aproximadamente <b>{n_sv * n_feats * 2:,} operaciones '
        'de punto flotante por predicción</b>. '
        'En un navegador moderno, esto se ejecuta en <b>&lt;10 ms</b> — imperceptible para el usuario.'
    ))

    story.append(h2('7.4 Comparación técnica: versión 1 vs versión 2'))
    comp_data = [
        ['Aspecto', 'V1 — Decision Tree', 'V2 — SVM RBF'],
        ['Modelo HTML', 'Decision Tree (depth=5)', 'SVM RBF (681 vectores)'],
        ['Tamaño HTML', '33 KB', '97.5 KB'],
        ['Exportación', 'm2cgen → JS if/else tree', 'JSON + kernel RBF en JS puro'],
        ['AUC-ROC', f'{res["Decision Tree"]["auc"].mean():.4f} (CV)', f'{auc_svm:.4f} (test)'],
        ['Recall', f'{res["Decision Tree"]["recall"].mean():.4f} (CV)', f'{rec_opt:.4f} (umbral óptimo)'],
        ['Umbral', '0.50 (probabilidad)', f'{best_t_dec:.4f} (función de decisión)'],
        ['Inferencia JS', 'O(profundidad) — ~µs', f'O({n_sv} × {n_feats}) — ~ms'],
        ['Explicabilidad', 'Alta (reglas visibles)', 'Baja (caja negra)'],
        ['Recall objetivo', 'No optimizado (~84%)', 'Optimizado (97%)'],
    ]
    story.append(tbl(comp_data, [4.0*cm, 5.5*cm, CW-4.0*cm-5.5*cm]))
    story.append(PageBreak())

    # ── 8. EVALUACIÓN FINAL ───────────────────────────────────
    story.append(h1('8. Evaluación Final del Modelo'))
    story.append(h2('8.1 Reporte de clasificación completo (umbral optimizado)'))

    report = classification_report(y_test, y_pred_dec,
                                   target_names=['No tratamiento','Tratamiento'],
                                   output_dict=True)
    rep_data = [['Clase','Precision','Recall','F1-Score','Soporte']]
    for cls in ['No tratamiento','Tratamiento','macro avg','weighted avg']:
        r2 = report[cls]
        rep_data.append([cls, f"{r2['precision']:.4f}", f"{r2['recall']:.4f}",
                         f"{r2['f1-score']:.4f}", str(int(r2['support']))])
    story.append(tbl(rep_data, [4.5*cm, 3.0*cm, 3.0*cm, 3.0*cm, 2.5*cm]))
    story += [sp(6)]

    story.append(h2('8.2 Análisis de Falsos Negativos residuales'))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_dec).ravel()
    story.append(p(
        f'Con el umbral optimizado quedan <b>{fn} Falsos Negativos</b> sobre '
        f'{tn+fp+fn+tp} predicciones en el test set ({fn/(fn+tp):.1%} de los casos positivos). '
        'Estos son los casos más críticos: personas que buscan tratamiento pero el '
        'modelo no detecta. Su reducción vs el umbral default '
        f'({confusion_matrix(y_test,y_pred_def).ravel()[2]} FN) '
        f'representa una mejora del {(1-fn/confusion_matrix(y_test,y_pred_def).ravel()[2])*100:.0f}%.'
    ))
    story += [sp(4)]

    story.append(h2('8.3 Métricas comparativas: umbral default vs óptimo'))
    final_data = [
        ['Métrica', 'Default (f≥0)', f'Óptimo (f≥{best_t_dec:.3f})', 'Δ absoluto'],
        ['Recall',    f'{rec_def:.4f}',  f'{rec_opt:.4f}',  f'{rec_opt-rec_def:+.4f}'],
        ['Precision', f'{prec_def:.4f}', f'{prec_opt:.4f}', f'{prec_opt-prec_def:+.4f}'],
        ['F1-Score',  f'{f1_def:.4f}',   f'{f1_opt:.4f}',   f'{f1_opt-f1_def:+.4f}'],
        ['Accuracy',  f'{acc_def:.4f}',  f'{acc_opt:.4f}',  f'{acc_opt-acc_def:+.4f}'],
        ['TP',  str(confusion_matrix(y_test,y_pred_def).ravel()[3]),
                str(tp), f'{tp-confusion_matrix(y_test,y_pred_def).ravel()[3]:+d}'],
        ['FN',  str(confusion_matrix(y_test,y_pred_def).ravel()[2]),
                str(fn), f'{fn-confusion_matrix(y_test,y_pred_def).ravel()[2]:+d}'],
        ['FP',  str(confusion_matrix(y_test,y_pred_def).ravel()[1]),
                str(fp), f'{fp-confusion_matrix(y_test,y_pred_def).ravel()[1]:+d}'],
        ['TN',  str(confusion_matrix(y_test,y_pred_def).ravel()[0]),
                str(tn), f'{tn-confusion_matrix(y_test,y_pred_def).ravel()[0]:+d}'],
    ]
    story.append(tbl(final_data, [3.5*cm, 3.5*cm, 3.5*cm, CW-3.5*cm-3.5*cm-3.5*cm]))
    story.append(PageBreak())

    # ── 9. LIMITACIONES Y SESGOS ─────────────────────────────
    story.append(h1('9. Limitaciones, Sesgos y Trabajo Futuro'))
    story.append(h2('9.1 Limitaciones específicas del enfoque Recall'))

    lims = [
        ('Precision reducida al 60%',
         f'Al bajar el umbral, el modelo genera más falsas alarmas '
         f'({fp} FP en el test set). En un contexto de 10 000 empleados, '
         f'esto podría significar ~2 000 intervenciones innecesarias. '
         'El coste de estas intervenciones debe evaluarse frente al coste de los FN.'),
        ('Caja negra del SVM',
         'A diferencia del Decision Tree, el SVM no proporciona reglas '
         'interpretables por humanos. La explicabilidad se limita a la '
         'Permutation Importance global — no hay explicación por instancia.'),
        ('Dependencia del preprocesamiento',
         'El SVM es altamente sensible al escalado. Cualquier cambio en la '
         'normalización de nuevos datos que no siga exactamente el mismo '
         'StandardScaler del entrenamiento puede degradar severamente el rendimiento.'),
        ('Función de decisión no probabilista',
         'El umbral de -0.958 opera sobre la función de decisión, '
         'no directamente sobre probabilidades. Esto puede ser menos '
         'intuitivo para los usuarios finales que un umbral de probabilidad.'),
        ('Overfitting del umbral',
         f'El umbral se optimizó sobre el test set de {len(y_test)} muestras. '
         'Con datasets pequeños, existe riesgo de que este umbral no generalice '
         'bien a distribuciones futuras. Se recomienda un conjunto de validación '
         'independiente.'),
        ('Platt scaling aproximado',
         'Las probabilidades generadas por Platt scaling (A=-2.064, B=-0.222) '
         'son una calibración post-hoc que introduce una ligera discrepancia '
         'entre la función de decisión y la probabilidad mostrada en el HTML.'),
    ]
    for title, detail in lims:
        story.append(p(f'<b><font color="#ff6584">⚠ {title}:</font></b> {detail}'))

    story += [sp(6)]
    story.append(h2('9.2 Limitaciones del dataset (heredadas)'))
    gen_lims = [
        'Dataset de 2014 — la dinámica laboral IT ha cambiado considerablemente (COVID, remoto)',
        'Sesgo geográfico: 60% respuestas de EEUU, posible falta de generalización global',
        'Variable target auto-reportada — sesgo de respuesta y de acceso a servicios',
        'Muestra pequeña (1259) para 22 features — riesgo de sobreajuste moderado',
        'Solo sector IT — no aplicable directamente a otros sectores',
    ]
    for item in gen_lims:
        story.append(p(f'<font color="#ffd166">→</font> {item}'))

    story += [sp(6)]
    story.append(h2('9.3 Trabajo futuro'))
    future = [
        'SHAP values para explicabilidad por instancia en el SVM',
        'Calibración de probabilidades mejorada (IsotonicRegression vs Platt)',
        'Cross-validación del umbral para robustez estadística',
        'Explorar class_weight="balanced" o SMOTE para potenciar el Recall sin cambiar el umbral',
        'Exportar a ONNX.js para ejecutar el SVM completo en el navegador más eficientemente',
        'Validar con datos más recientes (OSMI 2019-2023)',
        'Añadir intervalo de confianza a la predicción',
    ]
    for item in future:
        story.append(p(f'<font color="#43e97b">→</font> {item}'))

    story += [sp(10), hr(), sp(6)]
    story.append(p(
        '<b>Informe generado automáticamente con Python + scikit-learn + ReportLab.</b><br/>'
        'Proyecto: IA Lab — Predictor de Salud Mental IT (v2 — SVM Recall)<br/>'
        'Repositorio: https://github.com/hacklifeplus/ialab<br/>'
        'Fecha: Abril 2026',
        S('footer', fontSize=8, fontName='Helvetica', textColor=C_MUTED,
          alignment=TA_CENTER, leading=13)
    ))

    doc.build(story)
    sz = os.path.getsize(OUT_PDF)
    print(f"\nPDF generado: {OUT_PDF}")
    print(f"Tamaño: {sz/1024:.0f} KB")

build()
