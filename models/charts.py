import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import os

# ── DARK THEME PALETTE
BG      = '#080C10'
SURFACE = '#0D1117'
SURFACE2= '#161B22'
BORDER  = '#21262D'
CYAN    = '#00E5FF'
GREEN   = '#39D353'
RED     = '#EF4444'
ORANGE  = '#F97316'
MUTED   = '#7D8590'
TEXT    = '#E6EDF3'
PURPLE  = '#7C3AED'
BAR_DEF = '#1C2E3D'

plt.rcParams.update({
    'font.family':      'monospace',
    'axes.facecolor':   SURFACE,
    'figure.facecolor': BG,
    'axes.edgecolor':   BORDER,
    'axes.labelcolor':  MUTED,
    'xtick.color':      MUTED,
    'ytick.color':      MUTED,
    'text.color':       TEXT,
    'grid.color':       BORDER,
    'grid.linestyle':   '--',
    'grid.alpha':       0.4,
})

def save_dir(static='static'):
    d = os.path.join(static, 'plots')
    os.makedirs(d, exist_ok=True)
    return d

def short_names(models):
    mapping = {
        'Mean Baseline':      'Baseline',
        'Linear Regression': 'Linear',
        'Random Forest':     'Rnd Forest',
        'Gradient Boosting': 'Grad Boost',
        'Extra Trees':       'Ext Trees',
        'SVR':               'SVR',
    }
    return [mapping.get(m, m[:10]) for m in models]

# ── METRICS BAR CHART
def plot_metrics(results, static='static'):
    models = list(results.keys())
    maes   = [results[m]['mae']   for m in models]
    rmses  = [results[m]['rmse']  for m in models]
    r2s    = [results[m]['cv_r2'] for m in models]
    short  = short_names(models)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor(BG)
    fig.subplots_adjust(wspace=0.35)

    for ax, vals, title, lower_better in zip(
        axes,
        [maes, rmses, r2s],
        ['MAE  (lower is better)', 'RMSE  (lower is better)', 'CV R²  (higher is better)'],
        [True, True, False]
    ):
        best_idx = vals.index(min(vals)) if lower_better else vals.index(max(vals))
        colors = [CYAN if i == best_idx else BAR_DEF for i in range(len(vals))]
        bars = ax.bar(short, vals, color=colors, edgecolor=BORDER, width=0.55, linewidth=0.5)
        ax.set_title(title, fontsize=10, fontweight='bold', color=MUTED, pad=10)
        ax.set_facecolor(SURFACE)
        for spine in ax.spines.values():
            spine.set_color(BORDER)
        ax.tick_params(axis='x', labelsize=8, rotation=20, colors=MUTED)
        ax.tick_params(axis='y', labelsize=8, colors=MUTED)
        ax.yaxis.grid(True, color=BORDER, alpha=0.5)
        ax.set_axisbelow(True)
        for bar, val in zip(bars, vals):
            color = CYAN if bar.get_facecolor()[:3] == matplotlib.colors.to_rgb(CYAN) else TEXT
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + abs(max(vals)-min(vals))*0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8, color=color)

    fig.suptitle('Model Comparison — Baseline + All 5 Models', fontsize=13, fontweight='bold',
                 color=TEXT, y=1.02)
    path = os.path.join(save_dir(static), 'metrics.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()

# ── COEFFICIENTS
def plot_coefficients(results, static='static'):
    reg_models = {m: results[m] for m in results if 'coef' in results[m]}
    n = len(reg_models)
    if n == 0: return
    fig, axes = plt.subplots(1, n, figsize=(6*n, 5), sharey=False)
    if n == 1: axes = [axes]
    fig.patch.set_facecolor(BG)

    for ax, (mname, mdata) in zip(axes, reg_models.items()):
        coef = mdata['coef']
        features = list(coef.keys())
        values   = list(coef.values())
        sorted_pairs = sorted(zip(values, features))
        vals, feats = zip(*sorted_pairs)
        colors = [RED if v < 0 else CYAN for v in vals]
        bars = ax.barh(feats, vals, color=colors, edgecolor='none', height=0.6, alpha=0.85)
        ax.axvline(0, color=MUTED, linewidth=1, linestyle='--', alpha=0.5)
        ax.set_title(mname, fontsize=11, fontweight='bold', color=TEXT, pad=10)
        ax.set_facecolor(SURFACE)
        for spine in ax.spines.values(): spine.set_color(BORDER)
        ax.tick_params(labelsize=9, colors=MUTED)
        for bar, val in zip(bars, vals):
            offset = abs(max(vals)-min(vals))*0.02
            ax.text(val + (offset if val >= 0 else -offset),
                    bar.get_y() + bar.get_height()/2,
                    f'{val:+.3f}', va='center',
                    ha='left' if val >= 0 else 'right', fontsize=8, color=TEXT)

    fig.suptitle('Feature Coefficients — Regression Models', fontsize=13,
                 fontweight='bold', color=TEXT, y=1.01)
    plt.tight_layout()
    path = os.path.join(save_dir(static), 'coefficients.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()

# ── FEATURE IMPORTANCE
def plot_importance(results, static='static'):
    imp_model = None
    for m in ['Gradient Boosting', 'Random Forest', 'Extra Trees']:
        if m in results and 'importance' in results[m]:
            imp_model = m
            break
    if not imp_model: return

    imp  = results[imp_model]['importance']
    feat = list(imp.keys())
    vals = list(imp.values())
    sorted_pairs = sorted(zip(vals, feat))
    vals, feat = zip(*sorted_pairs)

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(BG)
    colors = [CYAN if v == max(vals) else BAR_DEF for v in vals]
    # Add glow effect for top bar
    for i, (v, f, c) in enumerate(zip(vals, feat, colors)):
        ax.barh(f, v, color=c, edgecolor='none', height=0.6, alpha=0.9)
        if c == CYAN:
            ax.barh(f, v, color=CYAN, edgecolor='none', height=0.6, alpha=0.15)

    ax.set_title(f'Feature Importances — {imp_model}', fontsize=12,
                 fontweight='bold', color=TEXT, pad=12)
    ax.set_xlabel('Importance Score', fontsize=9, color=MUTED)
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values(): spine.set_color(BORDER)
    ax.tick_params(labelsize=9, colors=MUTED)
    ax.xaxis.grid(True, color=BORDER, alpha=0.5)
    ax.set_axisbelow(True)
    for i, (v, f) in enumerate(zip(vals, feat)):
        ax.text(v + 0.001, i, f'{v:.3f}', va='center', ha='left', fontsize=8.5, color=TEXT)
    plt.tight_layout()
    path = os.path.join(save_dir(static), 'importance.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()

# ── HEATMAP
def plot_heatmap(df, static='static'):
    cols = ['study_hours','attendance_pct','social_media','sleep_hours',
            'exercise_num','society_count','diet_enc','age','cgpa']
    lbls = ['Study Hrs','Attendance','Social Media','Sleep Hrs',
            'Exercise','Societies','Diet','Age','CGPA']

    sub = df[cols].copy()
    sub.columns = lbls
    corr = sub.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(BG)

    import matplotlib.colors as mcolors
    dark_cmap = mcolors.LinearSegmentedColormap.from_list('dark_div', [
        '#00897B', '#0D4A42', '#161B22', '#4A1515', '#B71C1C'
    ])
    sns.heatmap(
        corr, ax=ax, annot=True, fmt='.2f', annot_kws={'size': 9, 'color': TEXT},
        cmap=dark_cmap, center=0, vmin=-1, vmax=1,
        linewidths=1.5, linecolor=BG, square=True,
        cbar_kws={'shrink': 0.8}
    )
    ax.set_title('Feature Correlation Heatmap', fontsize=13,
                 fontweight='bold', color=TEXT, pad=14)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_facecolor(SURFACE)

    # Style colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(colors=MUTED, labelsize=8)
    cbar.outline.set_edgecolor(BORDER)

    plt.tight_layout()
    path = os.path.join(save_dir(static), 'heatmap.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()

# ── CV BARS
def plot_cv_bars(results, static='static'):
    models = list(results.keys())
    means  = [results[m]['cv_r2']  for m in models]
    stds   = [results[m]['cv_std'] for m in models]
    short  = short_names(models)
    best_i = means.index(max(means))
    colors = [CYAN if i == best_i else BAR_DEF for i in range(len(models))]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(BG)
    bars = ax.bar(short, means, yerr=stds, color=colors, edgecolor=BORDER,
                  width=0.5, capsize=5, linewidth=0.5,
                  error_kw={'color': MUTED, 'linewidth': 1.2, 'capthick': 1.2})
    ax.axhline(0, color=MUTED, linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_title('Cross-Validation R² by Model (5-fold)', fontsize=12,
                 fontweight='bold', color=TEXT, pad=12)
    ax.set_ylabel('CV R²', fontsize=10, color=MUTED)
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values(): spine.set_color(BORDER)
    ax.tick_params(axis='x', labelsize=9, colors=MUTED, rotation=10)
    ax.tick_params(axis='y', labelsize=9, colors=MUTED)
    ax.yaxis.grid(True, color=BORDER, alpha=0.5)
    ax.set_axisbelow(True)
    for bar, val, std in zip(bars, means, stds):
        ypos = bar.get_height() + std + abs(max(means)-min(means))*0.03
        color = CYAN if bar.get_facecolor()[:3] == matplotlib.colors.to_rgb(CYAN) else TEXT
        ax.text(bar.get_x() + bar.get_width()/2, ypos,
                f'{val:.3f}', ha='center', fontsize=9, color=color)
    plt.tight_layout()
    path = os.path.join(save_dir(static), 'cv_bars.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()

def generate_all_charts(results, df, static='static'):
    plot_metrics(results, static)
    plot_coefficients(results, static)
    plot_importance(results, static)
    plot_heatmap(df, static)
    plot_cv_bars(results, static)
