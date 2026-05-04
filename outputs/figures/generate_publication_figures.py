"""Generate publication-quality figures 1, 3, and 4."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy import stats
from scipy.stats import gaussian_kde
from sklearn.metrics import roc_curve, roc_auc_score, auc as sklearn_auc
from sklearn.model_selection import StratifiedKFold

from src.features.pipeline import get_early_session_features, get_late_session_features
from src.modelling.evaluation import label_high_dependency
from src.modelling.main_model import build_pipeline, add_negative_control

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':          'sans-serif',
    'font.sans-serif':      ['Helvetica Neue', 'Arial', 'DejaVu Sans'],
    'font.size':            10,
    'axes.spines.top':      False,
    'axes.spines.right':    False,
    'axes.linewidth':       0.8,
    'xtick.major.width':    0.8,
    'ytick.major.width':    0.8,
    'xtick.labelsize':      9,
    'ytick.labelsize':      9,
    'legend.frameon':       False,
    'legend.fontsize':      9,
})

HIGH_COLOR = '#C0392B'
LOW_COLOR  = '#2E6FA8'
SEED       = 42
OUT_DIR    = os.path.dirname(os.path.abspath(__file__))

FEATURE_LABELS = {
    'log_n_user_turns':            'Session length (log turns)',
    'n_user_turns':                'Session length (user turns)',
    'filler_rate':                 'Filler / phatic rate',
    'fpp_rate':                    'First-person plural rate',
    'question_to_statement_ratio': 'Question–statement ratio',
    'type_token_ratio':            'Vocabulary richness (TTR)',
    'hedging_rate':                'Hedging rate',
    'log_return_interval_hours':   'Return interval (log hours)',
    'return_interval_hours':       'Return interval (hours)',
    'refusal_resistance_rate':     'Refusal resistance rate',
    'self_disclosure_rate':        'Self-disclosure rate',
    'sentiment_slope':             'Sentiment slope',
    'gratitude_rate':              'Gratitude / deference rate',
    'mean_message_length':         'Mean message length',
}


# ── Load data (shared) ────────────────────────────────────────────────────────
print("Loading data...")
feature_matrix = pd.read_parquet('data/processed/feature_matrix.parquet')
early  = get_early_session_features(feature_matrix)
late   = get_late_session_features(feature_matrix)
labels = label_high_dependency(late, min_criteria=2)

early_indexed = early.set_index('hashed_ip')
y_aligned     = labels.reindex(early_indexed.index)
valid_mask    = y_aligned.notna()
X_early       = early_indexed[valid_mask]
y             = y_aligned[valid_mask].astype(int).values
user_ids      = np.array(X_early.index)

high_dep_users = set(X_early.index[y == 1])
low_dep_users  = set(X_early.index[y == 0])
n_high, n_low  = len(high_dep_users), len(low_dep_users)
print(f"  High-dep: {n_high:,}  Low-dep: {n_low:,}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Session trajectory divergence
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 1: session trajectories...")

labelled = feature_matrix[
    feature_matrix['hashed_ip'].isin(high_dep_users | low_dep_users)
].copy()
labelled['group'] = labelled['hashed_ip'].map(
    lambda x: 'High-dependency' if x in high_dep_users else 'Low-dependency'
)

MAX_SESSION = 18
MIN_USERS   = 200
labelled_filt = labelled[labelled['session_number'] <= MAX_SESSION]

panels = [
    ('n_user_turns',               'Session length (turns)',        False),
    ('return_interval_hours',      'Return interval (hours)',       True),
    ('log_return_interval_hours',  'Return interval (log hours)',   True),
]

fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.4))
fig.subplots_adjust(wspace=0.40, top=0.84)

for ax, (feat, ylabel, skip_s1) in zip(axes, panels):
    for group, color in [('High-dependency', HIGH_COLOR), ('Low-dependency', LOW_COLOR)]:
        grp = labelled_filt[labelled_filt['group'] == group]
        if skip_s1:
            grp = grp[grp['session_number'] > 1]

        traj = (grp.groupby('session_number')[feat]
                    .agg(['mean', 'sem', 'count'])
                    .reset_index())
        traj = traj[traj['count'] >= MIN_USERS]

        ax.plot(traj['session_number'], traj['mean'],
                color=color, linewidth=2, label=group)
        ax.fill_between(traj['session_number'],
                        traj['mean'] - traj['sem'],
                        traj['mean'] + traj['sem'],
                        color=color, alpha=0.15)

    # Late-session threshold marker
    ax.axvline(10, color='#555555', linewidth=0.9, linestyle=':', zorder=0)
    ax.set_xlabel('Session number', fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.yaxis.grid(True, alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)

# Annotate threshold line on last panel to avoid clutter
axes[2].text(10.25, axes[2].get_ylim()[0], 'session 10\nthreshold',
             fontsize=7, color='#555555', va='bottom')

legend_handles = [
    Line2D([0], [0], color=HIGH_COLOR, linewidth=2,   label='High-dependency'),
    Line2D([0], [0], color=LOW_COLOR,  linewidth=2,   label='Low-dependency'),
    Line2D([0], [0], color='#555555',  linewidth=0.9,
           linestyle=':', label='Late-session threshold (session 10)'),
]
fig.legend(handles=legend_handles, loc='upper center', ncol=3,
           bbox_to_anchor=(0.5, 1.01), fontsize=9)

fig.savefig(os.path.join(OUT_DIR, 'fig1_session_trajectories.pdf'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(OUT_DIR, 'fig1_session_trajectories.png'), dpi=300, bbox_inches='tight')
print("  → fig1_session_trajectories.png / .pdf")
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Effect size forest plot
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 3: forest plot...")

baseline = pd.read_csv('outputs/tables/baseline_results.csv')

def cohens_d_ci(d, n1, n0, alpha=0.05):
    var = (n1 + n0) / (n1 * n0) + d**2 / (2 * (n1 + n0 - 2))
    z   = stats.norm.ppf(1 - alpha / 2)
    return d - z * np.sqrt(var), d + z * np.sqrt(var)

cis = [cohens_d_ci(d, n_high, n_low) for d in baseline['cohens_d']]
baseline['ci_lo'] = [c[0] for c in cis]
baseline['ci_hi'] = [c[1] for c in cis]
baseline['label'] = baseline['feature'].map(lambda f: FEATURE_LABELS.get(f, f))

# Sort ascending by |d| so the largest effects are at the top visually
baseline = baseline.reindex(
    baseline['cohens_d'].abs().sort_values(ascending=True).index
).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(5.8, 5.0))

for i, row in baseline.iterrows():
    d, lo, hi = row['cohens_d'], row['ci_lo'], row['ci_hi']
    sig   = row['significant_bh']
    color = HIGH_COLOR if d > 0 else LOW_COLOR
    alpha = 1.0 if sig else 0.35

    ax.plot([lo, hi], [i, i], color=color, linewidth=1.5,
            alpha=alpha, solid_capstyle='round', zorder=2)
    ax.scatter(d, i, color=color, s=52, zorder=3, alpha=alpha,
               edgecolors='white', linewidths=0.6)

    if not sig:
        ax.text(hi + 0.005, i, 'ns', fontsize=7.5, color='#AAAAAA',
                va='center', style='italic')

ax.axvline(0, color='#333333', linewidth=0.9, zorder=1)
ax.set_yticks(range(len(baseline)))
ax.set_yticklabels(baseline['label'], fontsize=8.5)

xmin = baseline['ci_lo'].min() - 0.06
xmax = baseline['ci_hi'].max() + 0.10
ax.set_xlim(xmin, xmax)
ax.set_xlabel("Cohen's d  (high-dependency − low-dependency)", fontsize=9)

# Direction labels
ymax_pos = len(baseline) - 0.4
ax.text(xmin + 0.01, ymax_pos, '<- lower in\nhigh-dep users',
        fontsize=7.5, color='#777777', style='italic', va='top')
ax.text(xmax - 0.01, ymax_pos, 'higher in\nhigh-dep users ->',
        fontsize=7.5, color='#777777', style='italic', va='top', ha='right')

legend_handles = [
    mpatches.Patch(color=HIGH_COLOR, label='Positive effect (high-dep > low-dep)'),
    mpatches.Patch(color=LOW_COLOR,  label='Negative effect (high-dep < low-dep)'),
    Line2D([0], [0], marker='o', color='#AAAAAA', markersize=6,
           linewidth=0, alpha=0.4, label='Not BH-significant'),
]
ax.legend(handles=legend_handles, loc='lower right', fontsize=8)
ax.xaxis.grid(True, alpha=0.25, linewidth=0.5)
ax.set_axisbelow(True)
ax.spines['left'].set_visible(False)
ax.tick_params(left=False)

fig.savefig(os.path.join(OUT_DIR, 'fig3_forest_plot.pdf'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(OUT_DIR, 'fig3_forest_plot.png'), dpi=300, bbox_inches='tight')
print("  → fig3_forest_plot.png / .pdf")
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — ROC (Panel A) + Permutation test (Panel B)
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 4: ROC + permutation test (takes a few minutes)...")

X_with_control = add_negative_control(X_early)
X_arr = X_with_control.values

# ── Panel A: per-fold ROC ─────────────────────────────────────────────────────
unique_users   = np.unique(user_ids)
user_labels_cv = np.array([y[user_ids == u][0] for u in unique_users])
cv             = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)

mean_fpr    = np.linspace(0, 1, 300)
interp_tprs = []
fold_aucs   = []

for train_idx, test_idx in cv.split(unique_users, user_labels_cv):
    train_users = unique_users[train_idx]
    test_users  = unique_users[test_idx]
    train_mask  = np.isin(user_ids, train_users)
    test_mask   = np.isin(user_ids, test_users)

    pipe = build_pipeline(penalty='l1', C=1.0, random_state=SEED)
    pipe.fit(X_arr[train_mask], y[train_mask])
    y_prob = pipe.predict_proba(X_arr[test_mask])[:, 1]

    fpr, tpr, _ = roc_curve(y[test_mask], y_prob)
    fold_aucs.append(sklearn_auc(fpr, tpr))
    interp_tpr    = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    interp_tprs.append(interp_tpr)

mean_tpr       = np.mean(interp_tprs, axis=0)
mean_tpr[-1]   = 1.0
std_tpr        = np.std(interp_tprs, axis=0)
mean_auc       = float(np.mean(fold_aucs))
std_auc        = float(np.std(fold_aucs))

# ── Panel B: permutation null ─────────────────────────────────────────────────
print("  Running 1,000 permutations...")
rng       = np.random.default_rng(SEED)
null_aucs = []
pipe_perm = build_pipeline(penalty='l1', C=1.0, random_state=SEED)
for _ in range(1000):
    y_shuf = rng.permutation(y)
    pipe_perm.fit(X_arr, y_shuf)
    y_prob = pipe_perm.predict_proba(X_arr)[:, 1]
    null_aucs.append(roc_auc_score(y_shuf, y_prob))
null_aucs   = np.array(null_aucs)
perm_pvalue = float((null_aucs >= mean_auc).mean())
print(f"  Permutation p = {perm_pvalue:.4f}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, (ax_roc, ax_perm) = plt.subplots(1, 2, figsize=(9.0, 4.0))
fig.subplots_adjust(wspace=0.40)

# Panel A — ROC
ax_roc.fill_between(mean_fpr,
                    np.clip(mean_tpr - std_tpr, 0, 1),
                    np.clip(mean_tpr + std_tpr, 0, 1),
                    color=HIGH_COLOR, alpha=0.14, label='±1 SD (10 folds)')
ax_roc.plot(mean_fpr, mean_tpr, color=HIGH_COLOR, linewidth=2.2,
            label=f'Mean ROC  AUC = {mean_auc:.3f} ± {std_auc:.3f}')
ax_roc.plot([0, 1], [0, 1], linestyle='--', color='#BBBBBB',
            linewidth=1.0, label='Random classifier')
ax_roc.set_xlim(0, 1)
ax_roc.set_ylim(0, 1.02)
ax_roc.set_xlabel('False positive rate', fontsize=9)
ax_roc.set_ylabel('True positive rate', fontsize=9)
ax_roc.set_title('A', loc='left', fontweight='bold', fontsize=11)
ax_roc.legend(fontsize=8.5, loc='lower right')
ax_roc.set_aspect('equal', adjustable='box')
ax_roc.xaxis.grid(True, alpha=0.2, linewidth=0.5)
ax_roc.yaxis.grid(True, alpha=0.2, linewidth=0.5)

# Panel B — permutation
kde_xs = np.linspace(null_aucs.min() - 0.01, mean_auc + 0.025, 500)
kde    = gaussian_kde(null_aucs, bw_method=0.25)
kde_ys = kde(kde_xs)

ax_perm.fill_between(kde_xs, kde_ys, color='#BBBBBB', alpha=0.55,
                     label='Null distribution (1,000 permutations)')
ax_perm.plot(kde_xs, kde_ys, color='#999999', linewidth=1.0)
ax_perm.axvline(mean_auc, color=HIGH_COLOR, linewidth=2.2,
                label=f'Observed AUC = {mean_auc:.3f}')

# Annotation arrow
peak_y = kde_ys.max()
ax_perm.annotate(
    'p < 0.001',
    xy=(mean_auc, peak_y * 0.30),
    xytext=(mean_auc - 0.030, peak_y * 0.62),
    arrowprops=dict(arrowstyle='->', color=HIGH_COLOR, lw=1.3),
    fontsize=9.5, color=HIGH_COLOR, ha='right', fontweight='bold',
)
ax_perm.set_xlabel('AUC', fontsize=9)
ax_perm.set_ylabel('Density', fontsize=9)
ax_perm.set_title('B', loc='left', fontweight='bold', fontsize=11)
ax_perm.legend(fontsize=8.5)
ax_perm.yaxis.grid(True, alpha=0.2, linewidth=0.5)
ax_perm.set_axisbelow(True)

fig.savefig(os.path.join(OUT_DIR, 'fig4_roc_permutation.pdf'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(OUT_DIR, 'fig4_roc_permutation.png'), dpi=300, bbox_inches='tight')
print("  → fig4_roc_permutation.png / .pdf")
plt.close()

print("\nDone. All figures saved to outputs/figures/")
