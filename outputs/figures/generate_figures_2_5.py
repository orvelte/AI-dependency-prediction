"""Generate figures 2 (dependency fingerprint radar) and 5 (UMAP projection)."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler
import umap

from src.features.pipeline import get_early_session_features, get_late_session_features
from src.modelling.evaluation import label_high_dependency

plt.rcParams.update({
    'font.family':       'sans-serif',
    'font.sans-serif':   ['Helvetica Neue', 'Arial', 'DejaVu Sans'],
    'font.size':         10,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.linewidth':    0.8,
    'xtick.labelsize':   9,
    'ytick.labelsize':   9,
    'legend.frameon':    False,
    'legend.fontsize':   9,
})

HIGH_COLOR = '#C0392B'
LOW_COLOR  = '#2E6FA8'
SEED       = 42
OUT_DIR    = os.path.dirname(os.path.abspath(__file__))


# ── Load data ─────────────────────────────────────────────────────────────────
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
print(f"  N={len(y):,}  high-dep={y.sum():,}  low-dep={(1-y).sum():,}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Dependency fingerprint (radar / spider chart)
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 2: dependency fingerprint radar...")

# 8 features; type_token_ratio and log_return_interval_hours are sign-inverted
# so that "outward from centre" = more dependency-associated on every axis.
RADAR_FEATURES = [
    'log_n_user_turns',
    'fpp_rate',
    'refusal_resistance_rate',
    'hedging_rate',
    'filler_rate',
    'question_to_statement_ratio',
    'type_token_ratio',
    'log_return_interval_hours',
]
RADAR_LABELS = [
    'Session\nlength',
    'First-person\nplural',
    'Refusal\nresistance',
    'Hedging',
    'Filler /\nphatic',
    'Question\nratio',
    'Vocab.\nnarrowing*',
    'Return\nspeed*',
]
FLIP = {'type_token_ratio': -1, 'log_return_interval_hours': -1}

scaler = StandardScaler()
X_std = pd.DataFrame(
    scaler.fit_transform(X_early[RADAR_FEATURES]),
    index=X_early.index, columns=RADAR_FEATURES,
)
for feat, sign in FLIP.items():
    X_std[feat] *= sign

high_means = X_std[y == 1].mean().values
low_means  = X_std[y == 0].mean().values

N      = len(RADAR_FEATURES)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False)

# Shift so all plotted values are positive; zero ring = population mean
OFFSET   = 0.55
max_dev  = max(np.abs(high_means).max(), np.abs(low_means).max())
r_limit  = OFFSET + max_dev * 1.35

def close(arr):
    return np.concatenate([arr, [arr[0]]])

angles_c    = close(angles)
high_vals_c = close(high_means + OFFSET)
low_vals_c  = close(low_means  + OFFSET)
zero_ring   = np.full(len(angles_c), OFFSET)

fig, ax = plt.subplots(figsize=(6.0, 6.2), subplot_kw=dict(polar=True))
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# Background rings
for r in [OFFSET * 0.5, OFFSET, OFFSET * 1.5]:
    ax.plot(angles_c, np.full_like(angles_c, r),
            color='#DDDDDD', linewidth=0.6, zorder=0)

# Zero ring (population mean)
ax.plot(angles_c, zero_ring, color='#AAAAAA', linewidth=1.0,
        linestyle='--', zorder=1, label='_nolegend_')

# Low-dependency
ax.fill(angles_c, low_vals_c,  color=LOW_COLOR,  alpha=0.18, zorder=2)
ax.plot(angles_c, low_vals_c,  color=LOW_COLOR,  linewidth=2.0, zorder=3)

# High-dependency
ax.fill(angles_c, high_vals_c, color=HIGH_COLOR, alpha=0.22, zorder=4)
ax.plot(angles_c, high_vals_c, color=HIGH_COLOR, linewidth=2.0, zorder=5)

# Spoke labels
ax.set_xticks(angles)
ax.set_xticklabels(RADAR_LABELS, fontsize=9.5, fontweight='normal')
ax.tick_params(pad=10)

ax.set_ylim(0, r_limit)
ax.set_yticklabels([])
ax.set_rlabel_position(0)
ax.grid(color='#EEEEEE', linewidth=0.5, axis='y')
ax.spines['polar'].set_visible(False)

legend_handles = [
    mpatches.Patch(color=HIGH_COLOR, alpha=0.7, label=f'High-dependency  (n={y.sum():,})'),
    mpatches.Patch(color=LOW_COLOR,  alpha=0.7, label=f'Low-dependency   (n={(1-y).sum():,})'),
]
ax.legend(handles=legend_handles, loc='upper right',
          bbox_to_anchor=(1.38, 1.18), fontsize=9)

ax.set_title('Early-session dependency fingerprint\n(standardised z-scores, sessions 1–3)',
             fontsize=10.5, pad=22, fontweight='normal')

fig.text(0.5, 0.01,
         '* sign inverted: outward = higher dependency association on all axes',
         ha='center', fontsize=7.5, color='#777777', style='italic')

fig.savefig(os.path.join(OUT_DIR, 'fig2_dependency_fingerprint.pdf'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(OUT_DIR, 'fig2_dependency_fingerprint.png'), dpi=300, bbox_inches='tight')
print("  -> fig2_dependency_fingerprint.png / .pdf")
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — UMAP projection of early-session feature space
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 5: UMAP projection (takes ~1-2 min)...")

# Use log versions; exclude raw collinear counterparts
UMAP_FEATURES = [
    'log_n_user_turns', 'log_return_interval_hours',
    'fpp_rate', 'self_disclosure_rate', 'hedging_rate',
    'filler_rate', 'gratitude_rate', 'refusal_resistance_rate',
    'question_to_statement_ratio', 'mean_message_length',
    'type_token_ratio', 'sentiment_slope',
]

X_umap_raw = X_early[UMAP_FEATURES].fillna(0).values
X_umap_std = StandardScaler().fit_transform(X_umap_raw)

reducer   = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.12,
                      random_state=SEED, low_memory=False)
embedding = reducer.fit_transform(X_umap_std)
print("  UMAP done.")

ex, ey = embedding[:, 0], embedding[:, 1]

# KDE density contours per group
x_grid = np.linspace(ex.min() - 0.5, ex.max() + 0.5, 150)
y_grid = np.linspace(ey.min() - 0.5, ey.max() + 0.5, 150)
xx, yy = np.meshgrid(x_grid, y_grid)
grid_pts = np.vstack([xx.ravel(), yy.ravel()])

fig, ax = plt.subplots(figsize=(7.0, 5.5))

for mask, color, label in [
    (y == 0, LOW_COLOR,  f'Low-dependency  (n={(1-y).sum():,})'),
    (y == 1, HIGH_COLOR, f'High-dependency (n={y.sum():,})'),
]:
    pts = embedding[mask]

    # Scatter
    ax.scatter(pts[:, 0], pts[:, 1], c=color, s=3, alpha=0.18,
               linewidths=0, rasterized=True, zorder=2)

    # KDE contours
    kde    = gaussian_kde(pts.T, bw_method=0.18)
    z      = kde(grid_pts).reshape(xx.shape)
    levels = np.percentile(z[z > 0], [40, 60, 78, 92])
    ax.contourf(xx, yy, z, levels=levels, colors=[color],
                alpha=0.12, zorder=1)
    ax.contour(xx, yy, z,  levels=levels, colors=[color],
               linewidths=0.8, alpha=0.55, zorder=3)

ax.set_xlabel('UMAP dimension 1', fontsize=9)
ax.set_ylabel('UMAP dimension 2', fontsize=9)
ax.set_title(
    'UMAP projection of early-session (sessions 1–3) feature space\n'
    'Contours show kernel density estimate per group',
    fontsize=10, pad=10,
)

legend_handles = [
    mpatches.Patch(color=HIGH_COLOR, alpha=0.7, label=f'High-dependency  (n={y.sum():,})'),
    mpatches.Patch(color=LOW_COLOR,  alpha=0.7, label=f'Low-dependency   (n={(1-y).sum():,})'),
]
ax.legend(handles=legend_handles, fontsize=9, loc='best')
ax.tick_params(left=False, bottom=False)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

fig.savefig(os.path.join(OUT_DIR, 'fig5_umap_projection.pdf'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(OUT_DIR, 'fig5_umap_projection.png'), dpi=300, bbox_inches='tight')
print("  -> fig5_umap_projection.png / .pdf")
plt.close()

print("\nDone.")
