"""Pipeline overview figure — Nature publication style."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({
    'font.family':     'sans-serif',
    'font.sans-serif': ['Helvetica Neue', 'Arial', 'DejaVu Sans'],
})

# ── Palette ───────────────────────────────────────────────────────────────────
PC        = ['#1A6FA8', '#1E9070', '#C07820', '#B03030', '#6B4A9A']
BG_BOX    = '#FAFAFA'
BG_RESULT = '#1C1C2E'
TEXT_DARK = '#1A1A1A'
TEXT_MID  = '#505050'

FW, FH  = 18.5, 7.5
MARGIN  = 0.32
BOX_GAP = 0.22
BOX_Y0  = 1.10
BOX_Y1  = FH - 0.40
N       = 5
BOX_W   = (FW - 2*MARGIN - (N-1)*BOX_GAP) / N
R       = 0.15

def bx0(i): return MARGIN + i * (BOX_W + BOX_GAP)
def bcx(i): return bx0(i) + BOX_W / 2

fig = plt.figure(figsize=(FW, FH), facecolor='white')
ax  = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, FW)
ax.set_ylim(0, FH)
ax.axis('off')


# ── Box drawing helper ────────────────────────────────────────────────────────
def draw_box(i, title, items, stat, color, has_timeline=False):
    x0 = bx0(i)
    cx = bcx(i)
    bh = BOX_Y1 - BOX_Y0

    # Background
    ax.add_patch(FancyBboxPatch(
        (x0, BOX_Y0), BOX_W, bh,
        boxstyle=f'round,pad={R}',
        facecolor=BG_BOX, edgecolor='#DCDCDC', linewidth=1.1, zorder=2))

    # Left accent strip
    ax.add_patch(mpatches.Rectangle(
        (x0 + R * 0.4, BOX_Y0 + 0.28), 0.062, bh - 0.56,
        facecolor=color, edgecolor='none', zorder=3))

    # Phase number circle
    circ_cx = x0 + 0.33
    circ_cy = BOX_Y1 - 0.46
    ax.add_patch(mpatches.Circle(
        (circ_cx, circ_cy), 0.20,
        facecolor=color, edgecolor='none', zorder=4))
    ax.text(circ_cx, circ_cy, str(i + 1),
            ha='center', va='center',
            fontsize=9.5, fontweight='bold', color='white', zorder=5)

    # Phase title
    ax.text(x0 + 0.62, circ_cy, title,
            ha='left', va='center',
            fontsize=8.8, fontweight='bold', color=color, zorder=4)

    # Separator line
    sep_y = BOX_Y1 - 0.82
    ax.plot([x0 + 0.16, x0 + BOX_W - 0.10],
            [sep_y, sep_y],
            color='#E5E5E5', linewidth=0.9, zorder=3)

    # Content
    ty = sep_y - 0.20
    lx = x0 + 0.22

    if has_timeline:
        tl_x0 = x0 + 0.18
        tl_x1 = x0 + BOX_W - 0.12
        tl_y  = ty - 0.10
        tl_h  = 0.21
        n_s   = 18

        ax.add_patch(mpatches.FancyBboxPatch(
            (tl_x0, tl_y - tl_h/2), tl_x1 - tl_x0, tl_h,
            boxstyle='round,pad=0.03',
            facecolor='#E0E0E0', edgecolor='none', zorder=4))

        e_end = tl_x0 + (3/n_s) * (tl_x1 - tl_x0)
        ax.add_patch(mpatches.FancyBboxPatch(
            (tl_x0, tl_y - tl_h/2), e_end - tl_x0, tl_h,
            boxstyle='round,pad=0.03',
            facecolor=PC[2], edgecolor='none', zorder=5))

        l_start = tl_x0 + (9/n_s) * (tl_x1 - tl_x0)
        ax.add_patch(mpatches.Rectangle(
            (l_start, tl_y - tl_h/2), tl_x1 - l_start, tl_h,
            facecolor=PC[3], edgecolor='none', zorder=5))

        ax.text((tl_x0 + e_end)/2, tl_y + tl_h/2 + 0.09,
                'Sessions 1–3', ha='center', va='bottom',
                fontsize=6.6, color=PC[2], fontweight='bold', zorder=6)
        ax.text((l_start + tl_x1)/2, tl_y + tl_h/2 + 0.09,
                'Sessions 10+', ha='center', va='bottom',
                fontsize=6.6, color=PC[3], fontweight='bold', zorder=6)
        ax.text((tl_x0 + e_end)/2, tl_y - tl_h/2 - 0.08,
                'Predictors', ha='center', va='top',
                fontsize=6.2, color='#888888', zorder=6)
        ax.text((l_start + tl_x1)/2, tl_y - tl_h/2 - 0.08,
                'Outcome', ha='center', va='top',
                fontsize=6.2, color='#888888', zorder=6)

        ty = tl_y - tl_h/2 - 0.35

    for text, kind in items:
        if not text:
            ty -= 0.16
            continue
        if kind == 'head':
            ax.text(lx, ty, text, ha='left', va='top',
                    fontsize=7.7, fontweight='bold', color=TEXT_DARK, zorder=4)
            ty -= 0.35
        elif kind == 'cat':
            ax.text(lx, ty, text, ha='left', va='top',
                    fontsize=7.3, fontweight='bold', color=color, zorder=4)
            ty -= 0.31
        elif kind == 'bullet':
            ax.text(lx + 0.10, ty, f'–  {text}', ha='left', va='top',
                    fontsize=6.9, color=TEXT_MID, zorder=4)
            ty -= 0.28

    ax.text(cx, BOX_Y0 + 0.30, stat,
            ha='center', va='center',
            fontsize=7.7, fontweight='bold', color=color, zorder=4)


# ── Phase content ─────────────────────────────────────────────────────────────
phases = [
    {
        'title': 'DATA',
        'color': PC[0],
        'items': [
            ('WildChat-1M corpus',                    'head'),
            ('509,861 conversations',                 'bullet'),
            ('Apr 2023 – Apr 2024  (386 days)',  'bullet'),
            ('Role + content per turn',               'bullet'),
            ('',                                      ''),
            ('Cohort filter',                         'head'),
            ('≥ 10 sessions per user retained',  'bullet'),
            ('Redacted conversations excluded (0.9%)','bullet'),
            ('505,397 sessions in analysis',          'bullet'),
        ],
        'stat': '12,577 qualifying users',
        'timeline': False,
    },
    {
        'title': 'FEATURES',
        'color': PC[1],
        'items': [
            ('Lexical',                                     'cat'),
            ('FPP rate  |  Hedging  |  Self-disclosure',   'bullet'),
            ('Filler / phatic expression rate',            'bullet'),
            ('Pragmatic',                                  'cat'),
            ('Refusal resistance rate',                    'bullet'),
            ('Gratitude rate  |  Q–S ratio',         'bullet'),
            ('Session-level',                              'cat'),
            ('Type–token ratio  |  Message length',  'bullet'),
            ('Return interval  |  Sentiment slope',       'bullet'),
        ],
        'stat': '12 features  ·  single-pass token scan',
        'timeline': False,
    },
    {
        'title': 'WINDOWS & LABEL',
        'color': PC[2],
        'items': [
            ('Dependency label  (≥ 2 of 4)',       'head'),
            ('Return interval < cohort median',         'bullet'),
            ('Session length > 75th percentile',       'bullet'),
            ('Refusal resistance rate > 0',            'bullet'),
            ('FPP rate > cohort mean (late sessions)', 'bullet'),
        ],
        'stat': 'High-dep: 24.1%  (n = 3,014 / 12,497)',
        'timeline': True,
    },
    {
        'title': 'MODELLING',
        'color': PC[3],
        'items': [
            ('Univariate baselines',               'head'),
            ('Logistic regression per feature',    'bullet'),
            ('AUC-ROC  +  Cohen’s d',        'bullet'),
            ('BH FDR correction  (12 tests)',      'bullet'),
            ('',                                   ''),
            ('Main model',                         'head'),
            ('L1-regularised logistic regression', 'bullet'),
            ('User-level stratified 10-fold CV',   'bullet'),
            ('class_weight = balanced',  'bullet'),
        ],
        'stat': 'AUC = 0.712  ± 0.014',
        'timeline': False,
    },
    {
        'title': 'VALIDATION',
        'color': PC[4],
        'items': [
            ('Permutation test',                        'head'),
            ('1,000 label shuffles',                   'bullet'),
            ('p < 0.001  (0 / 1,000 exceed observed)', 'bullet'),
            ('',                                        ''),
            ('Negative control',                        'head'),
            ('Punctuation rate  (theoretically null)',  'bullet'),
            ('Rank: 14 / 15  ·  coef ≈ 0  [pass]', 'bullet'),
            ('',                                        ''),
            ('Sensitivity analysis',                    'head'),
            ('Threshold ≥1–≥3  |  AUC: 0.69–0.75', 'bullet'),
        ],
        'stat': 'Non-spurious  ·  Neg. control: PASSED',
        'timeline': False,
    },
]

for i, p in enumerate(phases):
    draw_box(i, p['title'], p['items'], p['stat'], p['color'],
             has_timeline=p['timeline'])

# ── Connecting arrows ─────────────────────────────────────────────────────────
arrow_y = (BOX_Y0 + BOX_Y1) / 2
for i in range(N - 1):
    ax.annotate('',
        xy=(bx0(i+1) - 0.02, arrow_y),
        xytext=(bx0(i) + BOX_W + 0.02, arrow_y),
        arrowprops=dict(
            arrowstyle='->', color='#BBBBBB',
            lw=2.0, mutation_scale=18))

# ── Results strip ─────────────────────────────────────────────────────────────
sx0, sx1 = MARGIN, FW - MARGIN
sy0, sy1 = 0.12, 0.84

ax.add_patch(FancyBboxPatch(
    (sx0, sy0), sx1 - sx0, sy1 - sy0,
    boxstyle='round,pad=0.08',
    facecolor=BG_RESULT, edgecolor='none', zorder=2))

results = [
    ('AUC = 0.712 ± 0.014',   PC[3]),
    ('Permutation  p < 0.001',                PC[0]),
    ('9 / 12 features  BH-significant',       PC[1]),
    ('Negative control  rank 14 / 15',        PC[4]),
    ('Feature ranking stable across thresholds',        '#888888'),
]
for j, (txt, col) in enumerate(results):
    x = sx0 + (sx1 - sx0) * (j + 0.5) / len(results)
    ax.text(x, (sy0 + sy1) / 2, txt,
            ha='center', va='center',
            fontsize=8.0, fontweight='bold', color=col, zorder=3)

# ── Title ─────────────────────────────────────────────────────────────────────
ax.text(FW / 2, FH - 0.18,
        'Analytical Pipeline  —  Linguistic Markers of Dependency Formation '
        'in Longitudinal Human–AI Conversations',
        ha='center', va='top',
        fontsize=9.5, fontweight='bold', color=TEXT_DARK)

fig.savefig('outputs/figures/fig_pipeline.pdf', dpi=300, bbox_inches='tight')
fig.savefig('outputs/figures/fig_pipeline.png', dpi=300, bbox_inches='tight')
print("Saved.")
