"""Generate the dataset overview figure for the WildChat longitudinal cohort.

Loads data/interim/conv_metadata.parquet (already processed by notebook 01).
Saves outputs/figures/dataset_overview.png.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_parquet("data/interim/conv_metadata.parquet")

# ── Derived ───────────────────────────────────────────────────────────────────
sessions_per_user = df.groupby("hashed_ip")["session_number"].max()
traj = df[df["session_number"] <= 30].copy()

# Mean ± SE of user turns per session number
turn_traj = (
    traj.groupby("session_number")["n_user_turns"]
    .agg(["mean", "std", "count"])
    .reset_index()
)
turn_traj["se"] = turn_traj["std"] / np.sqrt(turn_traj["count"])
turn_traj["ci_lo"] = turn_traj["mean"] - 1.96 * turn_traj["se"]
turn_traj["ci_hi"] = turn_traj["mean"] + 1.96 * turn_traj["se"]

# Median return interval per session number (hours, log-scaled)
ri_traj = (
    traj[traj["return_interval_hours"].notna()]
    .groupby("session_number")["return_interval_hours"]
    .agg(["median", lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)])
    .reset_index()
)
ri_traj.columns = ["session_number", "median", "q25", "q75"]

# ── Style ─────────────────────────────────────────────────────────────────────
BLUE   = "#2E6FA8"
ORANGE = "#E07B39"
GREY   = "#8A8A8A"
RED    = "#C0392B"

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})

fig = plt.figure(figsize=(14, 10))
fig.suptitle(
    "WildChat Longitudinal Cohort — Dataset Overview\n"
    "12,497 users · 505,397 conversations · 386-day span (Apr 2023 – Apr 2024)",
    fontsize=13, fontweight="bold", y=0.98
)

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

# ── Panel A: Session count distribution ───────────────────────────────────────
bins = np.logspace(np.log10(10), np.log10(sessions_per_user.max() + 1), 60)
ax1.hist(sessions_per_user, bins=bins, color=BLUE, alpha=0.85, edgecolor="white", linewidth=0.4)
for thresh, label, col in [(10, "≥10 (min)", RED), (50, "≥50", ORANGE), (100, "≥100", GREY)]:
    n = (sessions_per_user >= thresh).sum()
    ax1.axvline(thresh, color=col, linestyle="--", linewidth=1.4,
                label=f"{label}  (n={n:,})")
ax1.set_xscale("log")
ax1.set_xlabel("Conversations per user (log scale)")
ax1.set_ylabel("Number of users")
ax1.set_title("A  Cohort Composition")
ax1.legend(frameon=False, loc="upper right")

# ── Panel B: Mean user turns per session number ────────────────────────────────
ax2.fill_between(turn_traj["session_number"],
                 turn_traj["ci_lo"].clip(lower=0), turn_traj["ci_hi"],
                 color=BLUE, alpha=0.18)
ax2.plot(turn_traj["session_number"], turn_traj["mean"],
         color=BLUE, linewidth=2, marker="o", markersize=3.5)
ax2.axvspan(1, 3, color=BLUE, alpha=0.08, label="Predictor window (1–3)")
ax2.axvspan(10, 30, color=ORANGE, alpha=0.08, label="Outcome window (10+)")
ax2.set_xlabel("Session number (chronological)")
ax2.set_ylabel("Mean user turns per conversation")
ax2.set_title("B  Engagement Trajectory (mean ± 95% CI)")
ax2.legend(frameon=False, loc="upper left")
ax2.set_xlim(1, 30)

# ── Panel C: Median return interval per session number ────────────────────────
ax3.fill_between(ri_traj["session_number"], ri_traj["q25"], ri_traj["q75"],
                 color=ORANGE, alpha=0.2, label="IQR")
ax3.plot(ri_traj["session_number"], ri_traj["median"],
         color=ORANGE, linewidth=2, marker="o", markersize=3.5, label="Median")
ax3.axvspan(1, 3, color=BLUE, alpha=0.08, label="Predictor window")
ax3.axvspan(10, 30, color=ORANGE, alpha=0.08, label="Outcome window")
ax3.set_yscale("log")
ax3.set_xlabel("Session number (chronological)")
ax3.set_ylabel("Return interval (hours, log scale)")
ax3.set_title("C  Return Interval Trajectory (median + IQR)")
ax3.legend(frameon=False, loc="upper right", ncol=2)
ax3.set_xlim(1, 30)

# ── Panel D: Turn count distribution (raw vs log) ─────────────────────────────
ax4_twin = ax4.twinx()

clip_val = df["n_user_turns"].quantile(0.99)
raw = df["n_user_turns"].clip(upper=clip_val)
log_vals = np.log1p(df["n_user_turns"])

ax4.hist(raw, bins=50, color=BLUE, alpha=0.55, label="Raw (clipped at 99th pct)")
ax4_twin.hist(log_vals, bins=50, color=ORANGE, alpha=0.45, label="log(1 + turns)")

ax4.set_xlabel("User turns per conversation")
ax4.set_ylabel("Count (raw)", color=BLUE)
ax4_twin.set_ylabel("Count (log-transformed)", color=ORANGE)
ax4.tick_params(axis="y", colors=BLUE)
ax4_twin.tick_params(axis="y", colors=ORANGE)
ax4.set_title("D  Turn Count Distribution\n(median=1, mean=2.2, max=249)")

legend_elements = [
    Line2D([0], [0], color=BLUE, lw=6, alpha=0.55, label="Raw"),
    Line2D([0], [0], color=ORANGE, lw=6, alpha=0.55, label="log(1 + turns)"),
]
ax4.legend(handles=legend_elements, frameon=False, loc="upper right")

# ── Annotation: key numbers ───────────────────────────────────────────────────
fig.text(0.5, 0.005,
         "Panels B & C motivate the core hypothesis: engagement escalates and return intervals shorten across sessions — "
         "consistent with goal-directed → habitual transition.",
         ha="center", fontsize=9, style="italic", color=GREY)

out_path = "outputs/figures/dataset_overview.png"
plt.savefig(out_path, dpi=180, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.show()
