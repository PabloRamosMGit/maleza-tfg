import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_PATH = r"C:\Users\pablo\maleza-tfg\csiro-biomass"
TRAIN_CSV_PATH = os.path.join(BASE_PATH, "train.csv")

# ── Load & deduplicate ─────────────────────────────────────────────────────
# Each physical sample appears multiple times (one row per target_name).
# We count unique image_paths per species to get the real sample count.
df = pd.read_csv(TRAIN_CSV_PATH)
unique_samples = df.drop_duplicates(subset="image_path")
species_counts = (
    unique_samples["Species"]
    .value_counts()
    .sort_values(ascending=True)   # ascending → longest bar at top
)

# ── Color gradient (light salmon → deep red, proportional to count) ────────
norm = plt.Normalize(species_counts.min(), species_counts.max())
cmap = plt.cm.RdPu                             # alto = oscuro, bajo = claro
colors = [cmap(norm(v)) for v in species_counts.values]

# ── Plot ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 7))
fig.patch.set_facecolor("white")

bars = ax.barh(
    species_counts.index,
    species_counts.values,
    color=colors,
    edgecolor="white",
    linewidth=0.6,
    height=0.7,
)

# Value labels at the end of each bar
for bar, val in zip(bars, species_counts.values):
    ax.text(
        bar.get_width() + 0.5,
        bar.get_y() + bar.get_height() / 2,
        str(val),
        va="center",
        ha="left",
        fontsize=8,
        color="#333333",
    )

# ── Styling ────────────────────────────────────────────────────────────────
ax.set_xlabel("Count", fontsize=10, labelpad=8, color="#444444")
ax.set_title("Distriución de especies", fontsize=13, fontweight="bold",
             pad=14, color="#222222")

ax.set_xlim(0, species_counts.max() * 1.12)
ax.tick_params(axis="y", labelsize=8, colors="#333333")
ax.tick_params(axis="x", labelsize=9, colors="#555555")
ax.spines[["top", "right", "left"]].set_visible(False)
ax.spines["bottom"].set_color("#cccccc")
ax.xaxis.grid(True, linestyle="--", linewidth=0.5, color="#eeeeee", zorder=0)
ax.set_axisbelow(True)

plt.tight_layout()

# ── Save ───────────────────────────────────────────────────────────────────
OUT_PATH = os.path.join(BASE_PATH, "species_distribution.png")
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved → {OUT_PATH}")
plt.show()