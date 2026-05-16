import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_PATH = r"C:\Users\pablo\maleza-tfg\csiro-biomass"
TRAIN_CSV_PATH = os.path.join(BASE_PATH, "train.csv")

# ── Load ───────────────────────────────────────────────────────────────────
df = pd.read_csv(TRAIN_CSV_PATH)
vals = df[df["target_name"] == "Dry_Total_g"]["target"].dropna()
vals = vals[vals >= 0].values

q1, q2, q3 = np.percentile(vals, [25, 50, 75])
mean_v      = vals.mean()

clip   = np.percentile(vals, 99)
v_clip = vals[vals <= clip]

# ── KDE ────────────────────────────────────────────────────────────────────
x_lin   = np.linspace(0, clip, 600)
kde_lin = gaussian_kde(v_clip, bw_method=0.18)
y_lin   = kde_lin(x_lin)
norm_y  = y_lin / y_lin.max()

# ── Colores (tema claro) ───────────────────────────────────────────────────
BG    = "#ffffff"
PANEL = "#f7f7fb"
C1    = "#e07b54"
CMED  = "#c8a000"
CMEAN = "#22cc3e"
DARK  = "#1a1a2e"
MUTED = "#555577"

# ── Figura ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6), facecolor=BG)
ax.set_facecolor(PANEL)
for sp in ax.spines.values():
    sp.set_color("#ccccdd")

# Histograma
ax.hist(v_clip, bins=70, density=True,
        color="#dcdcec", edgecolor="#c0c0d8", linewidth=0.3, zorder=1)

# KDE con relleno degradado
for i in range(len(x_lin) - 1):
    alpha = 0.30 + 0.60 * norm_y[i]
    ax.fill_between(x_lin[i:i+2], y_lin[i:i+2],
                    alpha=alpha, color=C1, zorder=2)

ax.plot(x_lin, y_lin, color=C1, lw=2, zorder=3)

# Líneas de mediana y media
ax.axvline(q2,     color=CMED,  lw=1.6, ls="--", zorder=4, label=f"Mediana  {q2:.1f} g")
ax.axvline(mean_v, color=CMEAN, lw=1.6, ls=":",  zorder=4, label=f"Media    {mean_v:.1f} g")

# ── Anotaciones con separación dinámica ────────────────────────────────────
lines = [
    (q2,     CMED,  f"Mediana\n{q2:.1f} g"),
    (mean_v, CMEAN, f"Media\n{mean_v:.1f} g"),
]

# Umbral: si las líneas están muy juntas, escalonar verticalmente
PROX_UMBRAL = clip * 0.08          # 8 % del rango total
y_base       = y_lin.max()
y_positions  = [y_base * 0.96, y_base * 0.96]

if abs(q2 - mean_v) < PROX_UMBRAL:
    # La línea más a la derecha baja para no tapar a la otra
    if q2 < mean_v:
        y_positions[1] = y_base * 0.70   # media baja
    else:
        y_positions[0] = y_base * 0.70   # mediana baja

for (val, col, lbl), y_pos in zip(lines, y_positions):
    x_offset = clip * 0.012
    ha = "left"

    # Si la etiqueta se sale por la derecha, la ponemos a la izquierda
    if val + x_offset > clip * 0.88:
        x_offset = -clip * 0.012
        ha = "right"

    ax.text(val + x_offset, y_pos, lbl,
            color=col, fontsize=8, va="top", ha=ha,
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=col,
                      alpha=0.75, lw=0.8),   # caja de fondo para legibilidad
            zorder=5)

# Leyenda
ax.legend(loc="upper right", fontsize=8,
          facecolor="#efeffa", edgecolor="#ccccdd",
          labelcolor=DARK, framealpha=0.9)

# Etiquetas y título
ax.set_title("Distribución de Biomasa Seca Total",
             color=DARK, fontsize=13, fontweight="bold", pad=14,
             fontfamily="monospace")
ax.set_xlabel("Peso seco total (g)", color=MUTED, fontsize=9, labelpad=8)
ax.set_ylabel("Densidad",            color=MUTED, fontsize=9, labelpad=8)
ax.tick_params(colors=MUTED, labelsize=8)
ax.set_xlim(0, clip)

plt.tight_layout()

# ── Guardar como PDF ───────────────────────────────────────────────────────
OUT = os.path.join(BASE_PATH, "biomass_distribution.pdf")
plt.savefig(OUT, format="pdf", bbox_inches="tight", facecolor=BG)
print(f"Guardado → {OUT}")
plt.show()