import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# ── Ruta ───────────────────────────────────────────────────────────────────
TRAIN_PATH = r"C:\Users\pablo\maleza-tfg\csiro-biomass\train"

# ── Selección aleatoria ────────────────────────────────────────────────────
todas = [f for f in os.listdir(TRAIN_PATH) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
seleccion = random.sample(todas, 5)

# ── Colores (tema claro) ───────────────────────────────────────────────────
BG    = "#ffffff"
DARK  = "#1a1a2e"
MUTED = "#555577"

# ── Figura ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 5, figsize=(18, 5), facecolor=BG)
fig.subplots_adjust(left=0.02, right=0.98, top=0.82, bottom=0.05, wspace=0.08)

for ax, nombre in zip(axes, seleccion):
    img = mpimg.imread(os.path.join(TRAIN_PATH, nombre))
    ax.imshow(img)
    ax.axis("off")
    ax.set_facecolor(BG)

    # Borde sutil gris
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor("#ccccdd")
        sp.set_linewidth(1.2)

    # ID de muestra debajo de cada imagen
    sample_id = os.path.splitext(nombre)[0]
    ax.set_title(sample_id, color=MUTED, fontsize=7.5,
                 fontfamily="monospace", pad=5)

# Título principal
fig.text(0.5, 0.95,
         "Muestras Aleatorias del Conjunto de Entrenamiento",
         ha="center", va="top", color=DARK,
         fontsize=13, fontweight="bold", fontfamily="monospace")

fig.text(0.5, 0.88,
         f"{len(todas):,} imágenes totales  ·  5 seleccionadas al azar",
         ha="center", va="top", color=MUTED,
         fontsize=8, fontfamily="monospace")

# ── Guardar como PDF ───────────────────────────────────────────────────────
BASE_PATH = r"C:\Users\pablo\maleza-tfg\csiro-biomass"
OUT = os.path.join(BASE_PATH, "muestras_aleatorias.pdf")
plt.savefig(OUT, format="pdf", bbox_inches="tight", facecolor=BG)
print(f"Guardado → {OUT}")
plt.show()