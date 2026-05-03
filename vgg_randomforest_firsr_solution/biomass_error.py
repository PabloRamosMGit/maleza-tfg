# ============================================================
# BIOMASS_ERROR.PY
# Calcula el error de biomasa absoluto sobre 5 imágenes aleatorias
# y expone un módulo para obtener parámetros de cualquier imagen.
#
# Error de biomasa = abs(Dry_Total_g_real - Dry_Total_g_pred)
#
# Dependencias: joblib, numpy, pandas, tensorflow, albumentations
# ============================================================

import os
import random
import datetime
import numpy as np
import pandas as pd
import joblib

import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image as keras_image


# ============================================================
# RUTAS  (mismas que solution_one.py)
# ============================================================

BASE_PATH    = r"C:\Users\pablo\maleza-tfg\csiro-biomass"
TRAIN_CSV    = os.path.join(BASE_PATH, "train.csv")
SAVE_DIR     = "saved_model_vgg"
MODEL_PATH   = os.path.join(SAVE_DIR, "random_forest.pkl")
TARGETS_PATH = os.path.join(SAVE_DIR, "target_cols.pkl")
RESULTS_TXT  = "biomass_error_results.txt"

IMG_SIZE     = 224
RANDOM_SEED  = 999


# ============================================================
# CARGA DE ARTEFACTOS  (modelo + VGG16)
# ============================================================

def _load_vgg16_extractor():
    """Carga VGG16 preentrenado y devuelve el extractor (hasta fc2)."""
    full_model = VGG16(
        weights="imagenet",
        include_top=True,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    extractor = tf.keras.Model(
        inputs=full_model.input,
        outputs=full_model.layers[-2].output   # capa fc2 → 4096-dim
    )
    extractor.trainable = False
    return extractor


print("Cargando modelo y extractor VGG16...")
_vgg_extractor  = _load_vgg16_extractor()
_rf_model       = joblib.load(MODEL_PATH)
_target_cols    = joblib.load(TARGETS_PATH)   # lista con los 5 nombres de target
print("Artefactos cargados correctamente.\n")


# ============================================================
# MÓDULO: predict_image_params
# ============================================================

def predict_image_params(img_path: str) -> dict:
    """
    Dado el path absoluto a una imagen, extrae características con VGG16
    y devuelve un diccionario con las predicciones de todos los targets.

    Parámetros
    ----------
    img_path : str
        Ruta completa a la imagen (.jpg, .png, etc.).

    Retorna
    -------
    dict  con claves = nombres de target y valores = predicciones (g/m²).
          Ejemplo:
          {
              'Dry_Clover_g': 2.34,
              'Dry_Dead_g'  : 15.7,
              'Dry_Green_g' : 8.9,
              'Dry_Total_g' : 48.2,
              'GDM_g'       : 31.1,
          }
    """
    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"Imagen no encontrada: {img_path}")

    # --- Preprocesamiento ---
    img       = keras_image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    arr       = keras_image.img_to_array(img).astype(np.float32)
    arr       = np.expand_dims(arr, axis=0)
    arr       = preprocess_input(arr)

    # --- Extracción de características (4096-dim) ---
    features  = _vgg_extractor.predict(arr, verbose=0).flatten().reshape(1, -1)

    # --- Predicción con RandomForest ---
    preds     = _rf_model.predict(features)[0]          # array de 5 valores
    preds     = np.clip(preds, 0, None)                 # sin valores negativos

    return dict(zip(_target_cols, preds))


# ============================================================
# EVALUACIÓN DE ERROR SOBRE 5 IMÁGENES ALEATORIAS
# ============================================================

def evaluate_random_images(n: int = 5, seed: int = RANDOM_SEED) -> list[dict]:
    """
    Selecciona `n` imágenes al azar del CSV de entrenamiento,
    calcula el error de biomasa absoluto y escribe los resultados en TXT.

    Error de biomasa = abs(Dry_Total_g_real - Dry_Total_g_pred)

    Parámetros
    ----------
    n    : número de imágenes a evaluar.
    seed : semilla para reproducibilidad.

    Retorna
    -------
    Lista de dicts con los resultados por imagen.
    """
    # --- Cargar CSV y pivotar a formato ancho ---
    train_df      = pd.read_csv(TRAIN_CSV)
    train_wide    = train_df.pivot(
        index="image_path",
        columns="target_name",
        values="target"
    ).reset_index()

    # --- Seleccionar n imágenes aleatorias ---
    random.seed(seed)
    sample = train_wide.sample(n=n, random_state=seed).reset_index(drop=True)

    results = []

    print("=" * 62)
    print(f"EVALUANDO {n} IMÁGENES ALEATORIAS")
    print("=" * 62)

    for idx, row in sample.iterrows():
        img_rel_path = row["image_path"]
        img_full_path = os.path.join(BASE_PATH, img_rel_path)

        # Valor real de biomasa total
        real_total = float(row["Dry_Total_g"])

        # Predicciones del modelo
        params = predict_image_params(img_full_path)
        pred_total = params["Dry_Total_g"]

        # Error absoluto de biomasa
        biomass_error = abs(real_total - pred_total)

        entry = {
            "imagen"          : img_rel_path,
            "Dry_Total_g_real": real_total,
            "Dry_Total_g_pred": round(pred_total, 4),
            "error_biomasa"   : round(biomass_error, 4),
            "todos_params"    : {k: round(v, 4) for k, v in params.items()},
        }
        results.append(entry)

        print(f"\n  Imagen {idx + 1}: {img_rel_path}")
        print(f"    Real        : {real_total:.4f} g/m²")
        print(f"    Predicción  : {pred_total:.4f} g/m²")
        print(f"    Error abs.  : {biomass_error:.4f} g/m²")
        print(f"    Todos params: {entry['todos_params']}")

    print("\n" + "=" * 62)
    mean_err = np.mean([r["error_biomasa"] for r in results])
    print(f"  Error medio (5 imágenes): {mean_err:.4f} g/m²")
    print("=" * 62 + "\n")

    # --- Escribir resultados en TXT ---
    _write_results_txt(results, mean_err)

    return results


def _write_results_txt(results: list[dict], mean_error: float):
    """Anota los resultados de evaluación en un archivo de texto plano."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "=" * 62,
        "RESULTADOS DE ERROR DE BIOMASA",
        f"Generado el: {timestamp}",
        "=" * 62,
        "",
    ]

    for i, r in enumerate(results, start=1):
        lines += [
            f"Imagen {i}: {r['imagen']}",
            f"  Dry_Total_g real       : {r['Dry_Total_g_real']:.4f} g/m²",
            f"  Dry_Total_g predicción : {r['Dry_Total_g_pred']:.4f} g/m²",
            f"  Error de biomasa (abs) : {r['error_biomasa']:.4f} g/m²",
            "  Todos los parámetros predichos:",
        ]
        for target, val in r["todos_params"].items():
            lines.append(f"    {target:<20}: {val:.4f} g/m²")
        lines.append("")

    lines += [
        "-" * 62,
        f"Error medio de biomasa (5 imágenes): {mean_error:.4f} g/m²",
        "=" * 62,
    ]

    with open(RESULTS_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Resultados guardados en: {RESULTS_TXT}")


# ============================================================
# USO DIRECTO COMO SCRIPT
# ============================================================

if __name__ == "__main__":

    # --- Ejemplo 1: evaluar 5 imágenes aleatorias ---
    resultados = evaluate_random_images(n=5)

    # --- Ejemplo 2: obtener parámetros de una imagen individual ---
    # Descomentar y ajustar la ruta para probarlo:
    #
    # ruta = r"C:\Users\pablo\maleza-tfg\csiro-biomass\images\img_001.jpg"
    # params = predict_image_params(ruta)
    # print("\nParámetros predichos:")
    # for k, v in params.items():
    #     print(f"  {k:<20}: {v:.4f} g/m²")