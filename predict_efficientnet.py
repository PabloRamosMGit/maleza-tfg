import os
import joblib
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input

# ===============================
# CONFIG
# ===============================

BASE_PATH = r'C:\Users\pablo\pablo-tfg-malezas\maleza-tfg\csiro-biomass'
SAVE_DIR  = os.path.join(BASE_PATH, 'saved_model_efficientnet_simplified_attempt')
IMG_SIZE  = 300

# ===============================
# CARGAR ARTEFACTOS
# ===============================

print("Cargando modelos y artefactos...")
ensemble_models = joblib.load(os.path.join(SAVE_DIR, 'ensemble_models.pkl'))
TARGET_COLS     = joblib.load(os.path.join(SAVE_DIR, 'target_cols.pkl'))
scaler          = joblib.load(os.path.join(SAVE_DIR, 'scaler.pkl'))
print(f"✅ Modelos cargados. Targets: {TARGET_COLS}")

# ===============================
# CARGAR EfficientNetB3 (extractor)
# ===============================

base = EfficientNetB3(
    weights='imagenet',
    include_top=False,
    pooling='avg',
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base.trainable = False
print(f"✅ EfficientNetB3 cargado.")

# ===============================
# FUNCIONES
# ===============================

def extract_features(img_path: str) -> np.ndarray:
    """Carga una imagen y extrae el vector de features con EfficientNetB3."""
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    x   = img.astype(np.float32)
    x   = np.expand_dims(x, axis=0)
    x   = preprocess_input(x)
    return base.predict(x, verbose=0).flatten()


def predict_image(img_path: str) -> dict:
    """
    Recibe la ruta de una imagen y devuelve las predicciones de biomasa.
    Promedia las predicciones de todos los folds y modelos del ensemble.
    """
    # Extraer y escalar features
    features = extract_features(img_path)
    features_scaled = scaler.transform(features.reshape(1, -1))  # (1, 1536)

    result = {}

    for target in TARGET_COLS:
        fold_preds = []

        for fold_models in ensemble_models[target]:   # lista de folds
            fold_pred = 0.0
            for name, mdl in fold_models:              # lista de (name, model)
                fold_pred += mdl.predict(features_scaled)[0] / len(fold_models)
            fold_preds.append(fold_pred)

        # Promedio entre folds + clip a 0 (gramos no negativos)
        result[target] = max(0.0, float(np.mean(fold_preds)))

    return result

# ===============================
# EJEMPLO DE USO
# ===============================

img_path = r"C:\Users\pablo\pablo-tfg-malezas\maleza-tfg\csiro-biomass\train\ID95050718.jpg"
resultado = predict_image(img_path)

print("\n📊 Predicciones de biomasa:")
for target, value in resultado.items():
    print(f"  {target}: {value:.4f} g")
