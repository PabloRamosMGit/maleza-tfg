import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image

# ===============================
# CONFIG
# ===============================

BASE_PATH = r'C:\Users\pablo\maleza-tfg\csiro-biomass'
SAVE_DIR = os.path.join(BASE_PATH, 'saved_model_vgg')
IMG_SIZE = 224

# ===============================
# CARGAR MODELO Y TARGETS
# ===============================

print("Cargando modelo...")
model = joblib.load(os.path.join(SAVE_DIR, 'random_forest.pkl'))
TARGET_COLS = joblib.load(os.path.join(SAVE_DIR, 'target_cols.pkl'))
print(f"✅ Modelo cargado. Targets: {TARGET_COLS}")

# ===============================
# CARGAR VGG16 (extractor)
# ===============================

full_model = VGG16(weights="imagenet", include_top=True, input_shape=(224, 224, 3))
base_model = tf.keras.Model(
    inputs=full_model.input,
    outputs=full_model.layers[-2].output
)
base_model.trainable = False

# ===============================
# FUNCIÓN DE PREDICCIÓN
# ===============================

def extract_features(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = base_model.predict(img_array, verbose=0)
    return features.flatten()

def predict_image(img_path):
    """Recibe la ruta de una imagen y devuelve las predicciones."""
    features = extract_features(img_path)
    features = features.reshape(1, -1)  # shape (1, 4096)
    prediction = model.predict(features)[0]
    result = dict(zip(TARGET_COLS, prediction))
    return result

# ===============================
# USAR CON IMÁGENES NUEVAS
# ===============================

# Cambia esto por la ruta de tu imagen nueva
img_path = r'C:\Users\pablo\maleza-tfg\csiro-biomass\test\ID1001187975.jpg'

resultado = predict_image(img_path)

print("\n📊 Predicciones:")
for target, value in resultado.items():
    print(f"  {target}: {value:.4f} g")