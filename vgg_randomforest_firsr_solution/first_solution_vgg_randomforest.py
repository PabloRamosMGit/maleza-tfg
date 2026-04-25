# ============================================================
# SOLUCIÓN 1 - VGG16 + RandomForestRegressor (LOCAL)
# Incluye: Preprocesamiento, Aumento de Datos,
#          Extracción de Características, Regresión,
#          Validación y Calibración
# ============================================================
# NOTAS IMPORTANTES:
# Esta es una versión local que se ejecuta en CPU. Su equivalente para
# Kaggle está disponible en:
# https://www.kaggle.com/code/pabloramosmadrigal/solution-one-vg116
#
# SOBRE LAS MÉTRICAS DE DESEMPEÑO:
# El score de Kaggle no es directamente comparable con el R² calculado
# aquí debido a diferencias en la metodología de evaluación. El R²
# presentado en este archivo se calcula localmente sobre un conjunto de
# validación con distribución conocida. En contraste, el score de Kaggle
# se basa en un conjunto de prueba privado cuya distribución puede diferir
# significativamente, lo que invalida cualquier comparación objetiva entre
# ambas métricas.
# ============================================================
# ============================================================

# Para manipulación de datos
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# Para guardar el modelo
import joblib

# Para métricas de validación
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Para el modelo de aprendizaje automático
from sklearn.ensemble import RandomForestRegressor

# Para visión por computador
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image

# Para el aumento de datos
import albumentations as A


# ===============================
# RUTAS
# ===============================

BASE_PATH = r"C:\Users\pablo\maleza-tfg\csiro-biomass"   # carpeta raíz del dataset local

TRAIN_CSV_PATH = os.path.join(BASE_PATH, "train.csv")
TEST_CSV_PATH  = os.path.join(BASE_PATH, "test.csv")

TRAIN_IMG_DIR  = BASE_PATH
TEST_IMG_DIR   = BASE_PATH

SAVE_DIR = "saved_model_vgg"   # carpeta donde se guardan el modelo y artefactos
os.makedirs(SAVE_DIR, exist_ok=True)


# ===============================
# CARGA DE DATOS
# ===============================

train_df = pd.read_csv(TRAIN_CSV_PATH)
test_df  = pd.read_csv(TEST_CSV_PATH)

# Convertir de formato largo a ancho para tener una fila por imagen
# Cada columna corresponde a una variable objetivo (Dry_Clover_g, Dry_Dead_g, etc.)
train_wide_df = train_df.pivot(
    index="image_path",
    columns="target_name",
    values="target"
).reset_index()

print(train_wide_df)


# ===============================
# CONFIGURACIÓN GENERAL
# ===============================

IMG_SIZE    = 224       # Tamaño de entrada esperado por VGG16
RANDOM_SEED = 999       # Semilla para reproducibilidad
VAL_SIZE    = 0.2       # 20 % de los datos se reservan para validación


# ===============================
# MÓDULO DE AUMENTO DE DATOS
# ===============================
# Se definen transformaciones geométricas y de color sencillas que se
# aplican aleatoriamente a cada imagen SOLO durante el entrenamiento.
# El objetivo es aumentar artificialmente la variabilidad del conjunto
# de entrenamiento para que el modelo generalice mejor.

def get_train_transforms():
    """
    Devuelve un pipeline de albumentations con transformaciones simples:
      - HorizontalFlip : volteo horizontal con probabilidad 0.5
      - VerticalFlip   : volteo vertical con probabilidad 0.5
      - RandomRotate90 : rotación aleatoria de 90° con probabilidad 0.5
      - RandomBrightnessContrast : variación leve de brillo y contraste
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        A.Resize(IMG_SIZE, IMG_SIZE),
    ], seed=RANDOM_SEED)


# Instanciar el pipeline de aumento una sola vez
train_transforms = get_train_transforms()


# ===============================
# CARGA DEL MODELO VGG16
# ===============================
# Se carga VGG16 con pesos preentrenados en ImageNet directamente desde Keras.
# include_top=True incluye las capas densas fc1 y fc2, por lo que el vector
# de características tendrá 4096 dimensiones (salida de la capa fc2).

full_model = VGG16(
    weights="imagenet",
    include_top=True,
    input_shape=(224, 224, 3)
)

# layers[-2] corresponde a fc2, la última capa densa antes de la clasificación.
# Esto produce un vector de 4096 dimensiones por imagen.
base_model = tf.keras.Model(
    inputs=full_model.input,
    outputs=full_model.layers[-2].output
)
base_model.trainable = False   # Los pesos de VGG16 NO se actualizan

print(f"Model loaded. Feature vector output shape: {base_model.output_shape}")
# Debería imprimir: (None, 4096)


# ===============================
# FUNCIÓN DE EXTRACCIÓN DE CARACTERÍSTICAS
# ===============================

def extract_features(img_path, augment=False):
    """
    Carga una imagen desde disco, aplica (opcionalmente) aumento de datos
    y extrae el vector de características con VGG16.

    Parámetros
    ----------
    img_path : str
        Ruta completa a la imagen.
    augment : bool
        Si es True, aplica las transformaciones de aumento antes
        de pasar la imagen por el modelo.

    Retorna
    -------
    numpy.ndarray
        Vector 1-D de 4096 dimensiones con las características extraídas.
    """
    # --- Preprocesamiento base ---
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img).astype(np.uint8)   # albumentations espera uint8

    # --- Aumento de datos (solo en entrenamiento) ---
    if augment:
        augmented = train_transforms(image=img_array)
        img_array = augmented["image"]

    # Añadir dimensión de batch y aplicar preprocesamiento específico de VGG16
    img_array = np.expand_dims(img_array.astype(np.float32), axis=0)
    img_array = preprocess_input(img_array)

    # Extraer características y aplanar a un vector 1-D de 4096 dimensiones
    features = base_model.predict(img_array, verbose=0)
    return features.flatten()


# ===============================
# DIVISIÓN ENTRENAMIENTO / VALIDACIÓN
# ===============================
# Se separa el 20 % de las imágenes como conjunto de validación.
# Esto permite evaluar el rendimiento del modelo en datos no vistos
# durante el entrenamiento, lo que da una estimación honesta del error.

train_split_df, val_split_df = train_test_split(
    train_wide_df,
    test_size=VAL_SIZE,
    random_state=RANDOM_SEED
)

print(f"Imágenes de entrenamiento : {len(train_split_df)}")
print(f"Imágenes de validación    : {len(val_split_df)}")


# ===============================
# EXTRACCIÓN DE CARACTERÍSTICAS - ENTRENAMIENTO
# ===============================
# Se extrae el vector de características para cada imagen de entrenamiento.
# El parámetro augment=True activa las transformaciones definidas anteriormente.

train_features = []
print("Extrayendo características del conjunto de ENTRENAMIENTO...")

for img_path in tqdm(train_split_df["image_path"]):
    full_path = os.path.join(TRAIN_IMG_DIR, img_path)
    features  = extract_features(full_path, augment=True)
    train_features.append(features)

train_features = np.array(train_features)
print("Matriz de características de entrenamiento:", train_features.shape)


# ===============================
# EXTRACCIÓN DE CARACTERÍSTICAS - VALIDACIÓN
# ===============================
# Para validación NO se aplica aumento de datos: queremos evaluar el modelo
# en imágenes tal cual llegan, sin modificaciones artificiales.

val_features = []
print("Extrayendo características del conjunto de VALIDACIÓN...")

for img_path in tqdm(val_split_df["image_path"]):
    full_path = os.path.join(TRAIN_IMG_DIR, img_path)
    features  = extract_features(full_path, augment=False)
    val_features.append(features)

val_features = np.array(val_features)
print("Matriz de características de validación:", val_features.shape)


# ===============================
# ENTRENAMIENTO DEL MODELO
# ===============================

TARGET_COLS = train_wide_df.columns[1:]   # todas las columnas excepto image_path

X_train = train_features
y_train = train_split_df[TARGET_COLS].values   # shape: (285, 5)

X_val   = val_features
y_val   = val_split_df[TARGET_COLS].values     # shape: (72, 5)

print(f"X_train: {X_train.shape}  |  y_train: {y_train.shape}")
print("Entrenando RandomForestRegressor...")

# sklearn soporta múltiples variables objetivo de forma nativa,
# por lo que se entrena un único modelo para todas las columnas target.
model = RandomForestRegressor(
    n_estimators=300,
    random_state=RANDOM_SEED,
    n_jobs=-1,   # usa todos los núcleos CPU disponibles
    verbose=2
)

model.fit(X_train, y_train)
print("Entrenamiento completado.")


# ============================================================
# MÓDULO DE VALIDACIÓN Y CALIBRACIÓN
# ============================================================
# En esta sección se evalúa el rendimiento del modelo comparando
# las predicciones generadas contra los valores reales de biomasa
# del conjunto de validación.
#
# Métricas utilizadas:
#   - MSE  (Mean Squared Error / Error Cuadrático Medio):
#       Promedio de los errores al cuadrado. Penaliza fuertemente
#       los errores grandes. Unidad: (g/m²)²
#
#   - RMSE (Root Mean Squared Error / Raíz del ECM):
#       Raíz cuadrada del MSE. Está en la misma unidad que la variable
#       objetivo (g/m²), por lo que es más interpretable que el MSE.
#
#   - R²  (Coeficiente de Determinación):
#       Indica qué proporción de la varianza de los datos es explicada
#       por el modelo. Va de 0 a 1; un valor cercano a 1 indica un
#       mejor ajuste.
# ============================================================

print("\n" + "=" * 60)
print("MÓDULO DE VALIDACIÓN Y CALIBRACIÓN")
print("=" * 60)

results = []

# Generar predicciones sobre el conjunto de validación
# y_pred tiene shape (72, 5): una fila por imagen, una columna por target
y_pred = model.predict(X_val)

for i, col in enumerate(TARGET_COLS):

    # Valores reales y predichos para esta variable objetivo
    y_true_col = y_val[:, i]
    y_pred_col = y_pred[:, i]

    # --- MSE: promedio de (real - predicho)² ---
    mse  = mean_squared_error(y_true_col, y_pred_col)

    # --- RMSE: raíz del MSE, misma unidad que la biomasa (g/m²) ---
    rmse = np.sqrt(mse)

    # --- R²: proporción de varianza explicada por el modelo ---
    r2   = r2_score(y_true_col, y_pred_col)

    results.append({"Variable objetivo": col, "MSE": mse, "RMSE": rmse, "R²": r2})

    print(f"  {col:<20}  MSE={mse:>10.4f}  RMSE={rmse:>8.4f} g/m²  R²={r2:>6.4f}")

# Promedio global para tener una visión general del modelo
results_df = pd.DataFrame(results)
mean_mse  = results_df["MSE"].mean()
mean_rmse = results_df["RMSE"].mean()
mean_r2   = results_df["R²"].mean()

print("-" * 60)
print(f"  {'PROMEDIO GLOBAL':<20}  MSE={mean_mse:>10.4f}  RMSE={mean_rmse:>8.4f} g/m²  R²={mean_r2:>6.4f}")
print("=" * 60 + "\n")


# =============================== TEST ===============================

# ===============================
# EXTRACCIÓN DE CARACTERÍSTICAS - TEST
# ===============================
# Al igual que en validación, NO se aplica aumento de datos en test.

test_features = []
unique_image_paths = test_df["image_path"].unique()

print("Extrayendo características del conjunto de TEST...")

for img_path in tqdm(unique_image_paths):
    full_path = os.path.join(TEST_IMG_DIR, img_path)
    features  = extract_features(full_path, augment=False)
    test_features.append(features)

test_features = np.array(test_features)


# ===============================
# PREDICCIÓN Y ENVÍO
# ===============================

print("Generando predicciones finales...")

# Un único modelo predice las 5 variables objetivo simultáneamente
# test_predictions tiene shape (n_test, 5)
test_predictions = model.predict(test_features)

# Armar DataFrame ancho con las predicciones
predictions_df = pd.DataFrame(test_predictions, columns=TARGET_COLS)
predictions_df.insert(0, "image_path", unique_image_paths)

# Convertir de formato ancho a largo (long format requerido para la entrega)
predictions_long_df = predictions_df.melt(
    id_vars=["image_path"],
    value_vars=list(TARGET_COLS),
    var_name="target_name",
    value_name="target"
)

# Unir con test_df para recuperar el sample_id original
submission_df = pd.merge(
    test_df[["sample_id", "image_path", "target_name"]],
    predictions_long_df,
    on=["image_path", "target_name"]
)[["sample_id", "target"]].dropna()

submission_df.to_csv("submission_vgg_randomforest.csv", index=False)
print("Archivo de entrega creado: submission_vgg_randomforest.csv")
print(submission_df.head())


# ===============================
# GUARDAR MODELO Y ARTEFACTOS
# ===============================

# Guardar el modelo único que predice todas las variables objetivo
joblib.dump(model, os.path.join(SAVE_DIR, "random_forest.pkl"))

# Guardar los nombres de los targets (para reconstruir el submission en inferencia)
joblib.dump(list(TARGET_COLS), os.path.join(SAVE_DIR, "target_cols.pkl"))

# Guardar los features de entrenamiento (opcional, para análisis futuros)
np.save(os.path.join(SAVE_DIR, "train_features.npy"), train_features)

print(f"Modelo y artefactos guardados en: {SAVE_DIR}")