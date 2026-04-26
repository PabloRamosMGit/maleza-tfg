# ============================================================
# SOLUCIÓN 2 - EfficientNetB3 + Ridge Regression
# Incluye: Preprocesamiento, Aumento de Datos,
#          Extracción de Características, Regresión,
#          Validación y Calibración
#Nota: Esta solucion se ejecuto en Kaggle con GPU, este es el link:
# https://www.kaggle.com/code/pabloramosmadrigal/b3-ridge-second-solution
# ============================================================

# Para manipulación de datos
import os
import joblib
import pandas as pd
import numpy as np
from tqdm import tqdm

# Para el aumento de datos
import albumentations as A
import cv2

# Para el modelo de aprendizaje automático
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Para visión por computador
import tensorflow as tf
import keras_cv


# ===============================
# RUTAS
# ===============================

BASE_PATH = "/kaggle/input/competitions/csiro-biomass"   # carpeta raíz del dataset local

TRAIN_CSV_PATH = os.path.join(BASE_PATH, "train.csv")
TEST_CSV_PATH  = os.path.join(BASE_PATH, "test.csv")

TRAIN_IMG_DIR  = BASE_PATH
TEST_IMG_DIR   = BASE_PATH

SAVE_DIR = "saved_model_efficientnet"
os.makedirs(SAVE_DIR, exist_ok=True)


# ===============================
# CONFIGURACIÓN GENERAL
# ===============================

IMG_SIZE    = 300       # Tamaño de entrada esperado por EfficientNetB3
RANDOM_SEED = 999       # Semilla para reproducibilidad
VAL_SIZE    = 0.2       # 20 % de los datos se reservan para validación

BATCH_SIZE      = 32    # Número de imágenes procesadas simultáneamente en GPU
# Nota: NO se aplica augmentación ni en entrenamiento ni en test para Ridge.
# La augmentación provoca data leakage en LOO-CV y no aporta beneficio real
# a un modelo lineal con extractor de características congelado.

# Alpha controla la fuerza de regularización L2 de Ridge.
# Un valor alto penaliza más los coeficientes grandes → modelo más simple.
# Se usa 1.0 que es el valor por defecto recomendado por sklearn como
# punto de partida razonable sin necesidad de búsqueda exhaustiva.
ALPHA = 1000


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

TARGET_COLS = [c for c in train_wide_df.columns if c != "image_path"]

print("Targets:", TARGET_COLS)
print(f"Imágenes de train: {len(train_wide_df)}")
print(f"Imágenes de test:  {len(test_df['image_path'].unique())}")


# ===============================
# MÓDULO DE AUMENTO DE DATOS
# ===============================
# Esta etapa no ayuda a mejorar el score usando Ridge

# def get_train_transforms():
#     """
#     Devuelve un pipeline de albumentations con transformaciones simples:
#       - HorizontalFlip : volteo horizontal con probabilidad 0.5
#       - VerticalFlip   : volteo vertical con probabilidad 0.5
#       - RandomRotate90 : rotación aleatoria de 90° con probabilidad 0.5
#       - RandomBrightnessContrast : variación leve de brillo y contraste
#     """
#     return A.Compose([
#         A.HorizontalFlip(p=0.5),
#         A.VerticalFlip(p=0.5),
#         A.RandomRotate90(p=0.5),
#         A.RandomBrightnessContrast(
#             brightness_limit=0.2,
#             contrast_limit=0.2,
#             p=0.5
#         ),
#         A.Resize(IMG_SIZE, IMG_SIZE),
#     ], seed=RANDOM_SEED)


# # Instanciar el pipeline de aumento una sola vez
# train_augment = get_train_transforms()


# ===============================
# CARGA DEL MODELO EfficientNetV2B3
# ===============================
# Se carga EfficientNetV2B3 usando keras_cv con pesos preentrenados en ImageNet.
# from_preset descarga y construye automáticamente la arquitectura completa,
# incluyendo el rescaling interno (include_rescaling=True por defecto), por lo
# que las imágenes deben pasarse en rango [0, 255] y NO dividirse entre 255.
#
# La salida del backbone es un mapa de características espacial (H×W×C).
# GlobalAveragePooling2D promedia ese mapa sobre las dimensiones espaciales
# y produce un vector 1-D de 1536 dimensiones por imagen, que es lo que
# se pasa como entrada al modelo de Ridge.

base_net = keras_cv.models.EfficientNetV2Backbone.from_preset("efficientnetv2_b3")
base_net.trainable = False   # Los pesos preentrenados NO se actualizan

# Construir el extractor: backbone + pooling global
inputs  = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x       = base_net(inputs, training=False)
outputs = tf.keras.layers.GlobalAveragePooling2D()(x)

feature_extractor = tf.keras.Model(inputs=inputs, outputs=outputs)

print(f"Modelo cargado. Forma del vector de características: {feature_extractor.output_shape}")
# Debería imprimir: (None, 1536)


# ===============================
# FUNCIONES DE CARGA Y EXTRACCIÓN
# ===============================

def load_image_as_array(img_path):
    """
    Carga una imagen desde disco con OpenCV y la convierte a RGB.
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return img


def extract_features_batch(img_arrays):
    """
    Recibe una lista de arrays HxWx3 uint8 y extrae sus vectores de
    características en un único forward pass por la GPU.

    Procesar en batches en lugar de imagen por imagen es clave para
    aprovechar el paralelismo de la GPU: en lugar de hacer N llamadas
    con batch_size=1, se hace una sola llamada con batch_size=N,
    lo que reduce drásticamente el tiempo de inferencia.

    Parámetros
    ----------
    img_arrays : list of np.ndarray
        Lista de imágenes en formato HxWx3 uint8.

    Retorna
    -------
    numpy.ndarray
        Matriz de shape (N, 1536) con un vector de características por imagen.
    """
    # Apilar todas las imágenes en un único tensor (N, 300, 300, 3)
    # keras_cv con from_preset maneja el rescaling internamente,
    # por lo tanto las imágenes se pasan en rango [0, 255] sin dividir entre 255.
    batch = np.stack([img.astype(np.float32) for img in img_arrays], axis=0)
    features = feature_extractor(batch, training=False)
    return features.numpy()   # shape: (N, 1536)


# ===============================
# DIVISIÓN ENTRENAMIENTO / VALIDACIÓN
# ===============================
# Se separa el 20 % de las imágenes como conjunto de validación ANTES
# de extraer características, para evitar cualquier fuga de información.

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
# Se extrae un único vector de características por imagen original, SIN augmentación.
#
# ¿Por qué no se augmenta en entrenamiento para Ridge?
#   Aplicar augmentación genera múltiples copias de cada imagen con la misma
#   etiqueta. Cuando LOO-CV deja fuera una muestra para validarla, todas sus
#   copias aumentadas siguen en el entrenamiento, provocando data leakage:
#   el modelo "ya conoce" esa imagen en otras formas, por lo que LOO-CV
#   selecciona un alpha artificialmente bajo (poca regularización) y el modelo
#   no generaliza bien a imágenes completamente nuevas.
#   La augmentación tiene sentido para entrenar redes end-to-end, no para
#   un extractor congelado + modelo lineal como este.

print("\nExtrayendo features de TRAIN...")

train_imgs, train_labels = [], []

for _, row in tqdm(train_split_df.iterrows(), total=len(train_split_df)):
    full_path = os.path.join(TRAIN_IMG_DIR, row["image_path"])
    img_array = load_image_as_array(full_path)
    train_imgs.append(img_array)
    train_labels.append(row[TARGET_COLS].values.astype(np.float32))

# Procesar en batches para aprovechar la GPU
print(f"Procesando {len(train_imgs)} imágenes en batches de {BATCH_SIZE}...")
train_features = []

for i in tqdm(range(0, len(train_imgs), BATCH_SIZE)):
    batch_feats = extract_features_batch(train_imgs[i : i + BATCH_SIZE])
    train_features.append(batch_feats)

train_features = np.vstack(train_features)   # shape: (285, 1536)
train_labels   = np.array(train_labels)

print(f"Forma del vector de características: {train_features.shape}")


# ===============================
# EXTRACCIÓN DE CARACTERÍSTICAS - VALIDACIÓN
# ===============================
# Para validación NO se aplica aumento de datos: se evalúa el modelo
# en imágenes tal cual llegan, sin modificaciones artificiales.

print("\nExtrayendo features del conjunto de VALIDACIÓN...")

val_imgs, val_labels = [], []

for _, row in tqdm(val_split_df.iterrows(), total=len(val_split_df)):
    full_path = os.path.join(TRAIN_IMG_DIR, row["image_path"])
    img_array = load_image_as_array(full_path)
    label     = row[TARGET_COLS].values.astype(np.float32)

    val_imgs.append(img_array)
    val_labels.append(label)

# Procesar validación en batches (sin aumento de datos)
val_features = []
for i in tqdm(range(0, len(val_imgs), BATCH_SIZE)):
    batch_feats = extract_features_batch(val_imgs[i : i + BATCH_SIZE])
    val_features.append(batch_feats)

val_features = np.vstack(val_features)
val_labels   = np.array(val_labels)

print(f"Matriz de características de validación: {val_features.shape}")


# ===============================
# NORMALIZACIÓN
# ===============================
# Ridge Regression es sensible a la escala de los datos.
# StandardScaler estandariza cada característica a media=0 y desviación=1.
# IMPORTANTE: el scaler se ajusta SOLO con los datos de entrenamiento y
# luego se aplica a validación y test, nunca al revés.

scaler  = StandardScaler()
X_train = scaler.fit_transform(train_features)   # ajusta y transforma
X_val   = scaler.transform(val_features)          # solo transforma
y_train = train_labels
y_val   = val_labels


# ===============================
# EXTRACCIÓN DE CARACTERÍSTICAS - TEST
# ===============================
# Al igual que en validación, NO se aplica aumento de datos en test.
# Se extrae un único vector de características por imagen original.

unique_paths = test_df["image_path"].unique()
print(f"\nExtrayendo features de TEST...")

test_imgs = []
for img_path in tqdm(unique_paths):
    full_path = os.path.join(TEST_IMG_DIR, img_path)
    test_imgs.append(load_image_as_array(full_path))

# Procesar en batches para aprovechar la GPU
test_features = []
for i in tqdm(range(0, len(test_imgs), BATCH_SIZE)):
    batch_feats = extract_features_batch(test_imgs[i : i + BATCH_SIZE])
    test_features.append(batch_feats)

test_features = np.vstack(test_features)   # shape: (n_test, 1536)
X_test        = scaler.transform(test_features)


# ===============================
# ENTRENAMIENTO DEL MODELO - Ridge
# ===============================
# Ridge Regression minimiza: ||y - Xw||² + alpha * ||w||²
# El término alpha * ||w||² penaliza coeficientes grandes, evitando
# sobreajuste cuando hay muchas características (1536) y pocas muestras (285).
#
# Se usa un modelo por variable objetivo porque Ridge con y 2D entrena
# internamente un regresor independiente por columna de todas formas,
# pero el loop hace explícita esa separación para facilitar el análisis
# individual de cada componente de biomasa.

models     = {}
test_preds = np.zeros((X_test.shape[0], len(TARGET_COLS)))

print(f"\nEntrenando Ridge (alpha={ALPHA})...")

for t_idx, target in enumerate(TARGET_COLS):

    model_r = Ridge(
        alpha=ALPHA,         # regularización L2: penaliza coeficientes grandes
        fit_intercept=True   # incluir término independiente en la regresión
    )

    model_r.fit(X_train, y_train[:, t_idx])

    models[target]        = model_r
    test_preds[:, t_idx]  = model_r.predict(X_test)
    print(f"  {target:<20} → entrenado")

print("\nEntrenamiento completado.")


# ============================================================
# MÓDULO DE VALIDACIÓN
# ============================================================
# Se evalúa el rendimiento del modelo sobre el 20 % de imágenes
# reservadas al inicio como conjunto de validación.
#
# Estas imágenes NO participaron en el entrenamiento ni en la
# selección interna de alpha de RidgeCV, por lo que las métricas
# son una estimación honesta del rendimiento del modelo.
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

for t_idx, col in enumerate(TARGET_COLS):

    # Valores reales del conjunto de validación para esta variable
    y_true_col = y_val[:, t_idx]

    # El modelo de RidgeCV ya entrenado predice sobre el conjunto de validación
    y_pred_col = models[col].predict(X_val)

    # --- MSE: promedio de (real - predicho)² ---
    mse  = mean_squared_error(y_true_col, y_pred_col)

    # --- RMSE: raíz del MSE, misma unidad que la biomasa (g/m²) ---
    rmse = np.sqrt(mse)

    # --- R²: proporción de varianza explicada por el modelo ---
    r2   = r2_score(y_true_col, y_pred_col)

    results.append({"Variable objetivo": col, "MSE": mse, "RMSE": rmse, "R²": r2})
    print(f"  {col:<20}  MSE={mse:>10.4f}  RMSE={rmse:>8.4f} g/m²  R²={r2:>6.4f}")

results_df = pd.DataFrame(results)
print("-" * 60)
print(f"  {'PROMEDIO GLOBAL':<20}  MSE={results_df['MSE'].mean():>10.4f}  "
      f"RMSE={results_df['RMSE'].mean():>8.4f} g/m²  R²={results_df['R²'].mean():>6.4f}")
print("=" * 60 + "\n")


# ===============================
# PREDICCIÓN Y ENVÍO
# ===============================

# Clamp a 0 para evitar predicciones negativas (biomasa no puede ser negativa)
test_preds = np.clip(test_preds, 0, None)

# Armar DataFrame ancho con las predicciones
predictions_df = pd.DataFrame(test_preds, columns=TARGET_COLS)
predictions_df["image_path"] = unique_paths

# Convertir de formato ancho a largo (long format requerido para la entrega)
predictions_long_df = predictions_df.melt(
    id_vars=["image_path"],
    value_vars=TARGET_COLS,
    var_name="target_name",
    value_name="target"
)

# Unir con test_df para recuperar el sample_id original
submission_df = pd.merge(
    test_df[["sample_id", "image_path", "target_name"]],
    predictions_long_df,
    on=["image_path", "target_name"]
)[["sample_id", "target"]]

submission_df.to_csv("submission.csv", index=False)
print("Archivo de entrega creado: submission.csv")
print(submission_df.head())


# ===============================
# GUARDAR MODELO Y ARTEFACTOS
# ===============================

# Guardar el diccionario de modelos RidgeCV (un modelo por target)
joblib.dump(models, os.path.join(SAVE_DIR, "ridge_models.pkl"))

# Guardar el scaler para poder transformar nuevas imágenes en inferencia
joblib.dump(scaler, os.path.join(SAVE_DIR, "scaler.pkl"))

# Guardar los nombres de los targets
joblib.dump(TARGET_COLS, os.path.join(SAVE_DIR, "target_cols.pkl"))

# Guardar los features de entrenamiento (opcional, para análisis futuros)
np.save(os.path.join(SAVE_DIR, "train_features.npy"), train_features)

print(f"Modelo y artefactos guardados en: {SAVE_DIR}")