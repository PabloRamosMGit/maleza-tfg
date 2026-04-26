# ============================================================
# SOLUCIÓN 3 - ResNet50 + SVR (LOCAL)
# Incluye: Preprocesamiento, Aumento de Datos,
#          Extracción de Características, Regresión,
#          Validación y Calibración
# Nota: Esta solucion se ejecuto en Kaggle con GPU, este es el link:
# https://www.kaggle.com/code/pabloramosmadrigal/resnet50-svr-third-solution
# Nota este fue un intento de usar SVR pero no salio tan bien, pero lo dejo aquí como referencia de lo que se intentó hacer.
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
# MultiOutputRegressor envuelve SVR para soportar múltiples variables objetivo,
# ya que SVR de sklearn solo acepta una variable objetivo a la vez.
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Para visión por computador
import tensorflow as tf
import keras_cv


# ===============================
# RUTAS
# ===============================

BASE_PATH = "/kaggle/input/competitions/csiro-biomass"

TRAIN_CSV_PATH = os.path.join(BASE_PATH, "train.csv")
TEST_CSV_PATH  = os.path.join(BASE_PATH, "test.csv")

TRAIN_IMG_DIR  = BASE_PATH
TEST_IMG_DIR   = BASE_PATH

SAVE_DIR = "saved_model_resnet_svr"
os.makedirs(SAVE_DIR, exist_ok=True)


# ===============================
# CONFIGURACIÓN GENERAL
# ===============================

IMG_SIZE    = 224       # Tamaño de entrada esperado por ResNet50
RANDOM_SEED = 999       # Semilla para reproducibilidad
VAL_SIZE    = 0.2       # 20 % de los datos se reservan para validación


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
train_augment = get_train_transforms()


# ===============================
# CARGA DEL MODELO ResNet50V2
# ===============================
# Se carga ResNet50V2 usando keras_cv desde una ruta local en Kaggle,
# lo que permite funcionar con internet desactivado.
#
# from_preset con una ruta local lee los pesos directamente del disco
# sin necesitar conexión externa. La ruta apunta al modelo descargado
# previamente en el dataset de Kaggle.
#
# include_rescaling=True (por defecto en from_preset) indica que el modelo
# aplica internamente el reescalado de píxeles, por lo que las imágenes
# deben pasarse en rango [0, 255] sin preprocesamiento adicional.
#
# GlobalAveragePooling2D reduce la salida espacial del backbone
# a un vector de 2048 dimensiones por imagen.

MODEL_PATH = "/kaggle/input/models/keras/resnetv2/keras/resnet50_v2/2"

base_net = keras_cv.models.ResNetV2Backbone.from_preset(MODEL_PATH)
base_net.trainable = False   # Los pesos preentrenados NO se actualizan

# Construir el extractor: backbone + pooling global
inputs  = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x       = base_net(inputs, training=False)
outputs = tf.keras.layers.GlobalAveragePooling2D()(x)

feature_extractor = tf.keras.Model(inputs=inputs, outputs=outputs)

print(f"Modelo cargado. Forma del vector de características: {feature_extractor.output_shape}")
# Debería imprimir: (None, 2048)


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


def extract_features(img_array):
    """
    Recibe un array HxWx3 uint8 y extrae su vector de características.

    Parámetros
    ----------
    img_array : np.ndarray
        Imagen en formato HxWx3 uint8.

    Retorna
    -------
    numpy.ndarray
        Vector 1-D de 2048 dimensiones con las características extraídas.
    """
    # keras_cv con from_preset incluye rescaling interno (include_rescaling=True),
    # por lo tanto las imágenes se pasan en rango [0, 255] sin preprocesamiento
    # adicional. No se usa preprocess_input de keras.applications aquí.
    x = np.expand_dims(img_array.astype(np.float32), axis=0)  # (1, 224, 224, 3)
    features = feature_extractor(x, training=False)
    return features.numpy().flatten()   # vector 1-D de 2048 dimensiones


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
# Por cada imagen se generan dos muestras: la original y una versión aumentada.
# Esto duplica el conjunto de entrenamiento de 285 a 570 muestras.
#
# Se usa UNA sola copia aumentada por imagen para añadir variabilidad
# sin generar una correlación excesiva entre muestras del mismo original.

print("\nExtrayendo features de TRAIN (original + 1 aumentada por imagen)...")

train_imgs, train_labels = [], []

for _, row in tqdm(train_split_df.iterrows(), total=len(train_split_df)):
    full_path = os.path.join(TRAIN_IMG_DIR, row["image_path"])
    img_array = load_image_as_array(full_path)
    label     = row[TARGET_COLS].values.astype(np.float32)

    # Imagen original
    train_imgs.append(img_array)
    train_labels.append(label)

    # Una única versión aumentada con las transformaciones definidas
    aug_img = train_augment(image=img_array)["image"]
    train_imgs.append(aug_img)
    train_labels.append(label)

print(f"Extrayendo características de {len(train_imgs)} imágenes...")
train_features = []

for img in tqdm(train_imgs):
    train_features.append(extract_features(img))

train_features = np.array(train_features)   # shape: (570, 2048)
train_labels   = np.array(train_labels)

print(f"→ {len(train_split_df)} imágenes × 2 = {train_features.shape[0]} ejemplos totales")
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

    val_imgs.append(img_array)
    val_labels.append(row[TARGET_COLS].values.astype(np.float32))

val_features = []
for img in tqdm(val_imgs):
    val_features.append(extract_features(img))

val_features = np.array(val_features)   # shape: (72, 2048)
val_labels   = np.array(val_labels)

print(f"Matriz de características de validación: {val_features.shape}")


# ===============================
# NORMALIZACIÓN
# ===============================
# El escalado es un requisito crítico para SVR, NO opcional.
#
# ¿Por qué?
#   SVR calcula distancias entre puntos en el espacio de características
#   para construir el hiperplano de separación. Si las características tienen
#   escalas muy diferentes, las de mayor magnitud dominarán el cálculo de
#   distancias y el modelo ignorará efectivamente las de menor magnitud,
#   sin importar cuán informativas sean.
#
#   StandardScaler estandariza cada característica a media=0 y desviación=1,
#   garantizando que todas contribuyan por igual al cálculo de distancias.
#
# REGLA: el scaler se ajusta (fit) SOLO con datos de entrenamiento y luego
# se aplica (transform) a validación y test para evitar data leakage.

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

test_features = []
for img in tqdm(test_imgs):
    test_features.append(extract_features(img))

test_features = np.array(test_features)   # shape: (n_test, 2048)
X_test        = scaler.transform(test_features)


# ===============================
# ENTRENAMIENTO DEL MODELO - SVR
# ===============================
# SVR (Support Vector Regression) busca ajustar una función que prediga
# el valor objetivo cometiendo un error máximo de epsilon. Solo las muestras
# fuera de ese margen de tolerancia contribuyen a la función de pérdida,
# lo que hace al modelo robusto ante outliers.
#
# Parámetros clave:
#   - kernel : 'rbf' (Radial Basis Function) proyecta los datos a un espacio
#              de alta dimensión donde la relación puede ser lineal, permitiendo
#              capturar patrones no lineales entre características y biomasa.
#   - C      : penalización por errores fuera del margen. Un C alto permite
#              menos errores pero puede sobreajustar; un C bajo es más tolerante.
#   - epsilon: define el margen de tolerancia dentro del cual no se penalizan
#              los errores. Muestras dentro del tubo epsilon son ignoradas.
#
# MultiOutputRegressor es necesario porque SVR solo acepta una variable
# objetivo a la vez. Este wrapper entrena un SVR independiente por cada
# columna de y, de forma transparente al resto del código.

print("\nEntrenando SVR con MultiOutputRegressor...")

model = MultiOutputRegressor(
    SVR(
        kernel="rbf",   # kernel radial para capturar relaciones no lineales
        C=1.0,          # penalización por errores fuera del margen epsilon
        epsilon=0.1     # margen de tolerancia: errores menores no se penalizan
    ),
    n_jobs=-1   # entrenar los 5 SVR (uno por target) en paralelo
)

model.fit(X_train, y_train)
print("Entrenamiento completado.")


# ============================================================
# MÓDULO DE VALIDACIÓN Y CALIBRACIÓN
# ============================================================
# Se evalúa el rendimiento del modelo comparando las predicciones
# generadas contra los valores reales del conjunto de validación.
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
#
#   - R² ponderado (métrica oficial de la competición):
#       Cada variable objetivo tiene un peso distinto según su importancia.
#       Dry_Total_g tiene el mayor peso (0.5) por ser la biomasa total,
#       mientras que las fracciones individuales pesan menos (0.1 cada una).
# ============================================================

# Pesos oficiales de la competición
TARGET_WEIGHTS = {
    "Dry_Green_g":  0.1,
    "Dry_Dead_g":   0.1,
    "Dry_Clover_g": 0.1,
    "GDM_g":        0.2,
    "Dry_Total_g":  0.5
}

def weighted_r2_score(y_true, y_pred, weights):
    """
    Calcula el R² ponderado definido por la competición:
    R²_w = 1 - (sum(w*(y - ŷ)²) / sum(w*(y - ȳ_w)²))
    donde ȳ_w = sum(w * y) / sum(w)
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    w      = np.asarray(weights).ravel()

    # Media ponderada de los valores reales
    y_mean_w = np.sum(w * y_true) / np.sum(w)

    # Suma de cuadrados ponderada de los residuos y de la variación total
    ss_res = np.sum(w * (y_true - y_pred) ** 2)
    ss_tot = np.sum(w * (y_true - y_mean_w) ** 2)

    # Si no hay variación (ss_tot == 0) se evita división por cero
    if ss_tot == 0:
        return float('nan')
    return 1 - ss_res / ss_tot


print("\n" + "=" * 60)
print("MÓDULO DE VALIDACIÓN Y CALIBRACIÓN")
print("=" * 60)

results = []
# Vectores para acumular todas las observaciones y pesos
all_y_true  = []
all_y_pred  = []
all_weights = []

# Generar predicciones sobre el conjunto de validación
# y_pred tiene shape (72, 5): una fila por imagen, una columna por target
y_pred = model.predict(X_val)

for i, col in enumerate(TARGET_COLS):

    y_true_col = y_val[:, i]
    y_pred_col = y_pred[:, i]

    # Aplicar clipping a cero, igual que en la inferencia final
    y_pred_col = np.clip(y_pred_col, 0, None)

    # --- MSE: promedio de (real - predicho)² ---
    mse  = mean_squared_error(y_true_col, y_pred_col)

    # --- RMSE: raíz del MSE, misma unidad que la biomasa (g/m²) ---
    rmse = np.sqrt(mse)

    # --- R²: proporción de varianza explicada por el modelo ---
    r2   = r2_score(y_true_col, y_pred_col)

    results.append({"Variable objetivo": col, "MSE": mse, "RMSE": rmse, "R²": r2})
    print(f"  {col:<20}  MSE={mse:>10.4f}  RMSE={rmse:>8.4f} g/m²  R²={r2:>6.4f}")

    # Acumular para el R² ponderado global
    all_y_true.append(y_true_col)
    all_y_pred.append(y_pred_col)
    all_weights.append(np.full_like(y_true_col, TARGET_WEIGHTS[col]))

# Concatenar todos los targets en un solo vector
all_y_true  = np.concatenate(all_y_true)
all_y_pred  = np.concatenate(all_y_pred)
all_weights = np.concatenate(all_weights)

# Calcular R² ponderado global (métrica oficial de la competición)
r2_weighted = weighted_r2_score(all_y_true, all_y_pred, all_weights)

# Resumen global
results_df = pd.DataFrame(results)
mean_mse  = results_df["MSE"].mean()
mean_rmse = results_df["RMSE"].mean()
mean_r2   = results_df["R²"].mean()

print("-" * 60)
print(f"  {'PROMEDIO GLOBAL (R² simple)':<30}  MSE={mean_mse:>10.4f}  RMSE={mean_rmse:>8.4f} g/m²  R²={mean_r2:>6.4f}")
print(f"  {'R² PONDERADO (métrica oficial)':<30}  R²={r2_weighted:>6.4f}")
print("=" * 60 + "\n")


# ===============================
# PREDICCIÓN Y ENVÍO
# ===============================

print("Generando predicciones finales...")

# El modelo predice las 5 variables objetivo simultáneamente
test_predictions = model.predict(X_test)

# Clamp a 0 para evitar predicciones negativas (biomasa no puede ser negativa)
test_predictions = np.clip(test_predictions, 0, None)

# Armar DataFrame ancho con las predicciones
predictions_df = pd.DataFrame(test_predictions, columns=TARGET_COLS)
predictions_df.insert(0, "image_path", unique_paths)

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
)[["sample_id", "target"]].dropna()

submission_df.to_csv("submission.csv", index=False)
print("Archivo de entrega creado: submission_resnet_svr.csv")
print(submission_df.head())


# ===============================
# GUARDAR MODELO Y ARTEFACTOS
# ===============================

# Guardar el modelo MultiOutputRegressor (contiene los 5 SVR internos)
joblib.dump(model, os.path.join(SAVE_DIR, "svr_model.pkl"))

# Guardar el scaler para poder transformar nuevas imágenes en inferencia
joblib.dump(scaler, os.path.join(SAVE_DIR, "scaler.pkl"))

# Guardar los nombres de los targets
joblib.dump(TARGET_COLS, os.path.join(SAVE_DIR, "target_cols.pkl"))

# Guardar los features de entrenamiento (opcional, para análisis futuros)
np.save(os.path.join(SAVE_DIR, "train_features.npy"), train_features)

print(f"Modelo y artefactos guardados en: {SAVE_DIR}")