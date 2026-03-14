import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import albumentations as A
import cv2

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image

################################# THIS IS AN IMPLEMENTACION BASE OF CLAUDE AIs sUGGESTION, NOT A FINAL SOLUTION. #################################
# https://claude.ai/chat/d53a044c-08fb-47f0-be7a-e35260a6d4f0
# ===============================
# PATHS
# ===============================

BASE_PATH = r'C:\Users\pablo\pablo-tfg-malezas\maleza-tfg\csiro-biomass'
TRAIN_CSV_PATH = os.path.join(BASE_PATH, 'train.csv')
TEST_CSV_PATH  = os.path.join(BASE_PATH, 'test.csv')

IMG_SIZE    = 300   # EfficientNetB3 espera 300x300
RANDOM_SEED = 999
N_FOLDS     = 5
N_AUGMENTS  = 10    # copias aumentadas por imagen original


# ===============================
# LOAD DATA
# ===============================

train_df = pd.read_csv(TRAIN_CSV_PATH)
test_df  = pd.read_csv(TEST_CSV_PATH)

train_wide_df = train_df.pivot(
    index='image_path',
    columns='target_name',
    values='target'
).reset_index()

TARGET_COLS = [c for c in train_wide_df.columns if c != 'image_path']
print("Targets:", TARGET_COLS)


# ===============================
# DATA AUGMENTATION PIPELINE
# ===============================

augment = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=30, p=0.7),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.4),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    A.RandomCrop(height=260, width=260, p=0.3),
    A.Resize(IMG_SIZE, IMG_SIZE, always_apply=True),
])


# ===============================
# LOAD EfficientNetB3 (mejor que VGG16)
# Extraemos features de la penúltima capa
# ===============================

base = EfficientNetB3(
    weights='imagenet',
    include_top=False,
    pooling='avg',
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base.trainable = False

print(f"EfficientNetB3 cargado. Output shape: {base.output_shape}")


def load_image_as_array(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return img


def extract_features(img_array):
    """Recibe un numpy array HxWxC, devuelve vector de features."""
    x = img_array.astype(np.float32)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    feats = base.predict(x, verbose=0)
    return feats.flatten()


# ===============================
# EXTRACT TRAIN FEATURES + AUGMENTATION
# ===============================

print("Extrayendo features de TRAIN con augmentación...")

train_features = []
train_labels   = []

for idx, row in tqdm(train_wide_df.iterrows(), total=len(train_wide_df)):

    full_path  = os.path.join(BASE_PATH, row['image_path'])
    img_array  = load_image_as_array(full_path)
    label      = row[TARGET_COLS].values.astype(np.float32)

    # Original
    train_features.append(extract_features(img_array))
    train_labels.append(label)

    # Augmented copies
    for _ in range(N_AUGMENTS):
        aug_img = augment(image=img_array)['image']
        train_features.append(extract_features(aug_img))
        train_labels.append(label)   # mismos targets para todas las copias

train_features = np.array(train_features)
train_labels   = np.array(train_labels)

print(f"Dataset aumentado: {train_features.shape[0]} muestras "
      f"({len(train_wide_df)} originales × {N_AUGMENTS + 1})")


# ===============================
# NORMALIZAR FEATURES
# ===============================

scaler = StandardScaler()
X_train = scaler.fit_transform(train_features)
y_train = train_labels


# ===============================
# EXTRACT TEST FEATURES
# (sin augmentación, pero promediamos N pasadas para más estabilidad)
# ===============================

N_TEST_AUGMENTS = 5   # TTA: Test-Time Augmentation

print("Extrayendo features de TEST con TTA...")

test_features_list = []
unique_paths = test_df['image_path'].unique()

for img_path in tqdm(unique_paths):
    full_path = os.path.join(BASE_PATH, img_path)
    img_array = load_image_as_array(full_path)

    preds_tta = [extract_features(img_array)]   # original
    for _ in range(N_TEST_AUGMENTS):
        aug = augment(image=img_array)['image']
        preds_tta.append(extract_features(aug))

    # Promedio de features (TTA en espacio de features)
    test_features_list.append(np.mean(preds_tta, axis=0))

test_features = np.array(test_features_list)
X_test = scaler.transform(test_features)


# ===============================
# TRAIN: UN MODELO POR TARGET + ENSEMBLE
# K-Fold Cross-Validation para medir error real
# ===============================

def make_models():
    """Devuelve lista de regressores para ensamblar."""
    return [
        ('rf',  RandomForestRegressor(n_estimators=500, max_features=0.5,
                                       random_state=RANDOM_SEED, n_jobs=-1)),
        ('gbm', GradientBoostingRegressor(n_estimators=300, max_depth=4,
                                           learning_rate=0.05, subsample=0.8,
                                           random_state=RANDOM_SEED)),
        ('svr', SVR(C=10, epsilon=0.1, kernel='rbf')),
        ('ridge', Ridge(alpha=10.0)),
    ]


kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

# Guardaremos las predicciones OOF y de test por target
oof_preds  = np.zeros_like(y_train)
test_preds = np.zeros((X_test.shape[0], len(TARGET_COLS)))

print(f"\nEntrenando ensemble con {N_FOLDS}-Fold CV...")

for t_idx, target in enumerate(TARGET_COLS):

    y_t = y_train[:, t_idx]
    fold_test_preds = np.zeros((N_FOLDS, X_test.shape[0]))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):

        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_t[train_idx], y_t[val_idx]

        fold_preds_val  = np.zeros(len(val_idx))
        fold_preds_test = np.zeros(X_test.shape[0])

        models = make_models()
        for name, mdl in models:
            mdl.fit(X_tr, y_tr)
            fold_preds_val  += mdl.predict(X_val)  / len(models)
            fold_preds_test += mdl.predict(X_test) / len(models)

        oof_preds[val_idx, t_idx] = fold_preds_val
        fold_test_preds[fold] = fold_preds_test

        rmse = np.sqrt(np.mean((fold_preds_val - y_val) ** 2))
        print(f"  {target} | fold {fold+1}/{N_FOLDS} | RMSE = {rmse:.4f}")

    # Promedio de folds para test
    test_preds[:, t_idx] = fold_test_preds.mean(axis=0)

# RMSE global por target
print("\n--- RMSE OOF por target ---")
for t_idx, target in enumerate(TARGET_COLS):
    rmse = np.sqrt(np.mean((oof_preds[:, t_idx] - y_train[:, t_idx]) ** 2))
    print(f"  {target}: {rmse:.4f}")


# ===============================
# ASEGURAR PREDICCIONES >= 0
# (pesos en gramos no pueden ser negativos)
# ===============================

test_preds = np.clip(test_preds, 0, None)


# ===============================
# CREATE SUBMISSION
# ===============================

submission_rows = []

for i, img_path in enumerate(unique_paths):
    for j, target in enumerate(TARGET_COLS):
        submission_rows.append({
            'sample_id': f"{img_path}_{target}",
            'target': test_preds[i, j]
        })

submission_df = pd.DataFrame(submission_rows)
submission_df.to_csv('submission.csv', index=False)
print("\nSubmission creado: submission.csv")
print(submission_df.head(10))