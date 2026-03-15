import os
import joblib
import pandas as pd
import numpy as np
from tqdm import tqdm
import albumentations as A
import cv2

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input

# ===============================
# PATHS
# ===============================

BASE_PATH      = r'C:\Users\pablo\pablo-tfg-malezas\maleza-tfg\csiro-biomass'
TRAIN_CSV_PATH = os.path.join(BASE_PATH, 'train.csv')
TEST_CSV_PATH  = os.path.join(BASE_PATH, 'test.csv')
SAVE_DIR       = os.path.join(BASE_PATH, 'saved_model_efficientnet_simplified_attempt')

IMG_SIZE      = 300
RANDOM_SEED   = 999
N_FOLDS       = 3   # ↓ reducido de 5 → 3
N_AUGMENTS    = 3   # ↓ reducido de 10 → 3
N_TEST_AUGMENTS = 2 # ↓ reducido de 5  → 2

os.makedirs(SAVE_DIR, exist_ok=True)


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
# AUGMENTATION
# ===============================

augment = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.Resize(IMG_SIZE, IMG_SIZE, always_apply=True),
])


# ===============================
# LOAD EfficientNetB3
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
    x = img_array.astype(np.float32)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return base.predict(x, verbose=0).flatten()


# ===============================
# EXTRACT TRAIN FEATURES
# Intentamos cargar de disco primero para no recomputar
# ===============================

TRAIN_FEATS_PATH  = os.path.join(SAVE_DIR, 'train_features.npy')
TRAIN_LABELS_PATH = os.path.join(SAVE_DIR, 'train_labels.npy')

if os.path.exists(TRAIN_FEATS_PATH) and os.path.exists(TRAIN_LABELS_PATH):
    print("Cargando features de TRAIN desde disco...")
    train_features = np.load(TRAIN_FEATS_PATH)
    train_labels   = np.load(TRAIN_LABELS_PATH)
    print(f"Cargado: {train_features.shape}")
else:
    print("Extrayendo features de TRAIN con augmentación...")
    train_features, train_labels = [], []

    for _, row in tqdm(train_wide_df.iterrows(), total=len(train_wide_df)):
        full_path = os.path.join(BASE_PATH, row['image_path'])
        img_array = load_image_as_array(full_path)
        label     = row[TARGET_COLS].values.astype(np.float32)

        train_features.append(extract_features(img_array))
        train_labels.append(label)

        for _ in range(N_AUGMENTS):
            aug_img = augment(image=img_array)['image']
            train_features.append(extract_features(aug_img))
            train_labels.append(label)

    train_features = np.array(train_features)
    train_labels   = np.array(train_labels)

    np.save(TRAIN_FEATS_PATH,  train_features)
    np.save(TRAIN_LABELS_PATH, train_labels)
    print(f"Features guardadas en {SAVE_DIR}")

print(f"train_features: {train_features.shape} | train_labels: {train_labels.shape}")


# ===============================
# NORMALIZAR
# ===============================

scaler = StandardScaler()
X_train = scaler.fit_transform(train_features)
y_train = train_labels

joblib.dump(scaler, os.path.join(SAVE_DIR, 'scaler.pkl'))


# ===============================
# EXTRACT TEST FEATURES (TTA)
# ===============================

TEST_FEATS_PATH = os.path.join(SAVE_DIR, 'test_features.npy')
unique_paths    = test_df['image_path'].unique()

if os.path.exists(TEST_FEATS_PATH):
    print("Cargando features de TEST desde disco...")
    test_features = np.load(TEST_FEATS_PATH)
else:
    print("Extrayendo features de TEST con TTA...")
    test_features_list = []

    for img_path in tqdm(unique_paths):
        full_path = os.path.join(BASE_PATH, img_path)
        img_array = load_image_as_array(full_path)

        preds_tta = [extract_features(img_array)]
        for _ in range(N_TEST_AUGMENTS):
            aug = augment(image=img_array)['image']
            preds_tta.append(extract_features(aug))

        test_features_list.append(np.mean(preds_tta, axis=0))

    test_features = np.array(test_features_list)
    np.save(TEST_FEATS_PATH, test_features)
    print(f"Test features guardadas: {test_features.shape}")

X_test = scaler.transform(test_features)


# ===============================
# ENSEMBLE SIMPLIFICADO
# RF + Ridge (eliminamos GBM y SVR para ser más rápidos)
# ===============================

def make_models():
    return [
        ('rf',    RandomForestRegressor(n_estimators=300, max_features=0.5,   # ↓ 500→300
                                         random_state=RANDOM_SEED, n_jobs=-1, verbose=2)),
        ('ridge', Ridge(alpha=10.0)),
    ]


kf         = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
oof_preds  = np.zeros_like(y_train)
test_preds = np.zeros((X_test.shape[0], len(TARGET_COLS)))

all_models = {}   # guardamos modelos del último fold por target

print(f"\nEntrenando ensemble {N_FOLDS}-Fold CV...")

for t_idx, target in enumerate(TARGET_COLS):
    y_t = y_train[:, t_idx]
    fold_test_preds = np.zeros((N_FOLDS, X_test.shape[0]))
    target_models   = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_t[train_idx],     y_t[val_idx]

        fold_preds_val  = np.zeros(len(val_idx))
        fold_preds_test = np.zeros(X_test.shape[0])

        models = make_models()
        for name, mdl in models:
            mdl.fit(X_tr, y_tr)
            fold_preds_val  += mdl.predict(X_val)  / len(models)
            fold_preds_test += mdl.predict(X_test) / len(models)

        oof_preds[val_idx, t_idx] = fold_preds_val
        fold_test_preds[fold]     = fold_preds_test
        target_models.append(models)

        rmse = np.sqrt(np.mean((fold_preds_val - y_val) ** 2))
        print(f"  {target} | fold {fold+1}/{N_FOLDS} | RMSE = {rmse:.4f}")

    test_preds[:, t_idx] = fold_test_preds.mean(axis=0)
    all_models[target]   = target_models   # lista de folds, cada fold = lista de (name, model)


# ===============================
# GUARDAR MODELOS Y ARTEFACTOS
# ===============================

joblib.dump(all_models,        os.path.join(SAVE_DIR, 'ensemble_models.pkl'))
joblib.dump(list(TARGET_COLS), os.path.join(SAVE_DIR, 'target_cols.pkl'))

print(f"\nModelos guardados en: {SAVE_DIR}")


# ===============================
# RMSE OOF REPORT
# ===============================

print("\n--- RMSE OOF por target ---")
for t_idx, target in enumerate(TARGET_COLS):
    rmse = np.sqrt(np.mean((oof_preds[:, t_idx] - y_train[:, t_idx]) ** 2))
    print(f"  {target}: {rmse:.4f}")


# ===============================
# SUBMISSION
# ===============================

test_preds = np.clip(test_preds, 0, None)

submission_rows = []
for i, img_path in enumerate(unique_paths):
    for j, target in enumerate(TARGET_COLS):
        submission_rows.append({
            'sample_id': f"{img_path}_{target}",
            'target':    test_preds[i, j]
        })

submission_df = pd.DataFrame(submission_rows)
submission_df.to_csv('submission_fast.csv', index=False)
print("\nSubmission creado: submission_fast.csv")
print(submission_df.head(10))