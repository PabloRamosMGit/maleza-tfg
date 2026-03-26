import os
import sys
import torch
import numpy as np
import pandas as pd
import cv2
import albumentations as A
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import lightgbm as lgb
import torchvision.transforms as T
import timm

# ===============================
# PATHS
# ===============================

BASE_PATH      = "/kaggle/input/competitions/csiro-biomass"
TRAIN_CSV_PATH = os.path.join(BASE_PATH, "train.csv")
TEST_CSV_PATH  = os.path.join(BASE_PATH, "test.csv")
DINOV2_PATH    = "/kaggle/input/models/metaresearch/dinov2/pytorch/giant/1"

IMG_SIZE    = 518
RANDOM_SEED = 999

N_FOLDS         = 5
N_AUGMENTS      = 10
N_TEST_AUGMENTS = 20

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Dispositivo: {DEVICE}")


# ===============================
# PESOS DE LA COMPETENCIA
# ===============================

TARGET_WEIGHTS = {
    "Dry_Green_g":  0.1,
    "Dry_Dead_g":   0.1,
    "Dry_Clover_g": 0.1,
    "GDM_g":        0.2,
    "Dry_Total_g":  0.5,
}
TARGET_COLS = list(TARGET_WEIGHTS.keys())
WEIGHT_VEC  = np.array([TARGET_WEIGHTS[t] for t in TARGET_COLS], dtype=np.float32)


# ===============================
# MÉTRICA OFICIAL
# ===============================

def weighted_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    N       = y_true.shape[0]
    w_mat   = np.tile(WEIGHT_VEC, (N, 1))
    y_flat  = y_true.flatten()
    yh_flat = y_pred.flatten()
    w_flat  = w_mat.flatten()
    y_bar   = np.dot(w_flat, y_flat) / w_flat.sum()
    ss_res  = np.dot(w_flat, (y_flat - yh_flat) ** 2)
    ss_tot  = np.dot(w_flat, (y_flat - y_bar) ** 2)
    return float(1.0 - ss_res / ss_tot)


def weighted_r2_lgb(y_pred, dataset):
    y_true = dataset.get_label()
    w      = dataset.get_weight()
    y_bar  = np.dot(w, y_true) / w.sum()
    ss_res = np.dot(w, (y_true - y_pred) ** 2)
    ss_tot = np.dot(w, (y_true - y_bar) ** 2)
    return "weighted_r2", float(1.0 - ss_res / ss_tot), True


# ===============================
# LOAD DATA
# Inspeccionar columnas reales antes de asumir estructura
# ===============================

def inspect_csv(path, label="CSV"):
    """Muestra las primeras 2 filas tal cual para diagnosticar estructura."""
    raw = pd.read_csv(path, header=None, nrows=2, dtype=str)
    print(f"\n{label} — {raw.shape[1]} columnas detectadas:")
    for i, row in raw.iterrows():
        print(f"  fila {i}: {row.tolist()}")
    return raw.shape[1]

n_train_cols = inspect_csv(TRAIN_CSV_PATH, "TRAIN CSV")
n_test_cols  = inspect_csv(TEST_CSV_PATH,  "TEST CSV")


# Nombres para train (9 columnas con encabezado)
TRAIN_COL_NAMES = ["sample_id", "image_path", "date", "state",
                   "pasture_type", "numeric1", "numeric2", "target_name", "target"]

# Nombres para test — puede tener 3 o más columnas
# La estructura observada es: sample_id, image_path, target_name
TEST_COL_NAMES_3  = ["sample_id", "image_path", "target_name"]
TEST_COL_NAMES_8  = ["sample_id", "image_path", "date", "state",
                     "pasture_type", "numeric1", "numeric2", "target_name"]
TEST_COL_NAMES_9  = TRAIN_COL_NAMES[:-1]   # sin target


def load_csv_by_position(path, col_names):
    """Lee CSV, descarta encabezado si lo tiene, renombra por posición."""
    raw = pd.read_csv(path, header=None, dtype=str)
    # Detectar encabezado: primera celda no empieza con "ID"
    if not str(raw.iloc[0, 0]).startswith("ID"):
        raw = raw.iloc[1:].reset_index(drop=True)
    n = min(len(col_names), raw.shape[1])
    df = raw.iloc[:, :n].copy()
    df.columns = col_names[:n]
    return df


train_df = load_csv_by_position(TRAIN_CSV_PATH, TRAIN_COL_NAMES)
train_df["target"] = pd.to_numeric(train_df["target"], errors="coerce")

# Para test: elegir nombres según número de columnas real
if n_test_cols >= 9:
    test_col_names = TEST_COL_NAMES_9
elif n_test_cols >= 8:
    test_col_names = TEST_COL_NAMES_8
else:
    test_col_names = TEST_COL_NAMES_3

test_df = load_csv_by_position(TEST_CSV_PATH, test_col_names)

# Si test solo tiene 3 cols, las columnas tabulares serán desconocidas → se rellenan con 0
TEST_HAS_METADATA = all(c in test_df.columns
                        for c in ["date", "state", "pasture_type", "numeric1", "numeric2"])

print(f"\ntrain_df shape : {train_df.shape}  columnas: {list(train_df.columns)}")
print(f"test_df  shape : {test_df.shape}  columnas: {list(test_df.columns)}")
print(f"Test tiene metadata tabular: {TEST_HAS_METADATA}")
print(f"Targets únicos : {sorted(train_df['target_name'].unique())}")


# ===============================
# PIVOT TRAIN
# ===============================

train_df = train_df[train_df["target_name"].isin(TARGET_COLS)].copy()

train_wide_df = train_df.pivot(
    index="image_path",
    columns="target_name",
    values="target"
).reset_index()[["image_path"] + TARGET_COLS]

print(f"\nImágenes train : {len(train_wide_df)}")
print(f"Imágenes test  : {len(test_df['image_path'].unique())}")


# ===============================
# FEATURES TABULARES
# ===============================

def parse_date_safe(series: pd.Series) -> pd.Series:
    for fmt in ("%Y/%m/%d", "%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%m/%d/%Y"):
        try:
            return pd.to_datetime(series, format=fmt, errors="raise")
        except Exception:
            continue
    return pd.to_datetime(series, errors="coerce")


def build_tabular_features(df_raw, is_test=False):
    df = df_raw.drop_duplicates("image_path").copy()

    # Fecha — solo si existe y tiene datos válidos
    if "date" in df.columns:
        df["date_parsed"] = parse_date_safe(df["date"])
        valid_dates = df["date_parsed"].notna().sum()
        if valid_dates > 0:
            df["month_sin"] = np.sin(2 * np.pi * df["date_parsed"].dt.month / 12)
            df["month_cos"] = np.cos(2 * np.pi * df["date_parsed"].dt.month / 12)
            df["doy_sin"]   = np.sin(2 * np.pi * df["date_parsed"].dt.dayofyear / 365)
            df["doy_cos"]   = np.cos(2 * np.pi * df["date_parsed"].dt.dayofyear / 365)
            season_map = {12:0,1:0,2:0, 3:1,4:1,5:1, 6:2,7:2,8:2, 9:3,10:3,11:3}
            df["season"] = df["date_parsed"].dt.month.map(season_map)
        else:
            print("  AVISO: fechas no parseables → columnas temporales = 0")
            for col in ["month_sin", "month_cos", "doy_sin", "doy_cos", "season"]:
                df[col] = 0.0
    else:
        for col in ["month_sin", "month_cos", "doy_sin", "doy_cos", "season"]:
            df[col] = 0.0

    # Numéricas
    for col in ["numeric1", "numeric2"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float32)
            df[col] = df[col].fillna(df[col].median() if not df[col].isna().all() else 0.0)
        else:
            df[col] = 0.0

    # One-hot
    cat_cols = [c for c in ["state", "pasture_type"] if c in df.columns]
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=False)

    # Limpiar columnas no-feature
    drop_cols = [c for c in ["sample_id", "date", "date_parsed", "target_name",
                              "target", "state", "pasture_type"] if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    return df.set_index("image_path")


print("\nConstruyendo features tabulares...")
train_meta = build_tabular_features(train_df, is_test=False)
test_meta  = build_tabular_features(test_df,  is_test=True)

EXCLUDE_COLS = set(TARGET_COLS)
TABULAR_COLS = [c for c in train_meta.columns if c not in EXCLUDE_COLS]

# Alinear: test puede tener menos columnas one-hot → rellenar con 0
train_meta, test_meta = train_meta.align(test_meta, join="left", axis=1, fill_value=0)
train_meta = train_meta.reindex(train_wide_df["image_path"].values)

print(f"Features tabulares ({len(TABULAR_COLS)}): {TABULAR_COLS}")


# ===============================
# CARGAR DINOv2 GIANT via timm
# (no requiere hubconf.py)
# ===============================

print(f"\nCargando DINOv2 Giant via timm desde: {DINOV2_PATH}")

# timm model name para ViT-Giant/14 DINOv2
TIMM_MODEL = "vit_giant_patch14_dinov2.lvd142m"

dinov2 = timm.create_model(TIMM_MODEL, pretrained=False, num_classes=0)

# Buscar archivo de pesos en el path de Kaggle
weights_path = None
print(f"Archivos en {DINOV2_PATH}:")
for f in sorted(os.listdir(DINOV2_PATH)):
    print(f"  {f}")
    if f.endswith((".pth", ".bin")) and weights_path is None:
        weights_path = os.path.join(DINOV2_PATH, f)

if weights_path is None:
    raise FileNotFoundError(
        f"No se encontró ningún archivo .pth/.bin en {DINOV2_PATH}\n"
        f"Contenido: {os.listdir(DINOV2_PATH)}"
    )

print(f"\nCargando pesos desde: {weights_path}")
state_dict = torch.load(weights_path, map_location="cpu")

# El state_dict puede venir envuelto en una clave "model" o "state_dict"
if isinstance(state_dict, dict):
    for key in ["model", "state_dict", "teacher", "student"]:
        if key in state_dict:
            print(f"  Desenvuelto desde clave '{key}'")
            state_dict = state_dict[key]
            break

# Cargar pesos (strict=False para tolerar capas extra como la cabeza de clasificación)
missing, unexpected = dinov2.load_state_dict(state_dict, strict=False)
print(f"  Keys faltantes  : {len(missing)}")
print(f"  Keys inesperadas: {len(unexpected)}")

dinov2 = dinov2.eval().to(DEVICE)
EMBED_DIM = dinov2(torch.zeros(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)).shape[-1]
print(f"DINOv2 Giant listo. Dim embedding: {EMBED_DIM}")


# ===============================
# TRANSFORMS
# ===============================

def get_train_transforms():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.GaussNoise(p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.75),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.5),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.75),
        A.Resize(IMG_SIZE, IMG_SIZE),
    ], seed=RANDOM_SEED)


def get_test_transforms():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=10, p=0.3),
        A.Resize(IMG_SIZE, IMG_SIZE),
    ], seed=RANDOM_SEED)


train_augment = get_train_transforms()
test_augment  = get_test_transforms()

to_tensor = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"No se pudo leer: {img_path}")
    return cv2.cvtColor(cv2.resize(img, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_BGR2RGB)


@torch.no_grad()
def extract_features(img_array):
    x = to_tensor(img_array).unsqueeze(0).to(DEVICE)
    return dinov2(x).cpu().numpy().flatten().astype(np.float32)


# ===============================
# EXTRAER FEATURES TRAIN
# ===============================

print(f"\nExtrayendo features TRAIN (×{N_AUGMENTS + 1})...")

image_paths_train = train_wide_df["image_path"].tolist()
train_features, train_labels = [], []

for img_path in tqdm(image_paths_train):
    full_path = os.path.join(BASE_PATH, img_path)
    img_array = load_image(full_path)
    label     = train_wide_df.loc[
        train_wide_df["image_path"] == img_path, TARGET_COLS
    ].values[0].astype(np.float32)

    train_features.append(extract_features(img_array))
    train_labels.append(label)

    for _ in range(N_AUGMENTS):
        aug = train_augment(image=img_array)["image"]
        train_features.append(extract_features(aug))
        train_labels.append(label)

train_features = np.array(train_features)
train_labels   = np.array(train_labels)
print(f"→ Shape features train: {train_features.shape}")

img_scaler  = StandardScaler()
X_img_train = img_scaler.fit_transform(train_features)

tab_scaler         = StandardScaler()
tab_train_scaled   = tab_scaler.fit_transform(train_meta[TABULAR_COLS].values.astype(np.float32))
tab_train_expanded = np.repeat(tab_train_scaled, N_AUGMENTS + 1, axis=0)

X_train = np.concatenate([X_img_train, tab_train_expanded], axis=1)
y_train = train_labels
print(f"Shape X_train: {X_train.shape}")


# ===============================
# EXTRAER FEATURES TEST (TTA)
# ===============================

unique_paths_test = test_df["image_path"].unique()
print(f"\nExtrayendo features TEST con TTA (×{N_TEST_AUGMENTS + 1})...")

test_features_list = []
for img_path in tqdm(unique_paths_test):
    full_path = os.path.join(BASE_PATH, img_path)
    img_array = load_image(full_path)

    preds_tta = [extract_features(img_array)]
    for _ in range(N_TEST_AUGMENTS):
        aug = test_augment(image=img_array)["image"]
        preds_tta.append(extract_features(aug))

    test_features_list.append(np.mean(preds_tta, axis=0))

test_features = np.array(test_features_list)
X_img_test    = img_scaler.transform(test_features)

test_meta_ordered = test_meta.reindex(unique_paths_test)
tab_test_scaled   = tab_scaler.transform(
    test_meta_ordered[TABULAR_COLS].values.astype(np.float32)
)

X_test = np.concatenate([X_img_test, tab_test_scaled], axis=1)
print(f"Shape X_test: {X_test.shape}")


# ===============================
# MODELO: LightGBM con pesos de competencia
# ===============================

lgb_params = {
    "objective":         "regression",
    "metric":            "None",
    "learning_rate":     0.05,
    "num_leaves":        63,
    "min_child_samples": 20,
    "feature_fraction":  0.8,
    "bagging_fraction":  0.8,
    "bagging_freq":      5,
    "lambda_l1":         0.1,
    "lambda_l2":         0.1,
    "verbose":           -1,
    "seed":              RANDOM_SEED,
}

kf         = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
oof_preds  = np.zeros_like(y_train)
test_preds = np.zeros((X_test.shape[0], len(TARGET_COLS)))

print(f"\nEntrenando LightGBM ({N_FOLDS}-Fold CV) ...\n")

for t_idx, target in enumerate(TARGET_COLS):
    y_t        = y_train[:, t_idx]
    row_weight = np.full(len(y_t), TARGET_WEIGHTS[target], dtype=np.float32)
    fold_test_preds = []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_t[tr_idx],     y_t[val_idx]
        w_tr, w_val = row_weight[tr_idx], row_weight[val_idx]

        dtrain = lgb.Dataset(X_tr, label=y_tr, weight=w_tr)
        dval   = lgb.Dataset(X_val, label=y_val, weight=w_val, reference=dtrain)

        model = lgb.train(
            lgb_params,
            dtrain,
            num_boost_round=2000,
            valid_sets=[dval],
            feval=weighted_r2_lgb,
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False),
                lgb.log_evaluation(period=500),
            ],
        )

        val_pred  = model.predict(X_val)
        test_pred = model.predict(X_test)

        oof_preds[val_idx, t_idx] = val_pred
        fold_test_preds.append(test_pred)

        w_sum = w_val.sum()
        y_bar = np.dot(w_val, y_val) / w_sum
        r2_f  = 1.0 - np.dot(w_val, (y_val - val_pred)**2) / np.dot(w_val, (y_val - y_bar)**2)
        print(f"  {target:15s} | fold {fold+1}/{N_FOLDS} "
              f"| R²={r2_f:.4f} | iter={model.best_iteration}")

    test_preds[:, t_idx] = np.mean(fold_test_preds, axis=0)
    print()


# ===============================
# REPORTE OOF — MÉTRICA OFICIAL
# ===============================

print("=" * 60)
print("RESULTADOS OOF — MÉTRICA OFICIAL DE LA COMPETENCIA")
print("=" * 60)

r2_global = weighted_r2(y_train, oof_preds)
print(f"\n  R² global ponderado (OOF): {r2_global:.5f}\n")

print(f"  {'Target':20s} {'Peso':>5}  {'R² OOF':>8}  {'RMSE':>8}")
print("  " + "-" * 48)
for t_idx, target in enumerate(TARGET_COLS):
    y_t   = y_train[:, t_idx]
    y_hat = oof_preds[:, t_idx]
    rmse  = np.sqrt(np.mean((y_hat - y_t) ** 2))
    r2_t  = 1.0 - np.sum((y_hat - y_t)**2) / np.sum((y_t - y_t.mean())**2)
    print(f"  {target:20s} {TARGET_WEIGHTS[target]:>5.1f}  {r2_t:>8.4f}  {rmse:>8.4f}")

print("=" * 60)


# ===============================
# SUBMISSION
# ===============================

test_preds_clipped = np.clip(test_preds, 0, None)

predictions_df = pd.DataFrame(test_preds_clipped, columns=TARGET_COLS)
predictions_df["image_path"] = unique_paths_test

predictions_long = predictions_df.melt(
    id_vars=["image_path"],
    value_vars=TARGET_COLS,
    var_name="target_name",
    value_name="target",
)

submission_df = pd.merge(
    test_df[["sample_id", "image_path", "target_name"]],
    predictions_long,
    on=["image_path", "target_name"],
)[["sample_id", "target"]]

submission_df.to_csv("submission.csv", index=False)
print(f"\nSubmission guardado: submission.csv  ({len(submission_df)} filas)")
print(submission_df.head(10))
