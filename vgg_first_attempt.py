# For data manipulation
import os
import joblib
import pandas as pd
import numpy as np
from tqdm import tqdm

# For machine learning model
from sklearn.ensemble import RandomForestRegressor

# For computer vision
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image

# For plotting
import matplotlib.pyplot as plt

# ===============================
# PATHS
# ===============================

# Base dataset path
BASE_PATH = r'C:\Users\pablo\pablo-tfg-malezas\maleza-tfg\csiro-biomass'

# CSV files
TRAIN_CSV_PATH = os.path.join(BASE_PATH, 'train.csv')
TEST_CSV_PATH = os.path.join(BASE_PATH, 'test.csv')

# Image folders
TRAIN_IMG_DIR = BASE_PATH
TEST_IMG_DIR = BASE_PATH


# ===============================
# LOAD DATA
# ===============================

train_df = pd.read_csv(TRAIN_CSV_PATH)
test_df = pd.read_csv(TEST_CSV_PATH)


# Convert to wide format
train_wide_df = train_df.pivot(
    index='image_path',
    columns='target_name',
    values='target'
).reset_index()

print("Train wide dataframe shape:\n", train_wide_df)


# ===============================
# CONFIG
# ===============================

IMG_SIZE = 224
RANDOM_SEED = 999


# ===============================
# LOAD VGG16 MODEL
# ===============================

full_model = VGG16(
    weights="imagenet",
    include_top=True,
    input_shape=(224, 224, 3)
)

base_model = tf.keras.Model(
    inputs=full_model.input,
    outputs=full_model.layers[-2].output
)

base_model.trainable = False

print(f"Model loaded. Feature vector output shape: {base_model.output_shape}")


# ===============================
# FEATURE EXTRACTION FUNCTION
# ===============================

def extract_features(img_path):

    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    features = base_model.predict(img_array, verbose=0)

    return features.flatten()


# ===============================
# EXTRACT TRAIN FEATURES
# ===============================

train_features = []

print("Extracting TRAIN features...")

for img_path in tqdm(train_wide_df["image_path"]):

    full_path = os.path.join(TRAIN_IMG_DIR, img_path)

    features = extract_features(full_path)

    train_features.append(features)

train_features = np.array(train_features)

print("Train feature matrix:", train_features)


# ===============================
# TRAIN MODEL
# ===============================

TARGET_COLS = train_wide_df.columns[1:]

X_train = train_features
y_train = train_wide_df[TARGET_COLS]

print("X_train shape:", X_train.shape)
print("y_train shape:\n", y_train)

print("Training RandomForest...")

model = RandomForestRegressor(
    n_estimators=300,
    random_state=RANDOM_SEED,
    n_jobs=-1,
    verbose=2
)

model.fit(X_train, y_train)

print("Training completed.")


# =============================== TEST FEATURES AND PREDICTION ===============================

# ===============================
# EXTRACT TEST FEATURES
# ===============================

test_features = []

unique_image_paths = test_df["image_path"].unique()  # <-- también aquí

print("Extracting TEST features...")

for img_path in tqdm(unique_image_paths):

    full_path = os.path.join(TEST_IMG_DIR, img_path)
    features = extract_features(full_path)
    test_features.append(features)

test_features = np.array(test_features)


# ===============================
# PREDICT
# ===============================

print("Predicting...")

test_predictions = model.predict(test_features)
print("Test predictions shape:", test_predictions)

# ===============================
# CREATE SUBMISSION
# ===============================

# Armar dataframe wide con las predicciones
predictions_df = pd.DataFrame(
    test_predictions,
    columns=TARGET_COLS
)
predictions_df.insert(0, "image_path", unique_image_paths)

# Pasar de wide a long
predictions_long_df = predictions_df.melt(
    id_vars=["image_path"],
    value_vars=TARGET_COLS,
    var_name="target_name",
    value_name="target"
)

# Merge con test_df para recuperar el sample_id original
submission_df = pd.merge(
    test_df[["sample_id", "image_path", "target_name"]],
    predictions_long_df,
    on=["image_path", "target_name"]
)[["sample_id", "target"]].dropna()

submission_df.to_csv("submission2.csv", index=False)

print("Submission file created: submission2.csv")
print(submission_df.head())

# ===============================
# SAVE MODEL & ARTIFACTS
# ===============================

SAVE_DIR = os.path.join(BASE_PATH, 'saved_model_vgg')
os.makedirs(SAVE_DIR, exist_ok=True)

# Guardar el RandomForest
joblib.dump(model, os.path.join(SAVE_DIR, 'random_forest.pkl'))

# Guardar los nombres de los targets (para reconstruir el submission)
joblib.dump(list(TARGET_COLS), os.path.join(SAVE_DIR, 'target_cols.pkl'))

# Guardar los features de train (opcional, para análisis futuros)
np.save(os.path.join(SAVE_DIR, 'train_features.npy'), train_features)

print(f"Modelo y artefactos guardados en: {SAVE_DIR}")