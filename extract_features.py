# For data manipulation
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# For machine learning model
from sklearn.ensemble import RandomForestRegressor

# For computer vision
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image


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

print("Train wide dataframe shape:", train_wide_df.shape)


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

print("Training RandomForest...")

model = RandomForestRegressor(
    n_estimators=300,
    random_state=RANDOM_SEED,
    n_jobs=-1
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

submission_rows = []

# Obtener imágenes únicas (test_df está en formato largo)
unique_image_paths = test_df["image_path"].unique()

for i, img_path in enumerate(unique_image_paths):

    for j, target in enumerate(TARGET_COLS):

        submission_rows.append({
            "sample_id": f"{img_path}_{target}",
            "target": test_predictions[i, j]
        })

submission_df = pd.DataFrame(submission_rows)

submission_df.to_csv("submission.csv", index=False)

print("Submission file created: submission.csv")