import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# ===============================
# PATHS & CONFIG
# ===============================

BASE_PATH = r'C:\Users\pablo\maleza-tfg\csiro-biomass'
IMG_SIZE = 224

# ===============================
# LOAD VGG16 BASE MODEL
# ===============================

full_model = tf.keras.applications.VGG16(
    weights="imagenet",
    include_top=True,
    input_shape=(224, 224, 3)
)

base_model = tf.keras.Model(
    inputs=full_model.input,
    outputs=full_model.layers[-2].output  # fc2 layer → 4096-dim vector
)
base_model.trainable = False


# ===============================
# FEATURE EXTRACTION FUNCTION
# ===============================

def extract_features(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.vgg16.preprocess_input(img_array)
    features = base_model.predict(img_array, verbose=0)
    return features.flatten()


# ===============================
# VISUALIZE: IMAGE → ABSTRACT NUMBERS
# ===============================

# NOTE: train_wide_df must be loaded in your main script before calling this.
# This block assumes train_wide_df is available in the current namespace.

def visualize_image_to_features(train_wide_df, n_samples=3):
    """
    For n_samples random images, plot the original image side-by-side
    with a heatmap of its VGG16 feature vector (4096 dimensions).
    """
    sample_img_paths = train_wide_df['image_path'].sample(n_samples).tolist()

    plt.figure(figsize=(20, 5 * n_samples))

    for i, row in enumerate(sample_img_paths):
        full_path = os.path.join(BASE_PATH, row)

        # --- Original image ---
        plt.subplot(n_samples, 2, 2 * i + 1)
        img = image.load_img(full_path, target_size=(IMG_SIZE, IMG_SIZE))
        plt.imshow(img)
        plt.title(f"Original Image\n{row}", fontsize=10)
        plt.axis('off')

        # --- Feature vector heatmap ---
        sample_features = extract_features(full_path)

        plt.subplot(n_samples, 2, 2 * i + 2)
        sns.heatmap(
            [sample_features],
            cmap='viridis',
            cbar=True,
            yticklabels=False
        )
        plt.title(f"VGG16 Representation\n(1×{len(sample_features)} Vector)", fontsize=10)
        plt.xlabel(f"Feature Index (0–{len(sample_features) - 1})")

    plt.suptitle(
        f"The raw image has been transformed into {len(sample_features)} abstract numbers!",
        fontsize=14,
        fontweight='bold',
        y=1.01
    )

    plt.tight_layout()

    save_path = os.path.join(BASE_PATH, "image_to_4096_abstract_numbers.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")

    plt.show()

    print(f"The raw image has been transformed into {len(sample_features)} abstract numbers!")


# ===============================
# ENTRY POINT (standalone usage)
# ===============================

if __name__ == "__main__":
    import pandas as pd

    TRAIN_CSV_PATH = os.path.join(BASE_PATH, 'train.csv')
    train_df = pd.read_csv(TRAIN_CSV_PATH)

    train_wide_df = train_df.pivot(
        index='image_path',
        columns='target_name',
        values='target'
    ).reset_index()

    visualize_image_to_features(train_wide_df, n_samples=3)
