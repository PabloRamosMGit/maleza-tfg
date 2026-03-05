# Load the datasets
import os
import pandas as pd

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

BASE_PATH = r'C:\Users\pablo\pablo-tfg-malezas\maleza-tfg\csiro-biomass'
TRAIN_CSV_PATH = os.path.join(BASE_PATH, 'train.csv')
TEST_CSV_PATH = os.path.join(BASE_PATH, 'test.csv')

df_train = pd.read_csv(TRAIN_CSV_PATH)
df_test = pd.read_csv(TEST_CSV_PATH)

# -----------------------
# Visualise some images
# -----------------------

sample_rows = df_train.sample(5)

plt.figure(figsize=(20,5))

for i, (idx, row) in enumerate(sample_rows.iterrows()):

    full_path = os.path.join(BASE_PATH, row['image_path'])

    img = Image.open(full_path)

    plt.subplot(1,5,i+1)
    plt.imshow(img)

    plt.title(f"""
sample_id: {row['sample_id']}
Species: {row['Species']}
Height_Ave_cm: {row['Height_Ave_cm']} cm
target: {row['target']:.1f} g
""", loc='left', fontsize=8)

    plt.axis('off')

plt.show()


# -----------------------
# Scatter plot
# -----------------------

sns.scatterplot(
    data=df_train,
    x='Height_Ave_cm',
    y='target',
    hue='target_name',
    alpha=0.5
)

plt.show()


# -----------------------
# Average biomass per class
# -----------------------

target_by_name = (
    df_train
    .groupby('target_name')['target']
    .mean()
    .reset_index()
)

print("\nAverage biomass by target_name:")
print(target_by_name)


df_submission = df_test.merge(
    target_by_name,
    on='target_name',
    how='left'
)[["sample_id","target"]]

print(df_submission)