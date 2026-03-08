# For data manipulation
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# For machine learning model
import sklearn
from sklearn.ensemble import RandomForestRegressor

# For computer vision
import tensorflow as tf

print("Setup complete. TensorFlow version:", tf.__version__, "SciKit-LEarn version:", sklearn.__version__)

gpus = tf.config.list_physical_devices('GPU')

if gpus:
    print(f"GPU Detected: {gpus[0]}")
else:
    print("No GPU Detected. Training will be very slow. Please enable an accelerator.")