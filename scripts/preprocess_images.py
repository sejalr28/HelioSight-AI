"""
preprocess_images.py

Preprocesses real thermal solar panel images
from the Infrared Solar Modules dataset.

Tasks:
- Load metadata JSON
- Map labels to images
- Resize & normalize images
- Encode labels
- Split into train/test sets
- Save processed NumPy arrays
"""

import os
import json
import cv2
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# =========================================================
# PATHS
# =========================================================

DATASET_PATH = r"data/raw/2020-02-14_InfraredSolarModules/InfraredSolarModules"

IMAGE_FOLDER = os.path.join(DATASET_PATH, "images")

METADATA_FILE = os.path.join(DATASET_PATH, "module_metadata.json")

PROCESSED_PATH = "data/processed"

MODEL_PATH = "models"

os.makedirs(PROCESSED_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

# =========================================================
# LOAD METADATA
# =========================================================

print("\nLoading metadata JSON...")

with open(METADATA_FILE, "r") as file:
    metadata = json.load(file)

print(f"Metadata loaded successfully!")

# =========================================================
# CREATE DATAFRAME
# =========================================================

records = []

for image_id, details in metadata.items():

    label = details.get("anomaly_class", "No-Anomaly")

    image_name = f"{image_id}.jpg"

    image_path = os.path.join(IMAGE_FOLDER, image_name)

    if os.path.exists(image_path):

        records.append({
            "image_path": image_path,
            "label": label
        })

df = pd.DataFrame(records)

print(f"\nTotal Images Found: {len(df)}")

print("\nClass Distribution:")
print(df["label"].value_counts())

# =========================================================
# IMAGE PROCESSING
# =========================================================

IMG_SIZE = 128

X = []
y = []

print("\nProcessing thermal images...")

df = df.sample(8000, random_state=42)

for _, row in df.iterrows():

    image = cv2.imread(row["image_path"])

    if image is None:
        continue

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    image = image.astype("float32") / 255.0

    X.append(image)

    y.append(row["label"])

X = np.array(X)
y = np.array(y)

print(f"\nImages Processed Successfully: {len(X)}")

# =========================================================
# LABEL ENCODING
# =========================================================

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)

y_categorical = to_categorical(y_encoded)

# Save encoder
joblib.dump(encoder, "models/label_encoder.pkl")

print("\nClasses Detected:")
print(encoder.classes_)

# =========================================================
# TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_categorical,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

# =========================================================
# SAVE DATA
# =========================================================

np.save("data/processed/X_train.npy", X_train)
np.save("data/processed/X_test.npy", X_test)

np.save("data/processed/y_train.npy", y_train)
np.save("data/processed/y_test.npy", y_test)

print("\nProcessed datasets saved successfully!")

print(f"\nX_train Shape: {X_train.shape}")
print(f"X_test Shape : {X_test.shape}")

print("\nPreprocessing Complete!")