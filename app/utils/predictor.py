"""
predictor.py

Loads trained CNN model and performs
thermal solar panel defect prediction.
"""

import cv2
import joblib
import numpy as np

from tensorflow.keras.models import load_model

# ── Load artefacts ───────────────────────────────────────────

model = load_model("models/cnn_model.h5")

encoder = joblib.load("models/label_encoder.pkl")

IMG_SIZE = 128

# ── Prediction Function ──────────────────────────────────────

def predict_image(image_path):

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Invalid image file")

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    image = image.astype("float32") / 255.0

    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image, verbose=0)

    predicted_index = np.argmax(prediction)

    confidence = float(np.max(prediction))

    label = encoder.inverse_transform([predicted_index])[0]

    probabilities = {
        class_name: float(prob)
        for class_name, prob in zip(encoder.classes_, prediction[0])
    }

    return {
        "label": label,
        "confidence": confidence,
        "probabilities": probabilities
    }