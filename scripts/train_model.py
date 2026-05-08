"""
train_model.py

Trains a deep learning image classifier on
thermal solar panel inspection images.

Uses:
- MobileNetV2 Transfer Learning
- Real Infrared Solar Dataset

Saves:
- CNN model
- metadata
- training graphs
"""

import os
import json
import joblib
import warnings
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import classification_report, accuracy_score

# ── Load datasets ──────────────────────────────────────────────────────────────

print("\nLoading processed image datasets...")

X_train = np.load("data/processed/X_train.npy")
X_test  = np.load("data/processed/X_test.npy")

y_train = np.load("data/processed/y_train.npy")
y_test  = np.load("data/processed/y_test.npy")

print(f"X_train Shape : {X_train.shape}")
print(f"X_test Shape  : {X_test.shape}")

# ── Load label encoder ─────────────────────────────────────────────────────────

le = joblib.load("models/label_encoder.pkl")

CLASS_NAMES = le.classes_

NUM_CLASSES = len(CLASS_NAMES)

print("\nClasses:")
print(CLASS_NAMES)

# ── Build model ────────────────────────────────────────────────────────────────

print("\nBuilding MobileNetV2 model...")

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(128, 128, 3)
)

base_model.trainable = False

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dropout(0.3)(x)

x = Dense(128, activation="relu")(x)

output = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

# ── Compile model ──────────────────────────────────────────────────────────────

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ── Callbacks ──────────────────────────────────────────────────────────────────

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

# ── Train model ────────────────────────────────────────────────────────────────

print("\nTraining deep learning model...")

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=32,
    callbacks=[early_stop]
)

# ── Evaluate model ─────────────────────────────────────────────────────────────

print("\nEvaluating model...")

loss, accuracy = model.evaluate(X_test, y_test)

print(f"\nTest Accuracy : {accuracy:.4f}")

# ── Predictions ────────────────────────────────────────────────────────────────

y_pred = model.predict(X_test)

y_pred_classes = np.argmax(y_pred, axis=1)

y_true = np.argmax(y_test, axis=1)

print("\nClassification Report:\n")

print(
    classification_report(
        y_true,
        y_pred_classes,
        target_names=CLASS_NAMES
    )
)

# ── Save model ─────────────────────────────────────────────────────────────────

os.makedirs("models", exist_ok=True)

model.save("models/cnn_model.h5")

# ── Save metadata ──────────────────────────────────────────────────────────────

meta = {
    "model": "MobileNetV2",
    "image_size": 128,
    "classes": CLASS_NAMES.tolist(),
    "test_accuracy": round(float(accuracy), 4),
}

with open("models/model_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

# ── Save training graphs ───────────────────────────────────────────────────────

plt.figure(figsize=(10, 5))

plt.plot(history.history["accuracy"], label="Train Accuracy")

plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.title("Training Accuracy")

plt.legend()

plt.savefig("models/training_accuracy.png")

plt.close()

plt.figure(figsize=(10, 5))

plt.plot(history.history["loss"], label="Train Loss")

plt.plot(history.history["val_loss"], label="Validation Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("Training Loss")

plt.legend()

plt.savefig("models/training_loss.png")

plt.close()

print("\nModel saved to models/cnn_model.h5")

print("Metadata saved to models/model_meta.json")

print("Training graphs saved successfully!")

print("\nDeep Learning Pipeline Complete!")