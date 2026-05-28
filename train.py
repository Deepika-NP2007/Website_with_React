"""
Aksha_AI — Model Training Script
Fine-tune MobileNetV2 on your skin disease image dataset.

Dataset folder structure expected:
    dataset/
      train/
        Acne & Pimples/   ← folder name must match DISEASE_CLASSES in app.py
        Eczema/
        Normal Skin/
        ...
      val/
        Acne & Pimples/
        ...

Usage:
    python train.py --dataset ./dataset --epochs 20 --batch_size 32
"""

import argparse
import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

IMG_SIZE = (224, 224)
MODEL_SAVE_PATH = "models/aksha_skin_model.h5"
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)


def build_model(num_classes: int, fine_tune_at: int = 100) -> Model:
    """
    Build a MobileNetV2 transfer learning model.

    Phase 1: Train only the top layers (base frozen).
    Phase 2 (fine-tune): Unfreeze layers after `fine_tune_at` and train
                         with a low learning rate.
    """
    base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base.trainable = False  # Phase 1: freeze all

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.2)(x)
    out = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base.input, outputs=out)
    return model, base


def get_data_generators(dataset_dir: str, batch_size: int):
    """Create augmented train and validation data generators."""

    train_gen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        horizontal_flip=True,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2],
        shear_range=0.1,
    )

    val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_data = train_gen.flow_from_directory(
        os.path.join(dataset_dir, "train"),
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
    )

    val_data = val_gen.flow_from_directory(
        os.path.join(dataset_dir, "val"),
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    return train_data, val_data


def plot_history(history, save_path="training_history.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history["accuracy"], label="Train Acc")
    axes[0].plot(history.history["val_accuracy"], label="Val Acc")
    axes[0].set_title("Accuracy")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history.history["loss"], label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title("Loss")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training plots saved to {save_path}")


def train(dataset_dir: str, epochs: int, batch_size: int, fine_tune: bool):
    train_data, val_data = get_data_generators(dataset_dir, batch_size)
    num_classes = len(train_data.class_indices)
    print(f"\nDetected {num_classes} classes: {list(train_data.class_indices.keys())}\n")

    model, base = build_model(num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor="val_accuracy", verbose=1),
        EarlyStopping(patience=8, restore_best_weights=True, monitor="val_accuracy"),
        ReduceLROnPlateau(factor=0.3, patience=4, min_lr=1e-7, monitor="val_loss", verbose=1),
        TensorBoard(log_dir="logs/phase1"),
    ]

    # ── Phase 1: Train top layers ─────────────────────────────────────────────
    print("\n=== Phase 1: Training top layers (base frozen) ===\n")
    history1 = model.fit(
        train_data,
        epochs=epochs,
        validation_data=val_data,
        callbacks=callbacks,
    )

    # ── Phase 2: Fine-tune ────────────────────────────────────────────────────
    if fine_tune:
        print("\n=== Phase 2: Fine-tuning (unfreezing top 30 base layers) ===\n")
        base.trainable = True
        for layer in base.layers[:-30]:
            layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        callbacks[3] = TensorBoard(log_dir="logs/phase2")
        history2 = model.fit(
            train_data,
            epochs=epochs // 2,
            validation_data=val_data,
            callbacks=callbacks,
        )

        # Merge histories for plotting
        for k in history1.history:
            if k in history2.history:
                history1.history[k].extend(history2.history[k])

    model.save(MODEL_SAVE_PATH)
    print(f"\nModel saved to {MODEL_SAVE_PATH}")
    plot_history(history1)

    # Print class index mapping
    print("\nClass index mapping (must match DISEASE_CLASSES in app.py):")
    for cls, idx in sorted(train_data.class_indices.items(), key=lambda x: x[1]):
        print(f"  {idx}: {cls}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Aksha_AI skin model")
    parser.add_argument("--dataset", default="./dataset", help="Path to dataset folder")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--fine_tune", action="store_true", default=True, help="Enable fine-tuning phase 2")
    args = parser.parse_args()

    train(args.dataset, args.epochs, args.batch_size, args.fine_tune)
