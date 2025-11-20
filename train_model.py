import os
import time
import numpy as np
import tensorflow as tf
from PIL import Image
from pathlib import Path
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

layers = tf.keras.layers
models = tf.keras.models

DATA_DIR = "data"
MODELS_DIR = "models"
LATEST_MODEL = os.path.join(MODELS_DIR, "latest.h5")

IMG_SIZE = (299, 299)
BATCH_SIZE = 16
EPOCHS = 20

VALID_GENRES = ["goa", "psy", "dark"]


# -------------------------------------------------
# LOAD DATA (SAFE – no tf.numpy_function)
# -------------------------------------------------
def load_dataset():
    img_list = []
    emb_list = []
    labels = []

    for genre in VALID_GENRES:
        genre_path = Path(DATA_DIR) / genre
        if not genre_path.exists():
            continue

        for root, dirs, files in os.walk(genre_path):
            for f in files:
                if f.endswith(".png"):
                    img_path = os.path.join(root, f)
                    emb_path = img_path.replace(".png", ".npy")
                    if not os.path.exists(emb_path):
                        continue

                    # Load image
                    img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
                    img = np.array(img, dtype=np.float32) / 255.0

                    # Load embedding
                    emb = np.load(emb_path).astype(np.float32)

                    img_list.append(img)
                    emb_list.append(emb)
                    labels.append(genre)

    print(f"✔ Loaded {len(img_list)} samples")

    # Convert to numpy
    X_img = np.array(img_list, dtype=np.float32)
    X_emb = np.array(emb_list, dtype=np.float32)

    # Labels
    genre_to_idx = {g: i for i, g in enumerate(VALID_GENRES)}
    y_idx = np.array([genre_to_idx[g] for g in labels], dtype=np.int32)
    y = tf.keras.utils.to_categorical(y_idx, num_classes=len(VALID_GENRES))

    # Shuffle
    idx = np.arange(len(X_img))
    np.random.shuffle(idx)
    X_img = X_img[idx]
    X_emb = X_emb[idx]
    y = y[idx]
    y_idx = y_idx[idx]

    # Split 80/20
    val_size = int(0.2 * len(X_img))

    X_img_train = X_img[val_size:]
    X_emb_train = X_emb[val_size:]
    y_train = y[val_size:]
    y_train_idx = y_idx[val_size:]

    X_img_val = X_img[:val_size]
    X_emb_val = X_emb[:val_size]
    y_val = y[:val_size]

    return (
        X_img_train, X_emb_train, y_train, y_train_idx,
        X_img_val, X_emb_val, y_val
    )


# -------------------------------------------------
# MODEL
# -------------------------------------------------
def build_model(num_classes):

    # Image input
    img_input = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(img_input)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)

    # CRNN reshape
    shape = x.shape
    time_steps = shape[2]
    features = shape[1] * shape[3]

    x = layers.Reshape((time_steps, features))(x)
    x = layers.GRU(128, return_sequences=False)(x)
    x = layers.Dropout(0.3)(x)

    img_vec = x

    # Embedding input (length = 58)
    emb_input = layers.Input(shape=(58,))
    e = layers.Dense(64, activation="relu")(emb_input)
    e = layers.Dense(32, activation="relu")(e)
    emb_vec = e

    # Merge
    combined = layers.Concatenate()([img_vec, emb_vec])

    out = layers.Dense(64, activation="relu")(combined)
    out = layers.Dense(num_classes, activation="softmax")(out)

    model = models.Model(inputs=[img_input, emb_input], outputs=out)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()
    return model


# -------------------------------------------------
# SAVE MODEL
# -------------------------------------------------
def save_versioned_model(model):
    os.makedirs(MODELS_DIR, exist_ok=True)

    versions = [
        int(f.name.replace("v", ""))
        for f in Path(MODELS_DIR).glob("v*")
        if f.is_dir() and f.name.replace("v", "").isdigit()
    ]
    next_v = (max(versions) + 1) if versions else 1

    version_dir = Path(MODELS_DIR) / f"v{next_v}"
    version_dir.mkdir(parents=True, exist_ok=True)

    model_path = version_dir / "model.h5"
    model.save(model_path)
    shutil.copy(model_path, LATEST_MODEL)

    return version_dir


# -------------------------------------------------
# ANALYSIS
# -------------------------------------------------
def analyze_results(model, history, X_img_val, X_emb_val, y_val, version_dir):
    analysis_dir = version_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    y_true = np.argmax(y_val, axis=1)
    y_pred = np.argmax(model.predict([X_img_val, X_emb_val]), axis=1)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=VALID_GENRES,
                yticklabels=VALID_GENRES)
    plt.savefig(analysis_dir / "confusion_matrix.png")
    plt.close()

    report = classification_report(
        y_true, y_pred, target_names=VALID_GENRES, digits=3
    )
    with open(analysis_dir / "report.txt", "w") as f:
        f.write(report)

    plt.plot(history.history["accuracy"], label="Train")
    plt.plot(history.history["val_accuracy"], label="Val")
    plt.legend(); plt.grid()
    plt.savefig(analysis_dir / "accuracy.png")
    plt.close()

    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Val")
    plt.legend(); plt.grid()
    plt.savefig(analysis_dir / "loss.png")
    plt.close()


# -------------------------------------------------
# TRAINING
# -------------------------------------------------
def train_model():
    print("============== Training ==============")

    X_img_train, X_emb_train, y_train, y_train_idx, \
    X_img_val, X_emb_val, y_val = load_dataset()

    print(f"Train samples = {len(X_img_train)}  |  Val = {len(X_img_val)}")

    cw = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train_idx),
        y=y_train_idx
    )
    class_weights = {i: float(w) for i, w in enumerate(cw)}
    print("Class weights:", class_weights)

    model = build_model(num_classes=len(VALID_GENRES))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=4, restore_best_weights=True, monitor="val_loss"
        )
    ]

    history = model.fit(
        [X_img_train, X_emb_train], y_train,
        validation_data=([X_img_val, X_emb_val], y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks
    )

    version_dir = save_versioned_model(model)
    analyze_results(model, history, X_img_val, X_emb_val, y_val, version_dir)

    print("\n✔ DONE\n")


if __name__ == "__main__":
    train_model()
