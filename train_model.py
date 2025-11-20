import os
import time
import numpy as np
import tensorflow as tf
from pathlib import Path
import shutil
from sklearn.utils.class_weight import compute_class_weight

from data_loader import load_dataset
from analysis import analyze_results, TrainingProgressLogger

# ============================================
# ⚙️ חלק 0 — קונפיגורציה בסיסית וזרעים
# ============================================

layers = tf.keras.layers
models = tf.keras.models

MODELS_DIR = "models"
LATEST_MODEL = os.path.join(MODELS_DIR, "latest.h5")

IMG_SIZE = (299, 299)
BATCH_SIZE = 16

# מספר אפוקים
EPOCHS_STAGE1 = 20
EPOCHS_STAGE2 = 8

VALID_GENRES = ["goa", "psy", "dark"]

# זרעים לשחזוריות
np.random.seed(42)
tf.random.set_seed(42)


# ============================================
# 🧠 חלק 1 — בניית המודל (EfficientNetB0 + GRU)
# ============================================

def build_model(num_classes):
    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        weights="imagenet"
    )
    base.trainable = False

    img_input = layers.Input(shape=(299, 299, 3), name="image_input")
    x = layers.RandomFlip("horizontal")(img_input)
    x = layers.RandomRotation(0.05)(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    img_vec = layers.Dense(128, activation="relu")(x)

    emb_input = layers.Input(shape=(10, 68), name="embedding_input")
    e = layers.GRU(128, return_sequences=True)(emb_input)
    e = layers.GRU(64)(e)
    e = layers.Dropout(0.3)(e)
    emb_vec = layers.Dense(64, activation="relu")(e)

    combined = layers.Concatenate()([img_vec, emb_vec])
    x = layers.Dense(128, activation="relu")(combined)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(
        inputs=[img_input, emb_input],
        outputs=out,
        name="TranceCRNN_EfficientNet"
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================
# 💾 חלק 2 — שמירת מודלים
# ============================================

def save_base_model(model):
    os.makedirs(MODELS_DIR, exist_ok=True)

    versions = [
        int(f.name.replace("v", "")) for f in Path(MODELS_DIR).glob("v*")
        if f.is_dir() and f.name.replace("v", "").isdigit()
    ]
    next_v = max(versions) + 1 if versions else 1

    version_dir = Path(MODELS_DIR) / f"v{next_v}"
    version_dir.mkdir(exist_ok=True)

    path = version_dir / "model.h5"
    model.save(path)
    shutil.copy(path, LATEST_MODEL)

    print(f"✔ Saved BASE model → {path}")
    print(f"✔ Updated latest.h5 → {LATEST_MODEL}")

    return version_dir


def save_finetuned_model(model, version_dir):
    finetune_dir = Path(str(version_dir) + "_finetune")
    finetune_dir.mkdir(exist_ok=True)

    path = finetune_dir / "model.h5"
    model.save(path)
    shutil.copy(path, LATEST_MODEL)

    print(f"✔ Saved FINETUNED model → {path}")
    print(f"✔ latest.h5 updated")

    return finetune_dir


# ============================================
# 🏃‍♂️ חלק 3 — אימון המודל
# ============================================

def train_model():
    print("============== Training (Model v5) ==============")

    # טעינת דאטה מ־cache
    (
        X_img_train, X_emb_train, y_train, y_train_idx,
        X_img_val, X_emb_val, y_val
    ) = load_dataset()

    # חישוב class_weights
    cw = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train_idx),
        y=y_train_idx
    )
    class_weights = {i: float(w) for i, w in enumerate(cw)}
    print("Class Weights:", class_weights)

    # -------------------- Stage 1 --------------------
    print("\n==== Stage 1 — Base Training ====\n")
    model = build_model(len(VALID_GENRES))

    progress_log = TrainingProgressLogger("training_progress.txt")

    start_t1 = time.time()

    history1 = model.fit(
        [X_img_train, X_emb_train], y_train,
        validation_data=([X_img_val, X_emb_val], y_val),
        epochs=EPOCHS_STAGE1,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=4,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=2,
                min_lr=1e-6
            ),
            progress_log
        ],
        verbose=1
    )

    time1 = time.time() - start_t1
    base_dir = save_base_model(model)

    # -------------------- Stage 2 — Fine Tuning --------------------
    print("\n==== Stage 2 — Fine Tuning ====\n")

    base = model.get_layer("efficientnetb0")
    fine_tune_at = len(base.layers) - 20

    for layer in base.layers[:fine_tune_at]:
        layer.trainable = False
    for layer in base.layers[fine_tune_at:]:
        layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    start_t2 = time.time()

    history2 = model.fit(
        [X_img_train, X_emb_train], y_train,
        validation_data=([X_img_val, X_emb_val], y_val),
        epochs=EPOCHS_STAGE2,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=3,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=1,
                min_lr=1e-6
            ),
            progress_log
        ],
        verbose=1
    )

    time2 = time.time() - start_t2

    # איחוד היסטוריה
    history = {
        k: history1.history.get(k, []) + history2.history.get(k, [])
        for k in set(history1.history.keys()).union(history2.history.keys())
    }

    # שמירת מודל FT
    finetune_dir = save_finetuned_model(model, base_dir)

    # ניתוח תוצאות
    from analysis import analyze_results
    analyze_results(
        model=model,
        history_dict=history,
        X_img_val=X_img_val,
        X_emb_val=X_emb_val,
        y_val=y_val,
        version_dir=finetune_dir
    )

    print("\n✔ DONE — Full Training Completed (v5)\n")


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    train_model()
