import numpy as np
import tensorflow as tf

from data_loader import load_dataset
from analysis import analyze_results, TrainingProgressLogger
from train_utils import (
    create_new_version_dir,
    save_final_model,
    compute_balanced_class_weights,
    cosine_warmup_scheduler
)

# ============================================
# ⚙️ קונפיגורציה בסיסית
# ============================================

layers = tf.keras.layers
models = tf.keras.models

IMG_SIZE = (299, 299)
BATCH_SIZE = 32

EPOCHS_STAGE1 = 20
EPOCHS_STAGE2 = 8

VALID_GENRES = ["goa", "psy", "dark"]

np.random.seed(42)
tf.random.set_seed(42)


# ============================================
# 🧠 בניית המודל (EfficientNetB0 + GRU + Attention)
# ============================================

def build_model(num_classes):
    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        input_shape=(299, 299, 3),
        weights="imagenet"
    )
    base.trainable = False

    # ----- תמונה -----
    img_input = layers.Input(shape=(299, 299, 3), name="image_input")
    x = layers.RandomFlip("horizontal")(img_input)
    x = layers.RandomRotation(0.05)(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    img_vec = layers.Dense(128, activation="relu")(x)

    # ----- embedding + Attention -----
    emb_input = layers.Input(shape=(10, 68), name="embedding_input")
    e_seq = layers.GRU(128, return_sequences=True)(emb_input)
    attn = layers.Attention()([e_seq, e_seq])
    e = layers.GRU(64)(attn)
    e = layers.Dropout(0.3)(e)
    emb_vec = layers.Dense(64, activation="relu")(e)

    # ----- שילוב -----
    combined = layers.Concatenate()([img_vec, emb_vec])
    x = layers.Dense(128, activation="relu")(combined)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=[img_input, emb_input], outputs=out)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"]
    )

    return model


# ============================================
# 🏃‍♂️ אימון המודל
# ============================================

def train_model():
    print("============== Training (Model vX) ==============")

    X_img_train, X_emb_train, y_train, y_idx_train, X_img_val, X_emb_val, y_val = load_dataset()

    # class weights
    class_weights = compute_balanced_class_weights(y_idx_train)

    # ---------- שלב 1 ----------
    print("\n==== Stage 1 — Base Training ====\n")

    model = build_model(len(VALID_GENRES))
    progress_log = TrainingProgressLogger("training_progress.txt")

    lr_cb1 = cosine_warmup_scheduler(1e-4, EPOCHS_STAGE1, warmup_epochs=3)

    history1 = model.fit(
        [X_img_train, X_emb_train], y_train,
        validation_data=([X_img_val, X_emb_val], y_val),
        epochs=EPOCHS_STAGE1,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=[
            lr_cb1,
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=4, restore_best_weights=True
            ),
            progress_log
        ]
    )

    # ---------- שלב 2 ----------
    print("\n==== Stage 2 — Fine Tuning ====\n")

    base = model.get_layer("efficientnetb0")
    fine_at = len(base.layers) - 20

    for l in base.layers[:fine_at]:
        l.trainable = False
    for l in base.layers[fine_at:]:
        l.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"]
    )

    lr_cb2 = cosine_warmup_scheduler(1e-5, EPOCHS_STAGE2, warmup_epochs=2)

    history2 = model.fit(
        [X_img_train, X_emb_train], y_train,
        validation_data=([X_img_val, X_emb_val], y_val),
        epochs=EPOCHS_STAGE2,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=[
            lr_cb2,
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=3, restore_best_weights=True
            ),
            progress_log
        ]
    )

    # איחוד היסטוריות
    history = {
        k: history1.history.get(k, []) + history2.history.get(k, [])
        for k in set(history1.history.keys()).union(history2.history.keys())
    }

    # ---------- שמירת גרסה ----------
    version_dir = create_new_version_dir()
    save_final_model(model, version_dir)

    analyze_results(
        model=model,
        history_dict=history,
        X_img_val=X_img_val,
        X_emb_val=X_emb_val,
        y_val=y_val,
        version_dir=version_dir
    )

    print("\n✔ DONE — Final Model Saved\n")


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    train_model()
