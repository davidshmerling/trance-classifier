# ===============================================
# train_model.py – Training pipeline for TranceClassifier
# ===============================================

import tensorflow as tf
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Local project modules
from data_loader import load_dataset
from analysis import analyze_results
from train_utils import (
    create_new_version_dir,
    save_final_model,
    compute_balanced_class_weights,
    cosine_warmup_scheduler,
    TrainingLoggerCallback,
    save_config_snapshot
)

# 📌 טעינת קונפיג – כל הפרמטרים במקום אחד
from config import (
    VALID_GENRES,
    EMB_SHAPE,
    META_VECTOR_LENGTH,
    USE_META,
    META_SCALE,
    META_SCALER_TYPE,
    EMBEDDING_MODEL_TYPE,
    DENSE_EMB_CONFIG,
    GRU_CONFIG,
    CRNN_CONFIG,
    TRANSFORMER_CONFIG,
    META_CONFIG,
    COMBINED_HEAD_CONFIG,
    BATCH_SIZE,
    EPOCHS,
    INIT_LR,
    MIN_LR,
    WARMUP_EPOCHS,
    LABEL_SMOOTHING,
    EARLY_STOP_PATIENCE,
    USE_EMBEDDING
)

layers = tf.keras.layers
models = tf.keras.models


# ============================================================
# 🧱 1. בניית ענף ה־Embedding לפי EMBEDDING_MODEL_TYPE
# ============================================================
def build_embedding_branch(emb_input):
    """בונה את ענף האמבדינג בהתאם לקונפיג – Dense / GRU / CRNN / Transformer."""

    if EMBEDDING_MODEL_TYPE == "dense":
        x = layers.Flatten(name="emb_flatten")(emb_input)
        x = layers.Dense(DENSE_EMB_CONFIG["layer_1"],
                         activation=COMBINED_HEAD_CONFIG["activation"],
                         name="emb_dense_1")(x)
        x = layers.Dropout(DENSE_EMB_CONFIG["dropout"])(x)
        x = layers.Dense(DENSE_EMB_CONFIG["layer_2"],
                         activation=COMBINED_HEAD_CONFIG["activation"],
                         name="emb_dense_2")(x)
        return x

    if EMBEDDING_MODEL_TYPE == "gru":
        gru = layers.GRU(GRU_CONFIG["units"],
                         dropout=GRU_CONFIG["dropout"],
                         return_sequences=False,
                         name="emb_gru")
        return layers.Bidirectional(gru, name="emb_bigru")(emb_input) \
            if GRU_CONFIG["bidirectional"] else gru(emb_input)

    if EMBEDDING_MODEL_TYPE == "crnn":
        x = layers.Conv1D(CRNN_CONFIG["conv_filters"],
                          CRNN_CONFIG["kernel_size"],
                          padding="same",
                          activation=COMBINED_HEAD_CONFIG["activation"],
                          name="emb_conv1d")(emb_input)
        x = layers.Dropout(CRNN_CONFIG["conv_dropout"])(x)
        gru = layers.GRU(CRNN_CONFIG["gru_units"],
                         dropout=CRNN_CONFIG["gru_dropout"],
                         return_sequences=False,
                         name="emb_crnn_gru")
        return layers.Bidirectional(gru, name="emb_crnn_bigru")(x) \
            if CRNN_CONFIG["bidirectional"] else gru(x)

    if EMBEDDING_MODEL_TYPE == "transformer":
        x = layers.Dense(TRANSFORMER_CONFIG["d_model"], name="emb_proj_d_model")(emb_input)

        for i in range(TRANSFORMER_CONFIG["num_layers"]):
            attn = layers.MultiHeadAttention(
                num_heads=TRANSFORMER_CONFIG["num_heads"],
                key_dim=TRANSFORMER_CONFIG["d_model"],
                dropout=TRANSFORMER_CONFIG["dropout"],
                name=f"emb_mha_{i + 1}"
            )(x, x)
            x = layers.LayerNormalization()(layers.Add()([x, attn]))

            ff = layers.Dense(TRANSFORMER_CONFIG["d_model"] * 2,
                              activation=COMBINED_HEAD_CONFIG["activation"])(x)
            ff = layers.Dense(TRANSFORMER_CONFIG["d_model"])(ff)
            x = layers.LayerNormalization()(layers.Add()([x, ff]))

        return layers.GlobalAveragePooling1D(name="emb_transformer_gap")(x)

    raise ValueError(f"❌ EMBEDDING_MODEL_TYPE לא חוקי: {EMBEDDING_MODEL_TYPE}")


# ============================================================
# 🧱 2. בניית המודל המלא
# ============================================================
def build_model(num_classes: int) -> tf.keras.Model:
    """בונה מודל מלא: Embedding → Meta (אופציונלי) → Head → Softmax."""

    emb_input = layers.Input(shape=EMB_SHAPE, name="embedding_input")
    x = build_embedding_branch(emb_input)
    inputs = [emb_input]

    if USE_META and META_CONFIG.get("use", True):
        meta_input = layers.Input(shape=(META_VECTOR_LENGTH,), name="meta_input")
        m = layers.Dense(META_CONFIG["units"],
                         activation=META_CONFIG["activation"])(meta_input)
        m = layers.Lambda(lambda t: t * META_SCALE)(m)
        x = layers.Concatenate()([x, m])
        inputs.append(meta_input)

    x = layers.Dense(COMBINED_HEAD_CONFIG["dense_1"],
                     activation=COMBINED_HEAD_CONFIG["activation"])(x)
    x = layers.Dropout(COMBINED_HEAD_CONFIG["dropout"])(x)
    x = layers.Dense(COMBINED_HEAD_CONFIG["dense_2"],
                     activation=COMBINED_HEAD_CONFIG["activation"])(x)

    output = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = models.Model(inputs=inputs, outputs=output, name="trance_classifier")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(INIT_LR),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=["accuracy"],
    )
    return model


# ============================================================
# 🏃‍♂️ 3. הרצת אימון מלא
# ============================================================
def _build_meta_scaler():
    """בניית scaler לפי META_SCALER_TYPE מהקונפיג."""
    if not USE_META:
        return None
    return StandardScaler() if META_SCALER_TYPE == "standard" else \
        MinMaxScaler() if META_SCALER_TYPE == "minmax" else None


def train_model():
    print("============== Training Trance Classifier ==============")

    X_emb_train, X_meta_train, y_train, y_idx_train, X_emb_val, X_meta_val, y_val = load_dataset()

    scaler = _build_meta_scaler()
    if scaler is not None:
        X_meta_train = scaler.fit_transform(X_meta_train)
        X_meta_val = scaler.transform(X_meta_val)

    class_weights = compute_balanced_class_weights(y_idx_train)
    model = build_model(len(VALID_GENRES))

    version_dir = create_new_version_dir()
    analysis_dir = version_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)

    save_config_snapshot(version_dir)

    log_path = analysis_dir / "training_log.txt"
    log_cb = TrainingLoggerCallback(str(log_path))
    lr_cb = cosine_warmup_scheduler()

    if USE_EMBEDDING and USE_META:
        train_inputs = [X_emb_train, X_meta_train]
        val_inputs = [X_emb_val, X_meta_val]
    elif USE_EMBEDDING and not USE_META:
        train_inputs = X_emb_train
        val_inputs = X_emb_val
    elif USE_META and not USE_EMBEDDING:
        train_inputs = X_meta_train
        val_inputs = X_meta_val
    else:
        raise ValueError("❌ No inputs active — both USE_EMBEDDING and USE_META are False")

    history_obj = model.fit(
        train_inputs,
        y_train,
        validation_data=(val_inputs, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=[
            lr_cb,
            log_cb,
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=EARLY_STOP_PATIENCE,
                restore_best_weights=True,
            ),
        ],
    )

    save_final_model(model, version_dir)

    analyze_results(
        model=model,
        history_dict=history_obj.history,
        version_dir=version_dir,
        X_emb_val=X_emb_val if USE_EMBEDDING else None,
        X_meta_val=X_meta_val if USE_META else None,
        y_val=y_val
    )

    print("\n✔ DONE — Model trained & saved successfully!\n")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    train_model()
