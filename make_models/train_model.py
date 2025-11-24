# make_models/train_model.py
import tensorflow as tf
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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

# 📌 קונפיג
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
# 🧱 1. ענף ה־Embedding לפי EMBEDDING_MODEL_TYPE
# ============================================================
def build_embedding_branch(emb_input):
    """בונה את ענף ה־embedding לפי EMBEDDING_MODEL_TYPE מהקונפיג."""

    if EMBEDDING_MODEL_TYPE == "dense":
        x = layers.Flatten(name="emb_flatten")(emb_input)
        x = layers.Dense(
            DENSE_EMB_CONFIG["layer_1"],
            activation=COMBINED_HEAD_CONFIG["activation"],
            name="emb_dense_1",
        )(x)
        x = layers.Dropout(DENSE_EMB_CONFIG["dropout"], name="emb_dropout")(x)
        x = layers.Dense(
            DENSE_EMB_CONFIG["layer_2"],
            activation=COMBINED_HEAD_CONFIG["activation"],
            name="emb_dense_2",
        )(x)
        return x

    if EMBEDDING_MODEL_TYPE == "gru":
        gru = layers.GRU(
            GRU_CONFIG["units"],
            dropout=GRU_CONFIG["dropout"],
            return_sequences=False,
            name="emb_gru",
        )
        if GRU_CONFIG["bidirectional"]:
            x = layers.Bidirectional(gru, name="emb_bigru")(emb_input)
        else:
            x = gru(emb_input)
        return x

    if EMBEDDING_MODEL_TYPE == "crnn":
        # Conv1D על ציר הזמן → GRU
        x = layers.Conv1D(
            filters=CRNN_CONFIG["conv_filters"],
            kernel_size=CRNN_CONFIG["kernel_size"],
            padding="same",
            activation=COMBINED_HEAD_CONFIG["activation"],
            name="emb_conv1d",
        )(emb_input)
        x = layers.Dropout(CRNN_CONFIG["conv_dropout"], name="emb_conv_dropout")(x)

        gru = layers.GRU(
            CRNN_CONFIG["gru_units"],
            dropout=CRNN_CONFIG["gru_dropout"],
            return_sequences=False,
            name="emb_crnn_gru",
        )
        if CRNN_CONFIG["bidirectional"]:
            x = layers.Bidirectional(gru, name="emb_crnn_bigru")(x)
        else:
            x = gru(x)
        return x
    if EMBEDDING_MODEL_TYPE == "transformer":
        # Projection to d_model
        x = layers.Dense(
            TRANSFORMER_CONFIG["d_model"], name="emb_proj_d_model"
        )(emb_input)

        for i in range(TRANSFORMER_CONFIG["num_layers"]):
            # Multi-head self-attention
            attn_out = layers.MultiHeadAttention(
                num_heads=TRANSFORMER_CONFIG["num_heads"],
                key_dim=TRANSFORMER_CONFIG["d_model"],
                dropout=TRANSFORMER_CONFIG["dropout"],
                name=f"emb_mha_{i + 1}",
            )(x, x)
            x = layers.Add(name=f"emb_mha_add_{i + 1}")([x, attn_out])
            x = layers.LayerNormalization(name=f"emb_mha_ln_{i + 1}")(x)

            # Feed-forward
            ff = layers.Dense(
                TRANSFORMER_CONFIG["d_model"] * 2,
                activation=COMBINED_HEAD_CONFIG["activation"],
                name=f"emb_ffn_1_{i + 1}",
            )(x)
            ff = layers.Dense(
                TRANSFORMER_CONFIG["d_model"],
                name=f"emb_ffn_2_{i + 1}",
            )(ff)
            x = layers.Add(name=f"emb_ffn_add_{i + 1}")([x, ff])
            x = layers.LayerNormalization(name=f"emb_ffn_ln_{i + 1}")(x)

        x = layers.GlobalAveragePooling1D(name="emb_transformer_gap")(x)
        return x


    raise ValueError(f"❌ EMBEDDING_MODEL_TYPE לא מוכר: {EMBEDDING_MODEL_TYPE}")


# ============================================================
# 🧱 2. מודל מלא (Embedding + Meta + Head)
# ============================================================
def build_model(num_classes: int) -> tf.keras.Model:
    # --- קלט embedding ---
    emb_input = layers.Input(shape=EMB_SHAPE, name="embedding_input")
    x = build_embedding_branch(emb_input)

    inputs = [emb_input]

    # --- קלט META (אם פעיל) ---
    if USE_META and META_CONFIG.get("use", True):
        meta_input = layers.Input(shape=(META_VECTOR_LENGTH,), name="meta_input")
        m = layers.Dense(
            META_CONFIG["units"],
            activation=META_CONFIG["activation"],
            name="meta_dense",
        )(meta_input)
        m = layers.Lambda(lambda t: t * META_SCALE, name="meta_scaled")(m)
        x = layers.Concatenate(name="concat_emb_meta")([x, m])
        inputs.append(meta_input)

    # --- ראש משותף (Combined head) ---
    x = layers.Dense(
        COMBINED_HEAD_CONFIG["dense_1"],
        activation=COMBINED_HEAD_CONFIG["activation"],
        name="head_dense_1",
    )(x)
    x = layers.Dropout(COMBINED_HEAD_CONFIG["dropout"], name="head_dropout")(x)
    x = layers.Dense(
        COMBINED_HEAD_CONFIG["dense_2"],
        activation=COMBINED_HEAD_CONFIG["activation"],
        name="head_dense_2",
    )(x)

    output = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = models.Model(inputs=inputs, outputs=output, name="trance_classifier")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(INIT_LR),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=["accuracy"],
    )

    return model


# ============================================================
# 🏃‍♂️ 3. אימון המודל
# ============================================================
def _build_meta_scaler():
    """בוחר scaler לפי META_SCALER_TYPE מהקונפיג."""
    if not USE_META:
        return None
    if META_SCALER_TYPE == "standard":
        return StandardScaler()
    if META_SCALER_TYPE == "minmax":
        return MinMaxScaler()
    return None  # בלי נרמול


def train_model():
    print("============== Training Trance Classifier ==============")

    (
        X_emb_train,
        X_meta_train,
        y_train,
        y_idx_train,
        X_emb_val,
        X_meta_val,
        y_val,
    ) = load_dataset()

    # --- נרמול META (אם פעיל) ---
    scaler = _build_meta_scaler()
    if scaler is not None:
        X_meta_train = scaler.fit_transform(X_meta_train)
        X_meta_val = scaler.transform(X_meta_val)

    # --- class weights ---
    class_weights = compute_balanced_class_weights(y_idx_train)

    # --- בניית מודל ---
    model = build_model(len(VALID_GENRES))

    # --- תיקיית גרסה + לוגים+קונפיג טקסט ---
    version_dir = create_new_version_dir()
    analysis_dir = version_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    save_config_snapshot(analysis_dir)

    log_path = analysis_dir / "training_log.txt"
    log_cb = TrainingLoggerCallback(str(log_path))
    lr_cb = cosine_warmup_scheduler()

    # --- בחירת אינפוטים לפי USE_META ---
    if USE_META and META_CONFIG.get("use", True):
        train_inputs = [X_emb_train, X_meta_train]
        val_inputs = [X_emb_val, X_meta_val]
    else:
        train_inputs = X_emb_train
        val_inputs = X_emb_val

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

    history = history_obj.history

    # --- שמירת מודל אחרון + latest.h5 ---
    save_final_model(model, version_dir)

    # --- אנליזה (כולל META אם יש) ---
    analyze_results(
        model=model,
        history_dict=history,
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
