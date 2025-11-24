# ===============================================
# analysis.py – FULL ANALYSIS MODULE
# Supports EMBEDDING / META / BOTH dynamically
# ===============================================

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

from config import (
    VALID_GENRES,
    ENABLE_CONFUSION_MATRIX,
    ENABLE_TRAIN_PLOTS,
    ENABLE_GRADIENT_SENSITIVITY,
    ENABLE_INTEGRATED_GRADIENTS,
    SAVE_REPORT,
    SAVE_INPUT_IMPORTANCE,
    IG_STEPS,
    USE_EMBEDDING,
    USE_META,
)


# ============================================================
# 1) בסיס – גרפים ודו"ח
# ============================================================

def plot_confusion_matrix(y_true, y_pred, out_path: Path):
    """מצייר ושומר מטריצת בלבול."""
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=VALID_GENRES,
        yticklabels=VALID_GENRES,
        cmap="Blues",
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_accuracy_loss(history: dict, out_path: Path):
    """מצייר Accuracy + Loss לפי היסטוריית האימון."""
    if not history:
        return

    epochs = range(1, len(history.get("accuracy", [])) + 1)

    plt.figure(figsize=(10, 4))
    if "accuracy" in history:
        plt.plot(epochs, history["accuracy"], label="Train Acc")
    if "val_accuracy" in history:
        plt.plot(epochs, history["val_accuracy"], "--", label="Val Acc")
    if "loss" in history:
        plt.plot(epochs, history["loss"], label="Train Loss")
    if "val_loss" in history:
        plt.plot(epochs, history["val_loss"], "--", label="Val Loss")

    plt.xlabel("Epoch")
    plt.title("Accuracy & Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def write_report(y_true, y_pred, out_path: Path):
    """שומר classification_report לטקסט."""
    text = classification_report(
        y_true, y_pred, target_names=VALID_GENRES, digits=3
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)


# ============================================================
# 2) Gradient Sensitivity (רגישות לגרדיאנט)
# ============================================================

def gradient_sensitivity(model, X_emb, X_meta, y_true):
    """
    מחשב חשיבות גרדיאנט בהתאם לקונפיג:
    - אם רק EMB → מחשב רק לאמבדינג
    - אם רק META → מחשב רק למטא
    - אם שניהם → מחשב לשניהם בנפרד

    מחזיר:
        imp_emb (float or None), imp_meta (float or None)
    """
    # לוקחים עד 64 דוגמאות כדי לא לפוצץ זמן/זיכרון
    if X_emb is not None:
        X_emb = tf.convert_to_tensor(X_emb[:64])
    if X_meta is not None:
        X_meta = tf.convert_to_tensor(X_meta[:64])

    y_true = tf.convert_to_tensor(y_true[:64])

    # בחירת input בהתאם לקונפיג
    if USE_EMBEDDING and USE_META:
        inputs = [X_emb, X_meta]
    elif USE_EMBEDDING:
        inputs = X_emb
    elif USE_META:
        inputs = X_meta
    else:
        raise ValueError("❌ Neither EMBEDDING nor META is enabled in config")

    with tf.GradientTape() as tape:
        tape.watch(inputs)
        preds = model(inputs, training=False)
        loss = tf.keras.losses.categorical_crossentropy(y_true, preds)

    grads = tape.gradient(loss, inputs)

    # שני אינפוטים
    if USE_EMBEDDING and USE_META:
        imp_emb = tf.reduce_mean(tf.abs(grads[0])).numpy()
        imp_meta = tf.reduce_mean(tf.abs(grads[1])).numpy()
        return imp_emb, imp_meta

    # אינפוט יחיד
    return tf.reduce_mean(tf.abs(grads)).numpy(), None


# ============================================================
# 3) Integrated Gradients – תומך בכניסה אחת או שתיים
# ============================================================

# ============================================================
# 4) עזר – חיזוי על סט הוולידציה
# ============================================================

def _predict_validation(model, X_emb_val, X_meta_val):
    """בוחר איזה אינפוט להעביר למודל לפי הקונפיג."""
    if USE_EMBEDDING and not USE_META:
        return model.predict(X_emb_val, verbose=0)

    if USE_META and not USE_EMBEDDING:
        return model.predict(X_meta_val, verbose=0)

    if USE_EMBEDDING and USE_META:
        return model.predict([X_emb_val, X_meta_val], verbose=0)

    raise ValueError("ERROR: No active inputs (embedding/meta).")


def _save_confusion(y_true, y_pred, path: Path):
    if ENABLE_CONFUSION_MATRIX:
        plot_confusion_matrix(y_true, y_pred, path)


def _save_accuracy_loss(history_dict, path: Path):
    if ENABLE_TRAIN_PLOTS:
        plot_accuracy_loss(history_dict, path)


def _save_report(y_true, y_pred, path: Path):
    if SAVE_REPORT:
        write_report(y_true, y_pred, path)


# ============================================================
# 5) Gradient Sensitivity – גרפים + טקסט
# ============================================================

def _save_input_importance(model, X_emb_val, X_meta_val, y_val, analysis_dir: Path):
    """שומר רגישות גרדיאנטים + טקסט השוואה."""
    if not (SAVE_INPUT_IMPORTANCE and ENABLE_GRADIENT_SENSITIVITY):
        return

    imp_text_lines = []

    # --- Embedding Only ---
    if USE_EMBEDDING and not USE_META:
        imp_emb, _ = gradient_sensitivity(
            model,
            X_emb_val[:64],
            None,
            y_val[:64],
        )
        imp_text_lines.append(f"Embedding importance: {imp_emb:.6f}")

    # --- Meta Only ---
    if USE_META and not USE_EMBEDDING:
        _, imp_meta = gradient_sensitivity(
            model,
            None,
            X_meta_val[:64],
            y_val[:64],
        )
        imp_text_lines.append(f"Meta importance: {imp_meta:.6f}")

    # --- Both Inputs ---
    if USE_EMBEDDING and USE_META:
        imp_emb, imp_meta = gradient_sensitivity(
            model,
            X_emb_val[:64],
            X_meta_val[:64],
            y_val[:64],
        )


        better = "Embedding" if imp_emb > imp_meta else "Meta"
        imp_text_lines.append(f"Embedding: {imp_emb:.6f}")
        imp_text_lines.append(f"Meta:      {imp_meta:.6f}")
        imp_text_lines.append(f"Conclusion: {better} input is more influential.")

    # כתיבה לקובץ טקסט (אם יש מה לכתוב)
    if imp_text_lines:
        with open(analysis_dir / "input_importance.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(imp_text_lines))


# ============================================================
# 6) Integrated Gradients – גרף לוקטור + מטריצה
# ============================================================
# ============================================================
# 7) פונקציה ראשית – analyze_results
# ============================================================

def analyze_results(
    model,
    history_dict: dict,
    version_dir: Path,
    X_emb_val=None,
    X_meta_val=None,
    y_val=None,
):
    """
    מריץ את כל האנליזה עבור מודל נתון וסט ולידציה:
    - Confusion Matrix
    - Accuracy/Loss
    - classification_report
    - Gradient Sensitivity (Embedding/Meta)
    - Integrated Gradients (וקטור + מטריצה)
    """

    analysis_dir = version_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # y_true מתוך one-hot
    y_true = np.argmax(y_val, axis=1)

    # --- חיזוי ---
    preds = _predict_validation(model, X_emb_val, X_meta_val)
    y_pred = np.argmax(preds, axis=1)

    # --- 1) Confusion Matrix ---
    _save_confusion(y_true, y_pred, analysis_dir / "confusion_matrix.png")

    # --- 2) Accuracy/Loss ---
    _save_accuracy_loss(history_dict, analysis_dir / "accuracy_loss.png")

    # --- 3) Classification Report ---
    _save_report(y_true, y_pred, analysis_dir / "report.txt")

    # --- 4) Gradient Sensitivity ---
    _save_input_importance(model, X_emb_val, X_meta_val, y_val, analysis_dir)
