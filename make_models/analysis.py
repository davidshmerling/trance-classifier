# ===============================================
# analysis.py – Analysis utilities
# Produces confusion matrix, acc/loss curves,
# classification report and gradient sensitivity.
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
    SAVE_REPORT,
    SAVE_INPUT_IMPORTANCE,
    USE_EMBEDDING,
    USE_META,
)

# -----------------------------------------------
# Confusion Matrix
# -----------------------------------------------
def plot_confusion_matrix(y_true, y_pred, out_path: Path):
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=VALID_GENRES,
                yticklabels=VALID_GENRES,
                cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# -----------------------------------------------
# Accuracy & Loss curves
# -----------------------------------------------
def plot_accuracy_loss(history, out_path: Path):
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

# -----------------------------------------------
# Classification Report
# -----------------------------------------------
def write_report(y_true, y_pred, out_path: Path):
    text = classification_report(
        y_true, y_pred,
        target_names=VALID_GENRES,
        digits=3
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)

def _save_report(y_true, y_pred, path: Path):
    if SAVE_REPORT:
        write_report(y_true, y_pred, path)

# -----------------------------------------------
# Gradient Sensitivity
# -----------------------------------------------
def gradient_sensitivity(model, X_emb, X_meta, y_true):
    if X_emb is not None:
        X_emb = tf.convert_to_tensor(X_emb[:64])
    if X_meta is not None:
        X_meta = tf.convert_to_tensor(X_meta[:64])
    y_true = tf.convert_to_tensor(y_true[:64])

    if USE_EMBEDDING and USE_META:
        inputs = [X_emb, X_meta]
    elif USE_EMBEDDING:
        inputs = X_emb
    elif USE_META:
        inputs = X_meta
    else:
        raise ValueError("No active inputs.")

    with tf.GradientTape() as tape:
        tape.watch(inputs)
        preds = model(inputs, training=False)
        loss = tf.keras.losses.categorical_crossentropy(y_true, preds)

    grads = tape.gradient(loss, inputs)

    if USE_EMBEDDING and USE_META:
        return (
            float(tf.reduce_mean(tf.abs(grads[0]))),
            float(tf.reduce_mean(tf.abs(grads[1])))
        )

    return float(tf.reduce_mean(tf.abs(grads))), None

def _save_input_importance(model, X_emb, X_meta, y_val, analysis_dir: Path):
    if not (SAVE_INPUT_IMPORTANCE and ENABLE_GRADIENT_SENSITIVITY):
        return

    lines = []

    # Emb only
    if USE_EMBEDDING and not USE_META:
        imp, _ = gradient_sensitivity(model, X_emb, None, y_val)
        lines.append(f"Embedding importance: {imp:.6f}")

    # Meta only
    if USE_META and not USE_EMBEDDING:
        _, imp = gradient_sensitivity(model, None, X_meta, y_val)
        lines.append(f"Meta importance: {imp:.6f}")

    # Both
    if USE_EMBEDDING and USE_META:
        imp_emb, imp_meta = gradient_sensitivity(model, X_emb, X_meta, y_val)
        better = "Embedding" if imp_emb > imp_meta else "Meta"
        lines += [
            f"Embedding: {imp_emb:.6f}",
            f"Meta:      {imp_meta:.6f}",
            f"Conclusion: {better} is more influential."
        ]

    if lines:
        with open(analysis_dir / "input_importance.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

# -----------------------------------------------
# Select correct prediction input
# -----------------------------------------------
def _predict_validation(model, X_emb_val, X_meta_val):
    if USE_EMBEDDING and not USE_META:
        return model.predict(X_emb_val, verbose=0)
    if USE_META and not USE_EMBEDDING:
        return model.predict(X_meta_val, verbose=0)
    if USE_EMBEDDING and USE_META:
        return model.predict([X_emb_val, X_meta_val], verbose=0)
    raise ValueError("No active inputs.")


def _save_confusion(y_true, y_pred, path: Path):
    if ENABLE_CONFUSION_MATRIX:
        plot_confusion_matrix(y_true, y_pred, path)


def _save_accuracy_loss(history_dict, path: Path):
    if ENABLE_TRAIN_PLOTS:
        plot_accuracy_loss(history_dict, path)


def _save_report(y_true, y_pred, path: Path):
    if SAVE_REPORT:
        write_report(y_true, y_pred, path)


# -----------------------------------------------
# Main analysis pipeline
# -----------------------------------------------
def analyze_results(model, history_dict, version_dir: Path,
                    X_emb_val=None, X_meta_val=None, y_val=None):

    analysis_dir = version_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    y_true = np.argmax(y_val, axis=1)
    preds = _predict_validation(model, X_emb_val, X_meta_val)
    y_pred = np.argmax(preds, axis=1)

    _save_confusion(y_true, y_pred, analysis_dir / "confusion_matrix.png")
    _save_accuracy_loss(history_dict, analysis_dir / "accuracy_loss.png")
    _save_report(y_true, y_pred, analysis_dir / "report.txt")
    _save_input_importance(model, X_emb_val, X_meta_val, y_val, analysis_dir)
