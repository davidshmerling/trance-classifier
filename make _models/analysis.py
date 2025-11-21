# ===============================================
# analysis.py
# מודול אנליזה + לוג אימון עבור Trance Classifier
# ===============================================

import os
import io
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras.callbacks import Callback


# ===============================================
# קבועים גלובליים
# ===============================================
VALID_GENRES = ["goa", "psy", "dark", "none"]   # כולל none לאינפוט של 4 תוויות


# ===============================================
# 🔹 מטריצת בלבול
# ===============================================
def plot_confusion_matrix(y_true, y_pred, out_path: Path):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=VALID_GENRES,
        yticklabels=VALID_GENRES,
        cmap="Blues"
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ===============================================
# 🔹 גרף דיוק
# ===============================================
def plot_accuracy(history_dict, out_path: Path):
    if "accuracy" not in history_dict or "val_accuracy" not in history_dict:
        return

    plt.figure()
    plt.plot(history_dict["accuracy"], label="Train Accuracy")
    plt.plot(history_dict["val_accuracy"], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ===============================================
# 🔹 גרף הפסד (Loss)
# ===============================================
def plot_loss(history_dict, out_path: Path):
    if "loss" not in history_dict or "val_loss" not in history_dict:
        return

    plt.figure()
    plt.plot(history_dict["loss"], label="Train Loss")
    plt.plot(history_dict["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ===============================================
# 🔹 כתיבת דוח טקסטואלי
# ===============================================
def write_report(y_true, y_pred, out_path: Path):
    report = classification_report(
        y_true, y_pred,
        target_names=VALID_GENRES,
        digits=3
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)


# ===============================================
# 🔥 Callback — שמירת התקדמות אימון
# ===============================================
class TrainingProgressLogger(Callback):
    """
    שומר לוג קצר של ההתקדמות באימון:
    loss / accuracy / val_loss / val_accuracy לכל epoch.
    """

    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath
        self.buffer = io.StringIO()

    def on_train_begin(self, logs=None):
        self.buffer.write("=== Training Progress Log ===\n\n")

    def on_epoch_end(self, epoch, logs=None):
        line = f"Epoch {epoch+1}: "
        for key, val in logs.items():
            try:
                line += f"{key}={val:.4f}  "
            except:
                pass
        self.buffer.write(line + "\n")

    def on_train_end(self, logs=None):
        with open(self.filepath, "w", encoding="utf-8") as f:
            f.write(self.buffer.getvalue())


# ===============================================
# 🔥 פונקציה ראשית — אנליזה מלאה
# ===============================================
def analyze_results(model,
                    history_dict,
                    X_img_val,
                    X_emb_val,
                    X_vec4_val,
                    y_val,
                    version_dir: Path):

    analysis_dir = version_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------
    # חיזוי מלא על הולידציה
    # --------------------------
    y_true = np.argmax(y_val, axis=1)

    # 🔥 מודל עם 3 אינפוטים:
    y_pred_probs = model.predict(
        [X_img_val, X_emb_val, X_vec4_val],
        verbose=0
    )

    y_pred = np.argmax(y_pred_probs, axis=1)

    # --------------------------
    # שמירת גרפים
    # --------------------------
    plot_confusion_matrix(
        y_true, y_pred,
        analysis_dir / "confusion_matrix.png"
    )
    plot_accuracy(
        history_dict,
        analysis_dir / "accuracy.png"
    )
    plot_loss(
        history_dict,
        analysis_dir / "loss.png"
    )

    # --------------------------
    # שמירת דוח
    # --------------------------
    write_report(
        y_true,
        y_pred,
        analysis_dir / "report.txt"
    )

    print(f"✔ Analysis saved in {analysis_dir}")
