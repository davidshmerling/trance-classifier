import os
import shutil
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# 📦 קונפיגורציה בסיסית
# ============================================================

MODELS_DIR = "models"
LATEST_MODEL = os.path.join(MODELS_DIR, "latest.h5")


# ============================================================
# 📁 1 — יצירת תיקיית גרסה חדשה (v7, v8 …)
# ============================================================

def create_new_version_dir():
    """
    מחפש את כל תיקיות vX ויוצר את התיקייה הבאה.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    versions = []
    for folder in Path(MODELS_DIR).glob("v*"):
        if folder.is_dir():
            try:
                num = int(folder.name.replace("v", ""))
                versions.append(num)
            except ValueError:
                pass

    next_v = max(versions) + 1 if versions else 1

    version_dir = Path(MODELS_DIR) / f"v{next_v}"
    version_dir.mkdir(exist_ok=True)

    print(f"✔ Created version directory: {version_dir}")
    return version_dir



# ============================================================
# 💾 2 — שמירת מודל הסופי בלבד
# ============================================================

def save_final_model(model, version_dir):
    """
    שומר רק את מודל ה-Fine-Tuning בתוך vX/model.h5
    ומעדכן גם את latest.h5.
    """
    out_path = version_dir / "model.h5"
    model.save(out_path)

    # Update latest model
    shutil.copy(out_path, LATEST_MODEL)

    print(f"✔ Saved FINAL model → {out_path}")
    print(f"✔ Updated latest.h5 → {LATEST_MODEL}")
    return out_path



# ============================================================
# ⚖️ 3 — חישוב Class Weights מאוזנים
# ============================================================

def compute_balanced_class_weights(y_idx_train):
    """
    מקבל y_idx ומחשב משקלות לכל מחלקה.
    """
    cw = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_idx_train),
        y=y_idx_train
    )
    class_weights = {i: float(w) for i, w in enumerate(cw)}

    print("✔ Computed class weights:", class_weights)
    return class_weights



# ============================================================
# 🎛 4 — Cosine Warmup + Cosine Decay Scheduler
# ============================================================

def cosine_warmup_scheduler(initial_lr, total_epochs, warmup_epochs=3, min_lr=1e-6):
    """
    יוצר Learning Rate Scheduler:
      - warmup למספר אפוקים ראשון
      - מעבר ל-cosine decay
    """
    def scheduler(epoch, lr):
        # WARMUP
        if epoch < warmup_epochs:
            return initial_lr * float(epoch + 1) / warmup_epochs

        # COSINE DECAY
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))

        return min_lr + (initial_lr - min_lr) * cosine_decay

    return tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)



# ============================================================
# 📜 5 — כתיבת Training Log
# ============================================================

def write_training_log(path, info_dict):
    """
    מקבל מילון ויוצר training_log.txt מסודר.
    """
    with open(path, "w", encoding="utf-8") as f:
        for key, value in info_dict.items():
            f.write(f"{key}: {value}\n")

    print(f"✔ Training log saved → {path}")


import time
import tensorflow as tf

class TrainingLoggerCallback(tf.keras.callbacks.Callback):
    """
    שומר כל אפוק ב-training_log.txt כולל:
    epoch, accuracy, loss, val_accuracy, val_loss, lr וזמן.
    """

    def __init__(self, log_path):
        super().__init__()
        self.log_path = log_path

        # כותב כותרת
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write("=== TRAINING LOG ===\n")

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start_time
        logs = logs or {}

        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))

        text = (
            f"Epoch {epoch+1}:\n"
            f"  accuracy:      {logs.get('accuracy'):.4f}\n"
            f"  loss:          {logs.get('loss'):.4f}\n"
            f"  val_accuracy:  {logs.get('val_accuracy'):.4f}\n"
            f"  val_loss:      {logs.get('val_loss'):.4f}\n"
            f"  lr:            {lr:.6f}\n"
            f"  epoch_time:    {epoch_time:.2f} sec\n"
            "--------------------------------------\n"
        )

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(text)

        print(f"✔ Logged epoch {epoch+1} to {self.log_path}")
