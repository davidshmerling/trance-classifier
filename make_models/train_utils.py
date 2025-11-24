import shutil
import time
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight

# 📌 שימוש בקונפיג בלבד
from config import (
    MODELS_DIR,
    LATEST_MODEL_NAME,
    CLASS_WEIGHT_MODE,
    EPOCHS,
    INIT_LR,
    MIN_LR,
    WARMUP_EPOCHS,
)


# ============================================================
# 📁 1️⃣ יצירת תיקיית מודל חדש (v1, v2, ...)
# ============================================================
def create_new_version_dir():
    Path(MODELS_DIR).mkdir(exist_ok=True)

    versions = [
        int(folder.name.replace("v", ""))
        for folder in Path(MODELS_DIR).glob("v*")
        if folder.is_dir() and folder.name[1:].isdigit()
    ]

    next_v = max(versions) + 1 if versions else 1
    version_dir = Path(MODELS_DIR) / f"v{next_v}"
    version_dir.mkdir(exist_ok=True)

    print(f"📂 Created version: {version_dir}")
    return version_dir


# ============================================================
# 💾 2️⃣ שמירת המודל הסופי
# ============================================================
def save_final_model(model, version_dir):
    out_path = version_dir / "model.h5"
    model.save(out_path)
    shutil.copy(out_path, Path(MODELS_DIR) / LATEST_MODEL_NAME)
    print(f"✔ Final model saved → {out_path}")
    print(f"✔ Updated latest model → {LATEST_MODEL_NAME}")
    return out_path

# ============================================================
# ⚖️ 3️⃣ חישוב Class Weights
# ============================================================
def compute_balanced_class_weights(y_idx_train):
    cw = compute_class_weight(
        class_weight=CLASS_WEIGHT_MODE,
        classes=np.unique(y_idx_train),
        y=y_idx_train
    )
    weights = {i: float(w) for i, w in enumerate(cw)}
    print("⚖ Class Weights:", weights)
    return weights


# ============================================================
# 📉 4️⃣ Learning Rate Scheduler
# ============================================================
def cosine_warmup_scheduler():
    def scheduler(epoch, lr):
        if epoch < WARMUP_EPOCHS:
            return INIT_LR * (epoch + 1) / WARMUP_EPOCHS

        progress = (epoch - WARMUP_EPOCHS) / max(1, EPOCHS - WARMUP_EPOCHS)
        cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
        return MIN_LR + (INIT_LR - MIN_LR) * cosine_decay

    print("📈 Using Cosine Warmup Scheduler")
    return tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)


# ============================================================
# 📝 5️⃣ Callbacks – שמירת לוג לכל אפוק
# ============================================================
class TrainingLoggerCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_path):
        super().__init__()
        self.log_path = log_path

        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write("=== TRAINING LOG ===\n")

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        epoch_time = time.time() - self.epoch_start

        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(
                f"Epoch {epoch+1}:\n"
                f"  accuracy:     {logs.get('accuracy', 0):.4f}\n"
                f"  loss:         {logs.get('loss', 0):.4f}\n"
                f"  val_accuracy: {logs.get('val_accuracy', 0):.4f}\n"
                f"  val_loss:     {logs.get('val_loss', 0):.4f}\n"
                f"  lr:           {lr:.6f}\n"
                f"  time:         {epoch_time:.2f} sec\n"
                "--------------------------------------\n"
            )

        print(f"📝 Logged epoch {epoch+1}")




from pathlib import Path
import inspect
import config

def save_config_snapshot(version_dir):
    config_text = inspect.getsource(config)
    with open(version_dir / "config.txt", "w", encoding="utf-8") as f:
        f.write(config_text)
    print(f"📎 Saved config snapshot → {version_dir / 'config_snapshot.txt'}")
