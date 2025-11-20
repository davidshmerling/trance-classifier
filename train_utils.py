import os
import numpy as np
import tensorflow as tf
from pathlib import Path
import shutil
from sklearn.utils.class_weight import compute_class_weight

# ============================================
# 📦 ניהול תיקיות וגרסאות
# ============================================

MODELS_DIR = "models"
LATEST_MODEL = os.path.join(MODELS_DIR, "latest.h5")


def create_new_version_dir():
    """
    יוצר תיקייה חדשה models/vX עבור המודל הבא.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    versions = [
        int(f.name.replace("v", ""))
        for f in Path(MODELS_DIR).glob("v*")
        if f.is_dir() and f.name.replace("v", "").isdigit()
    ]

    next_v = max(versions) + 1 if versions else 1

    version_dir = Path(MODELS_DIR) / f"v{next_v}"
    version_dir.mkdir(exist_ok=True)

    print(f"✔ Created version directory → {version_dir}")
    return version_dir


def save_final_model(model, version_dir):
    """
    שומר רק את מודל ה-Fine-Tuning בתוך התיקייה vX
    ומעדכן את latest.h5 בהתאם.
    """
    out_path = version_dir / "model.h5"
    model.save(out_path)
    shutil.copy(out_path, LATEST_MODEL)

    print(f"✔ Saved FINAL model → {out_path}")
    print(f"✔ Updated latest model → {LATEST_MODEL}")

    return out_path


# ============================================
# 🎚 לוגיקת Class Weights
# ============================================

def compute_balanced_class_weights(y_idx_train):
    """
    מחשב class_weight מאוזן לפי sklearn.
    """
    cw = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_idx_train),
        y=y_idx_train
    )
    class_weights = {i: float(w) for i, w in enumerate(cw)}

    print("Computed class weights:", class_weights)
    return class_weights


# ============================================
# 🎛 Cosine Learning Rate + Warmup
# ============================================

def cosine_warmup_scheduler(initial_lr, total_epochs, warmup_epochs=3, min_lr=1e-6):
    """
    מחזיר callback שמבצע:
    - warmup בשלושת האפוקים הראשונים
    - ואז cosine decay עד סוף האימון
    """

    def scheduler(epoch, lr):
        # שלב warmup
        if epoch < warmup_epochs:
            return initial_lr * float(epoch + 1) / warmup_epochs

        # שלב cosine decay
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        cosine = 0.5 * (1 + np.cos(np.pi * progress))
        return min_lr + (initial_lr - min_lr) * cosine

    return tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)


# ============================================
# 📜 לוגים לאימון (אופציונלי)
# ============================================

def write_training_log(path, info_dict):
    """
    כותב training_log.txt עם מידע על האימון:
    - זמנים
    - class_weights
    - ביצועים
    - היפר-פרמטרים
    """
    with open(path, "w", encoding="utf-8") as f:
        for k, v in info_dict.items():
            f.write(f"{k}: {v}\n")

    print(f"✔ Training log saved → {path}")
