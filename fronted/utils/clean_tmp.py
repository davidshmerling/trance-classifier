import os
import shutil
from pathlib import Path


TMP_DIR = Path("tmp")
AUDIO_DIR = TMP_DIR / "audio"
IMG_DIR = TMP_DIR / "images"


def ensure_dirs():
    """יוצר את התיקיות אם הן לא קיימות."""
    TMP_DIR.mkdir(exist_ok=True)
    AUDIO_DIR.mkdir(exist_ok=True)
    IMG_DIR.mkdir(exist_ok=True)


def cleanup_tmp():
    """
    מנקה *רק* את מה שצריך מהתיקיית tmp:
    - קבצי mp3 שנוצרו בניתוח
    - תמונות spectrogram ביניים
    - קבצי npy של embedding/meta
    ולא מוחק את התיקיות עצמן.
    """
    ensure_dirs()

    # מחיקה של קבצי אודיו
    for f in AUDIO_DIR.glob("*.mp3"):
        try:
            f.unlink()
        except Exception as e:
            print(f"⚠️ לא הצלחתי למחוק {f}: {e}")

    # מחיקה של תמונות
    for f in IMG_DIR.glob("*.png"):
        try:
            f.unlink()
        except Exception as e:
            print(f"⚠️ לא הצלחתי למחוק {f}: {e}")

    # מחיקת קבצי npy מהשורש
    for f in TMP_DIR.glob("*.npy"):
        try:
            f.unlink()
        except Exception as e:
            print(f"⚠️ לא הצלחתי למחוק {f}: {e}")

    # מחיקת spectrogram root (למקרה ששמרת שם)
    spectro = TMP_DIR / "spectro.png"
    if spectro.exists():
        try:
            spectro.unlink()
        except Exception as e:
            print(f"⚠️ בעיה במחיקת spectro.png: {e}")

    print("✔ tmp clean completed.")

