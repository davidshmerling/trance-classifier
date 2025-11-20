import json
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf

# ============================================
# ⚙️ קונפיגורציה
# ============================================

DATA_DIR = Path("data")
CACHE_DIR = Path("data_cache")

IMG_SIZE = (299, 299)
VALID_GENRES = ["goa", "psy", "dark"]

CACHE_DIR.mkdir(exist_ok=True)


# ============================================
# 🧱 בניית CACHE מהקבצים הגולמיים
# ============================================

def build_cache():
    img_list, emb_list, labels = [], [], []

    for genre in VALID_GENRES:
        genre_path = DATA_DIR / genre
        if not genre_path.exists():
            print(f"⚠️ {genre_path} לא קיימת — מדלג.")
            continue

        print(f"📁 סורק: {genre}")

        for png_path in genre_path.rglob("*.png"):
            npy_path = png_path.with_suffix(".npy")
            if not npy_path.exists():
                print(f"⚠️ חסר embedding עבור {png_path} — מדלג.")
                continue

            # תמונה
            img = Image.open(png_path).convert("RGB").resize(IMG_SIZE)
            img = np.asarray(img, dtype=np.float32) / 255.0

            # אמבדינג
            emb = np.load(npy_path).astype(np.float32)
            if emb.shape != (10, 68):
                print(f"⚠️ embedding פגום: {npy_path} — מדלג.")
                continue

            img_list.append(img)
            emb_list.append(emb)
            labels.append(genre)

    n = len(img_list)
    print(f"\n✔ נטענו {n} דגימות")

    if n == 0:
        raise RuntimeError("❌ אין דאטה — בדוק את תיקיית data/.")

    # המרות
    X_img = np.array(img_list, dtype=np.float32)
    X_emb = np.array(emb_list, dtype=np.float32)

    genre_to_idx = {g: i for i, g in enumerate(VALID_GENRES)}
    y_idx = np.array([genre_to_idx[g] for g in labels], dtype=np.int32)
    y = tf.keras.utils.to_categorical(y_idx, len(VALID_GENRES))

    # שמירה
    np.save(CACHE_DIR / "X_img.npy", X_img)
    np.save(CACHE_DIR / "X_emb.npy", X_emb)
    np.save(CACHE_DIR / "y.npy", y)

    meta = dict(
        genres=VALID_GENRES,
        num_samples=n,
        img_size=IMG_SIZE,
        emb_shape=(10, 68),
    )
    json.dump(meta, open(CACHE_DIR / "meta.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    print("📦 cache נוצר בהצלחה!")


# ============================================
# 📥 טעינת הדאטה מה-cache
# ============================================

def load_dataset(val_split=0.2):
    """טוען את ה־cache המוכן, מערבב ומחלק ל־train/val."""

    # נתיבים צפויים
    paths = {
        "img": CACHE_DIR / "X_img.npy",
        "emb": CACHE_DIR / "X_emb.npy",
        "y": CACHE_DIR / "y.npy",
    }

    # בדיקת קיום
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "❌ קבצי cache חסרים. הרץ build_cache().\n" + "\n".join(missing)
        )

    # טעינה
    X_img = np.load(paths["img"])
    X_emb = np.load(paths["emb"])
    y = np.load(paths["y"])

    N = len(X_img)
    print(f"✔ נטען cache: {N} דגימות")

    y_idx_full = np.argmax(y, axis=1)

    # ערבוב
    perm = np.random.permutation(N)
    X_img, X_emb, y, y_idx_full = X_img[perm], X_emb[perm], y[perm], y_idx_full[perm]

    # חלוקה
    val_size = int(N * val_split)

    X_img_val = X_img[:val_size]
    X_emb_val = X_emb[:val_size]
    y_val = y[:val_size]

    X_img_train = X_img[val_size:]
    X_emb_train = X_emb[val_size:]
    y_train = y[val_size:]
    y_train_idx = y_idx_full[val_size:]

    print(f"Train: {len(X_img_train)} | Val: {len(X_img_val)}")

    return (
        X_img_train, X_emb_train, y_train, y_train_idx,
        X_img_val, X_emb_val, y_val
    )


# ============================================
# 🔧 MAIN — רק Build Cache
# ============================================

if __name__ == "__main__":
    build_cache()
