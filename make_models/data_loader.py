import json
import numpy as np
import tensorflow as tf

# 📌 רק משתנים נחוצים מהקונפיג
from config import (
    DATA_DIR, CACHE_DIR, VALID_GENRES,
    EMB_SHAPE, META_VECTOR_LENGTH, NUM_GENRES
)

# יצירת תקיית cache אם לא קיימת
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 🧱 build_cache – סריקת data ויצירת קבצי cache
# ============================================================
def build_cache():
    emb_list, meta_list, labels = [], [], []

    for genre in VALID_GENRES:
        genre_path = DATA_DIR / genre
        if not genre_path.exists():
            print(f"⚠️ {genre_path} לא קיימת — מדלג.")
            continue

        print(f"📁 סורק: {genre}")

        for emb_path in genre_path.rglob("part_*.npy"):
            if "_meta.npy" in str(emb_path):
                continue  # מדלג על קבצי meta

            stem = emb_path.stem.replace("part_", "")
            meta_path = emb_path.parent / f"part_{stem}_meta.npy"

            if not meta_path.exists():
                print(f"⚠️ חסר meta עבור {emb_path}")
                continue

            emb = np.load(emb_path).astype(np.float32)
            meta = np.load(meta_path).astype(np.float32)

            if emb.shape != EMB_SHAPE:
                raise ValueError(f"❌ embedding {emb.shape}, ציפינו {EMB_SHAPE}")
            if meta.shape != (META_VECTOR_LENGTH,):
                raise ValueError(f"❌ meta {meta.shape}, ציפינו {(META_VECTOR_LENGTH,)}")

            emb_list.append(emb)
            meta_list.append(meta)
            labels.append(genre)

    n = len(emb_list)
    print(f"\n✔ נטענו {n} דגימות")
    if n == 0:
        raise RuntimeError("❌ אין דאטה — בדוק את data/")

    # המרה ל־numpy arrays
    X_emb = np.array(emb_list, dtype=np.float32)
    X_meta = np.array(meta_list, dtype=np.float32)

    # יצירת one-hot
    genre_to_idx = {g: i for i, g in enumerate(VALID_GENRES)}
    y_idx = np.array([genre_to_idx[g] for g in labels], dtype=np.int32)
    y = tf.keras.utils.to_categorical(y_idx, NUM_GENRES)

    # שמירה לדיסק
    np.save(CACHE_DIR / "X_emb.npy", X_emb)
    np.save(CACHE_DIR / "X_meta.npy", X_meta)
    np.save(CACHE_DIR / "y.npy", y)

    json.dump(
        dict(genres=VALID_GENRES, num_samples=n),
        open(CACHE_DIR / "meta.json", "w", encoding="utf-8"),
        indent=2, ensure_ascii=False
    )

    print("📦 cache נוצר בהצלחה!")


# ============================================================
# 📥 load_dataset – טעינה וחלוקה ל־Train/Val
# ============================================================
def load_dataset(val_split=0.2):
    paths = {
        "emb": CACHE_DIR / "X_emb.npy",
        "meta": CACHE_DIR / "X_meta.npy",
        "y": CACHE_DIR / "y.npy",
    }

    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("❌ חסרים קבצי cache:\n" + "\n".join(missing))

    X_emb = np.load(paths["emb"])
    X_meta = np.load(paths["meta"])
    y = np.load(paths["y"])

    N = len(X_emb)
    print(f"✔ נתונים נטענו: {N} דגימות")

    # ערבוב
    perm = np.random.permutation(N)
    X_emb, X_meta, y = X_emb[perm], X_meta[perm], y[perm]
    y_idx = np.argmax(y, axis=1)

    val_size = int(N * val_split)

    # חלוקה
    return (
        X_emb[val_size:], X_meta[val_size:], y[val_size:], y_idx[val_size:],  # Train
        X_emb[:val_size], X_meta[:val_size], y[:val_size]                    # Val
    )


# ============================================================
# ▶ MAIN – בניית cache כשהקובץ מורץ ישירות
# ============================================================
if __name__ == "__main__":
    build_cache()
