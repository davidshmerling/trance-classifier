import json
import numpy as np
import tensorflow as tf

# 📌 ייבוא *רק* של פרמטרים שנמצאים בקונפיג
from config import (
    DATA_DIR,
    CACHE_DIR,
    VALID_GENRES,
    EMB_SHAPE,
    META_VECTOR_LENGTH,
    NUM_GENRES
)

# יצירת cache אם לא קיים
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 🧱 יצירת Cache מה־data החדש
# ============================================================
def build_cache():
    emb_list, meta_list, labels = [], [], []

    for genre in VALID_GENRES:
        genre_path = DATA_DIR / genre
        if not genre_path.exists():
            print(f"⚠️ {genre_path} לא קיימת — מדלג.")
            continue

        print(f"📁 סורק: {genre}")

        # חיפוש קבצי embedding בלבד
        for emb_path in genre_path.rglob("part_*.npy"):
            if "_meta.npy" in str(emb_path):
                continue  # מדלג על קבצי מטא

            stem = emb_path.stem.replace("part_", "")
            meta_path = emb_path.parent / f"part_{stem}_meta.npy"

            if not emb_path.exists() or not meta_path.exists():
                print(f"⚠️ חסרים קבצים עבור {emb_path.parent} — מדלג.")
                continue

            # טעינת embedding
            emb = np.load(emb_path).astype(np.float32)
            if emb.shape != EMB_SHAPE:
                raise ValueError(
                    f"❌ צורת embedding שגויה ({emb.shape}) — בקובץ ציפינו {EMB_SHAPE}"
                )

            # טעינת meta
            meta = np.load(meta_path).astype(np.float32)
            if meta.shape != (META_VECTOR_LENGTH,):
                raise ValueError(
                    f"❌ meta שגוי ({meta.shape}) — בקונפיג ציפינו {META_VECTOR_LENGTH}"
                )

            emb_list.append(emb)
            meta_list.append(meta)
            labels.append(genre)

    # סיכום
    n = len(emb_list)
    print(f"\n✔ נטענו {n} דגימות")

    if n == 0:
        raise RuntimeError("❌ אין דאטה — בדוק את תיקיית data/")

    X_emb = np.array(emb_list, dtype=np.float32)
    X_meta = np.array(meta_list, dtype=np.float32)

    # המרת ל-one-hot
    genre_to_idx = {g: i for i, g in enumerate(VALID_GENRES)}
    y_idx = np.array([genre_to_idx[g] for g in labels], dtype=np.int32)
    y = tf.keras.utils.to_categorical(y_idx, NUM_GENRES)

    # שמירה לקאש
    np.save(CACHE_DIR / "X_emb.npy", X_emb)
    np.save(CACHE_DIR / "X_meta.npy", X_meta)
    np.save(CACHE_DIR / "y.npy", y)

    meta_info = dict(
        genres=VALID_GENRES,
        num_samples=n,
        emb_shape=EMB_SHAPE,
        meta_shape=(META_VECTOR_LENGTH,),
        author="🚀 Trance Classifier Automation"
    )
    json.dump(meta_info, open(CACHE_DIR / "meta.json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)

    print("📦 cache נוצר בהצלחה!")


# ============================================================
# 📥 טעינת הדאטה מה-cache
# ============================================================
def load_dataset(val_split=0.2):
    paths = {
        "emb": CACHE_DIR / "X_emb.npy",
        "meta": CACHE_DIR / "X_meta.npy",
        "y": CACHE_DIR / "y.npy",
    }

    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "❌ קבצי cache חסרים. הרץ build_cache().\n" + "\n".join(missing)
        )

    X_emb = np.load(paths["emb"])
    X_meta = np.load(paths["meta"])
    y = np.load(paths["y"])

    N = len(X_emb)
    print(f"✔ נטען cache: {N} דגימות")

    y_idx_full = np.argmax(y, axis=1)

    # ערבוב
    perm = np.random.permutation(N)
    X_emb, X_meta, y, y_idx_full = \
        X_emb[perm], X_meta[perm], y[perm], y_idx_full[perm]

    val_size = int(N * val_split)

    # חלוקה
    X_emb_val = X_emb[:val_size]
    X_meta_val = X_meta[:val_size]
    y_val = y[:val_size]

    X_emb_train = X_emb[val_size:]
    X_meta_train = X_meta[val_size:]
    y_train = y[val_size:]
    y_train_idx = y_idx_full[val_size:]

    print(f"📊 Train: {len(X_emb_train)} | Val: {len(X_emb_val)}")

    return (
        X_emb_train, X_meta_train, y_train, y_train_idx,
        X_emb_val, X_meta_val, y_val
    )


# ============================================
# ▶ MAIN
# ============================================
if __name__ == "__main__":
    build_cache()
