import json
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf

# ============================================
# ⚙️ קונפיגורציה
# ============================================

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (299, 299)
VALID_GENRES = ["goa", "psy", "dark"]



# ============================================
# 🧱 יצירת Cache מה-data החדש (Image + Embedding + Meta)
# ============================================

def build_cache():
    img_list, emb_list, meta_list, labels = [], [], [], []

    for genre in VALID_GENRES:
        genre_path = DATA_DIR / genre
        if not genre_path.exists():
            print(f"⚠️ {genre_path} לא קיימת — מדלג.")
            continue

        print(f"📁 סורק: {genre}")

        # מחפש קבצי part_*.png
        for png_path in genre_path.rglob("part_*.png"):
            stem = png_path.stem.replace("part_", "")
            folder = png_path.parent

            emb_path = folder / f"part_{stem}.npy"
            meta_path = folder / f"part_{stem}_meta.npy"

            if not emb_path.exists():
                print(f"⚠️ חסר embedding עבור {png_path} — מדלג.")
                continue

            if not meta_path.exists():
                print(f"⚠️ חסר meta עבור {png_path} — מדלג.")
                continue

            # ===== תמונה =====
            img = Image.open(png_path).convert("RGB").resize(IMG_SIZE)
            img = np.asarray(img, dtype=np.float32) / 255.0

            # ===== Embedded Audio (10×80) =====
            emb = np.load(emb_path).astype(np.float32)
            if emb.shape != (10, 68):
                print(f"⚠️ embedding פגום: {emb_path}, shape={emb.shape} — מדלג.")
                continue

            # ===== Meta Vector (4) =====
            meta = np.load(meta_path).astype(np.float32)
            if meta.shape != (4,):
                print(f"⚠️ meta פגום: {meta_path}, shape={meta.shape} — מדלג.")
                continue

            img_list.append(img)
            emb_list.append(emb)
            meta_list.append(meta)
            labels.append(genre)

    n = len(img_list)
    print(f"\n✔ נטענו {n} דגימות")

    if n == 0:
        raise RuntimeError("❌ אין דאטה — בדוק את תיקיית data/.")

    # המרות
    X_img = np.array(img_list, dtype=np.float32)
    X_emb = np.array(emb_list, dtype=np.float32)
    X_meta = np.array(meta_list, dtype=np.float32)

    genre_to_idx = {g: i for i, g in enumerate(VALID_GENRES)}
    y_idx = np.array([genre_to_idx[g] for g in labels], dtype=np.int32)
    y = tf.keras.utils.to_categorical(y_idx, len(VALID_GENRES))

    # שמירת cache
    np.save(CACHE_DIR / "X_img.npy", X_img)
    np.save(CACHE_DIR / "X_emb.npy", X_emb)
    np.save(CACHE_DIR / "X_meta.npy", X_meta)
    np.save(CACHE_DIR / "y.npy", y)

    meta = dict(
        genres=VALID_GENRES,
        num_samples=n,
        img_size=IMG_SIZE,
        emb_shape=(10, 68),
        meta_shape=(4,)
    )
    json.dump(meta, open(CACHE_DIR / "meta.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    print("📦 cache נוצר בהצלחה!")


# ============================================
# 📥 טעינת הדאטה מה-cache
# ============================================

def load_dataset(val_split=0.2):

    paths = {
        "img": CACHE_DIR / "X_img.npy",
        "emb": CACHE_DIR / "X_emb.npy",
        "meta": CACHE_DIR / "X_meta.npy",
        "y": CACHE_DIR / "y.npy",
    }

    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "❌ קבצי cache חסרים. הרץ build_cache().\n" + "\n".join(missing)
        )

    X_img = np.load(paths["img"])
    X_emb = np.load(paths["emb"])
    X_meta = np.load(paths["meta"])
    y = np.load(paths["y"])

    N = len(X_img)
    print(f"✔ נטען cache: {N} דגימות")

    y_idx_full = np.argmax(y, axis=1)

    # ערבוב אחיד
    perm = np.random.permutation(N)
    X_img, X_emb, X_meta, y, y_idx_full = \
        X_img[perm], X_emb[perm], X_meta[perm], y[perm], y_idx_full[perm]

    # חלוקה
    val_size = int(N * val_split)

    X_img_val = X_img[:val_size]
    X_emb_val = X_emb[:val_size]
    X_meta_val = X_meta[:val_size]
    y_val = y[:val_size]

    X_img_train = X_img[val_size:]
    X_emb_train = X_emb[val_size:]
    X_meta_train = X_meta[val_size:]
    y_train = y[val_size:]
    y_train_idx = y_idx_full[val_size:]

    print(f"Train: {len(X_img_train)} | Val: {len(X_img_val)}")

    return (
        X_img_train, X_emb_train, X_meta_train, y_train, y_train_idx,
        X_img_val, X_emb_val, X_meta_val, y_val
    )


# ============================================
# 🔧 MAIN — רק Build Cache
# ============================================

if __name__ == "__main__":
    build_cache()
