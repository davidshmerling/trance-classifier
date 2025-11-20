import os
import time
import numpy as np
import tensorflow as tf
from PIL import Image
from pathlib import Path
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

# ============================================
# ⚙️ חלק 0 — קונפיגורציה בסיסית וזרעים
# ============================================

layers = tf.keras.layers
models = tf.keras.models

DATA_DIR = "data"
MODELS_DIR = "models"
LATEST_MODEL = os.path.join(MODELS_DIR, "latest.h5")

IMG_SIZE = (299, 299)
BATCH_SIZE = 16
EPOCHS = 20

VALID_GENRES = ["goa", "psy", "dark"]

# זרעים לשחזוריות בסיסית
np.random.seed(42)
tf.random.set_seed(42)

sns.set(style="whitegrid")


# ============================================
# 📥 חלק 1 — טעינת הדאטה מהדיסק לזיכרון
# ============================================
# טוען תמונות (ספקטוגרמות) ו־embeddings (10×68) לכל קטע
# מחזיר סט אימון וסט ולידציה (80/20) + תוויות one-hot + אינדקסים למחלקות


def load_dataset():
    img_list = []
    emb_list = []
    labels = []

    for genre in VALID_GENRES:
        genre_path = Path(DATA_DIR) / genre
        if not genre_path.exists():
            continue

        for root, dirs, files in os.walk(genre_path):
            for f in files:
                if not f.endswith(".png"):
                    continue

                img_path = os.path.join(root, f)
                emb_path = img_path.replace(".png", ".npy")
                if not os.path.exists(emb_path):
                    continue

                # ---- תמונה ----
                img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
                img = np.array(img, dtype=np.float32) / 255.0  # [0,1]

                # ---- אמבדינג (10×68) ----
                emb = np.load(emb_path).astype(np.float32)  # shape: (10, 68)

                # בדיקה קשיחה שהצורה נכונה
                if emb.shape != (10, 68):
                    print(f"⚠️ אזהרה: embedding בנתיב {emb_path} הוא בצורה {emb.shape}, מדלג.")
                    continue

                img_list.append(img)
                emb_list.append(emb)
                labels.append(genre)

    print(f"✔ Loaded {len(img_list)} samples")

    if len(img_list) == 0:
        raise RuntimeError("לא נטענו דגימות. בדוק שהספרייה data/ קיימת ויש בה קבצים.")

    # המרה ל־numpy
    X_img = np.array(img_list, dtype=np.float32)          # (N, 299, 299, 3)
    X_emb = np.array(emb_list, dtype=np.float32)          # (N, 10, 68)

    # המרת תגיות לאינדקסים ו־one-hot
    genre_to_idx = {g: i for i, g in enumerate(VALID_GENRES)}
    y_idx = np.array([genre_to_idx[g] for g in labels], dtype=np.int32)
    y = tf.keras.utils.to_categorical(y_idx, num_classes=len(VALID_GENRES))

    # ערבוב
    idx = np.arange(len(X_img))
    np.random.shuffle(idx)
    X_img = X_img[idx]
    X_emb = X_emb[idx]
    y = y[idx]
    y_idx = y_idx[idx]

    # חלוקה 80/20 (ולידציה מההתחלה של המערך אחרי ערבוב)
    val_size = int(0.2 * len(X_img))

    X_img_val = X_img[:val_size]
    X_emb_val = X_emb[:val_size]
    y_val = y[:val_size]

    X_img_train = X_img[val_size:]
    X_emb_train = X_emb[val_size:]
    y_train = y[val_size:]
    y_train_idx = y_idx[val_size:]

    print(f"Train samples = {len(X_img_train)}  |  Val = {len(X_img_val)}")

    return (
        X_img_train, X_emb_train, y_train, y_train_idx,
        X_img_val, X_emb_val, y_val
    )


# ============================================
# 🧠 חלק 2 — בניית המודל (EfficientNetB0 + GRU)
# ============================================
# מודל דו-ענפי:
# 1. ענף תמונה: EfficientNetB0 (מוקפא בתחילה) + GAP + Dense
# 2. ענף אמבדינג: GRU דו-שלבי על רצף של 10×68
# לאחר מכן מאחדים (Concatenate) ומוסיפים שכבות Fully Connected


def build_model(num_classes):

    # ------------------
    # ענף תמונה — EfficientNetB0
    # ------------------
    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        weights="imagenet"
    )

    base.trainable = False  # בשלב ראשון מקפיאים. אפשר לפתוח בסוף האימון לפיין-טיונינג.

    img_input = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name="image_input")

    # אפשר להוסיף אוגמנטציה קלה (לא חובה)
    aug = layers.RandomFlip("horizontal")(img_input)
    aug = layers.RandomRotation(0.05)(aug)

    x = base(aug, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    img_vec = layers.Dense(128, activation="relu", name="img_dense")(x)

    # ------------------
    # ענף אמבדינג — GRU על רצף 10×68
    # ------------------
    emb_input = layers.Input(shape=(10, 68), name="embedding_input")

    e = layers.GRU(128, return_sequences=True, name="emb_gru_1")(emb_input)
    e = layers.GRU(64, return_sequences=False, name="emb_gru_2")(e)
    e = layers.Dropout(0.3)(e)
    emb_vec = layers.Dense(64, activation="relu", name="emb_dense")(e)

    # ------------------
    # איחוד וראש סיווג
    # ------------------
    combined = layers.Concatenate(name="concat")([img_vec, emb_vec])

    x = layers.Dense(128, activation="relu")(combined)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = models.Model(inputs=[img_input, emb_input], outputs=out, name="TranceCRNN_EfficientNet")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()
    return model


# ============================================
# 💾 חלק 3 — שמירת המודל בגרסאות
# ============================================
# שומר מודל בתיקייה חדשה models/vX/model.h5
# וגם עושה copy ל־models/latest.h5 לצורך שימוש באפליקציה


def save_versioned_model(model):
    os.makedirs(MODELS_DIR, exist_ok=True)

    versions = [
        int(f.name.replace("v", ""))
        for f in Path(MODELS_DIR).glob("v*")
        if f.is_dir() and f.name.replace("v", "").isdigit()
    ]
    next_v = (max(versions) + 1) if versions else 1

    version_dir = Path(MODELS_DIR) / f"v{next_v}"
    version_dir.mkdir(parents=True, exist_ok=True)

    model_path = version_dir / "model.h5"
    model.save(model_path)

    # עדכון latest.h5
    shutil.copy(model_path, LATEST_MODEL)

    print(f"✔ Saved model to {model_path}")
    print(f"✔ Updated latest model at {LATEST_MODEL}")

    return version_dir


# ============================================
# 📊 חלק 4 — ניתוח תוצאות (Confusion Matrix + Report + גרפים)
# ============================================
# מייצר:
# - מטריצת בלבול (confusion_matrix.png)
# - דוח טקסטואלי (report.txt)
# - גרף דיוק (accuracy.png)
# - גרף הפסד (loss.png)


def analyze_results(model, history, X_img_val, X_emb_val, y_val, version_dir):
    analysis_dir = version_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # חיזוי על סט הולידציה
    y_true = np.argmax(y_val, axis=1)
    y_pred = np.argmax(model.predict([X_img_val, X_emb_val]), axis=1)

    # מטריצת בלבול
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=VALID_GENRES,
                yticklabels=VALID_GENRES,
                cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(analysis_dir / "confusion_matrix.png")
    plt.close()

    # דוח טקסטואלי
    report = classification_report(
        y_true, y_pred, target_names=VALID_GENRES, digits=3
    )
    with open(analysis_dir / "report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    # גרף דיוק
    plt.figure()
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(analysis_dir / "accuracy.png")
    plt.close()

    # גרף הפסד
    plt.figure()
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(analysis_dir / "loss.png")
    plt.close()

    print(f"✔ Analysis saved under: {analysis_dir}")


# ============================================
# 🏃‍♂️ חלק 5 — לולאת האימון הראשית
# ============================================
# טוען דאטה, מחשב class_weights, מאמן את המודל עם callbacks,
# שומר גרסה ומריץ אנליזה על סט הולידציה


def train_model():
    print("============== Training ==============")

    (
        X_img_train, X_emb_train, y_train, y_train_idx,
        X_img_val, X_emb_val, y_val
    ) = load_dataset()

    # חישוב משקלי מחלקות לטיפול בחוסר איזון
    cw = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train_idx),
        y=y_train_idx
    )
    class_weights = {i: float(w) for i, w in enumerate(cw)}
    print("Class weights:", class_weights)

    model = build_model(num_classes=len(VALID_GENRES))

    # Callbacks — עצירת early stopping + שינוי lr כשהולידציה נתקעת
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=4,
            restore_best_weights=True,
            monitor="val_loss"
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            verbose=1,
            min_lr=1e-6
        )
    ]

    start_time = time.time()

    history = model.fit(
        [X_img_train, X_emb_train], y_train,
        validation_data=([X_img_val, X_emb_val], y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    total_time = time.time() - start_time
    print(f"\n⏱ Training time: {total_time:.2f} seconds")

    # שמירת מודל בגרסת vX + latest.h5
    version_dir = save_versioned_model(model)

    # ניתוח תוצאות ושמירת גרפים
    analyze_results(model, history, X_img_val, X_emb_val, y_val, version_dir)

    print("\n✔ DONE\n")


# ============================================
# 🔚 חלק 6 — נקודת כניסה לקובץ
# ============================================
if __name__ == "__main__":
    train_model()
