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

layers = tf.keras.layers
models = tf.keras.models

IMAGES_DIR = "images"
MODELS_DIR = "models"
LATEST_MODEL = os.path.join(MODELS_DIR, "latest.h5")

IMG_SIZE = (299, 299)
BATCH_SIZE = 16
EPOCHS = 20

VALID_GENRES = ["goa", "psy", "dark"]


# -------------------------------------------------
# טעינת התמונות
# -------------------------------------------------
def load_dataset():
    image_paths = []
    labels = []

    for genre in VALID_GENRES:
        genre_path = Path(IMAGES_DIR) / genre
        if not genre_path.exists():
            continue

        for root, dirs, files in os.walk(genre_path):
            for f in files:
                if f.lower().endswith(".png"):
                    img_path = os.path.join(root, f)
                    image_paths.append(img_path)
                    labels.append(genre)

    # המרת תמונות ל-Numpy
    X = []
    for path in image_paths:
        img = Image.open(path).convert("RGB").resize(IMG_SIZE)
        X.append(np.array(img) / 255.0)

    X = np.array(X)

    # המרת label ל-OneHot
    genre_to_idx = {g: i for i, g in enumerate(VALID_GENRES)}
    y_idx = np.array([genre_to_idx[g] for g in labels])
    y = tf.keras.utils.to_categorical(y_idx, num_classes=len(VALID_GENRES))

    # ערבוב + פיצול
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X, y, y_idx = X[indices], y[indices], y_idx[indices]

    val_size = int(0.2 * len(X))
    X_train, y_train = X[val_size:], y[val_size:]
    X_val, y_val = X[:val_size], y[:val_size]

    y_train_idx = y_idx[val_size:]  # נשתמש לזה לחישוב class weights

    return X_train, y_train, y_train_idx, X_val, y_val


# -------------------------------------------------
# בניית המודל - V2
# -------------------------------------------------
def build_model(num_classes):

    # הרחבות לתמונות (שיפור דיוק)
    data_aug = models.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])

    # EfficientNetB0 בסיסי
    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )

    # Fine-Tuning עדין — משחררים רק 20 שכבות אחרונות
    base.trainable = True
    for layer in base.layers[:-20]:
        layer.trainable = False

    inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = data_aug(inputs)
    x = base(x, training=False)  # יציבות ל-BatchNorm
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)

    # ⭐ שינוי מרכזי: sigmoid במקום softmax
    outputs = layers.Dense(num_classes, activation="sigmoid")(x)

    model = models.Model(inputs, outputs)

    # ⭐ שינוי מרכזי: binary_crossentropy + מטריקות מותאמות
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", threshold=0.5),
            tf.keras.metrics.Precision(name="precision", thresholds=0.5),
            tf.keras.metrics.Recall(name="recall", thresholds=0.5)
        ]
    )

    return model


# -------------------------------------------------
# שמירת מודל לפי גרסאות
# -------------------------------------------------
def save_versioned_model(model):
    os.makedirs(MODELS_DIR, exist_ok=True)

    # מציאת המספר הבא של הגרסה (v1, v2, v3...)
    versions = [
        int(f.name.replace("v", ""))
        for f in Path(MODELS_DIR).glob("v*")
        if f.is_dir() and f.name.replace("v", "").isdigit()
    ]
    next_version = (max(versions) + 1) if versions else 1

    # יצירת תיקייה לגרסה
    version_dir = Path(MODELS_DIR) / f"v{next_version}"
    version_dir.mkdir(parents=True, exist_ok=True)

    # שמירת המודל
    model_path = version_dir / "model.h5"
    model.save(model_path)
    print(f"✔ מודל נשמר בתיקייה: {model_path}")

    # עדכון latest.h5 מחוץ לתיקיות הגרסאות
    shutil.copy(model_path, LATEST_MODEL)
    print(f"✔ נשמר גם כמודל האחרון: {LATEST_MODEL}")

    return version_dir


# -------------------------------------------------
# ניתוח תוצאות ושמירה בתיקיית הגרסה
# -------------------------------------------------
def analyze_results(model, history, X_val, y_val, version_dir):

    print("\n🔍 מבצע ניתוח תוצאות...\n")

    analysis_dir = version_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # תחזיות
    y_true = np.argmax(y_val, axis=1)

    # כאן עדיין נשתמש ב-argmax כי יש לנו לייבל יחיד לכל דוגמה
    y_pred_proba = model.predict(X_val)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=VALID_GENRES,
                yticklabels=VALID_GENRES,
                cmap="Blues")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(analysis_dir / "confusion_matrix.png")
    plt.close()

    # Classification Report
    report = classification_report(
        y_true, y_pred,
        target_names=VALID_GENRES,
        digits=3
    )
    with open(analysis_dir / "classification_report.txt", "w") as f:
        f.write(report)

    # Accuracy / Binary Accuracy גרף
    if "binary_accuracy" in history.history:
        plt.plot(history.history['binary_accuracy'], label='Train Binary Acc')
        plt.plot(history.history['val_binary_accuracy'], label='Val Binary Acc')
        title = "Binary Accuracy Over Epochs"
    elif "accuracy" in history.history:
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        title = "Accuracy Over Epochs"
    else:
        title = "Accuracy Over Epochs"

    plt.legend()
    plt.grid()
    plt.title(title)
    plt.savefig(analysis_dir / "accuracy.png")
    plt.close()

    # Loss גרף
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.grid()
    plt.title("Loss Over Epochs")
    plt.savefig(analysis_dir / "loss.png")
    plt.close()

    print(f"📁 כל הניתוחים נשמרו בתיקייה: {analysis_dir}\n")


# -------------------------------------------------
# תהליך האימון - V2
# -------------------------------------------------
def train_model():
    print("\n============== התחלת אימון (גרסה 2) ==============\n")
    t0 = time.time()

    X_train, y_train, y_train_idx, X_val, y_val = load_dataset()
    print(f"✔ נטען: {len(X_train)} אימון | {len(X_val)} ולידציה\n")

    # חישוב class weights מאוזנים
    class_weights_arr = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train_idx),
        y=y_train_idx
    )
    class_weights = {i: w for i, w in enumerate(class_weights_arr)}
    print("⚖ class weights:", class_weights, "\n")

    model = build_model(num_classes=len(VALID_GENRES))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=4, restore_best_weights=True, monitor='val_loss'
        ),
        tf.keras.callbacks.ModelCheckpoint(
            "best_model_tmp.h5", monitor='val_loss', save_best_only=True
        )
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weights
    )

    # שמירת המודל לתיקייה vX
    version_dir = save_versioned_model(model)

    # ניתוח ושמירת תוצאות
    analyze_results(model, history, X_val, y_val, version_dir)

    total = time.time() - t0
    print(f"\n⏱ זמן כולל: {total:.2f} שניות ({total/60:.2f} דקות)")
    print("\n============== הסתיים ✔ (גרסה 2) ==============\n")


if __name__ == "__main__":
    train_model()
