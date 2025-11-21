import os
import json
import time
import random
import shutil
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
from mutagen.mp3 import MP3
from tensorflow.keras.models import load_model
from pathlib import Path

# -------------------------
# CONFIG
# -------------------------
TRACKS_DIR = "tracks"
TMP_DIR = "tmp_test"
MODEL_PATH = os.path.join("models", "latest.h5")
RESULT_DIR = "result_test_from_our_tracks"
JSON_OUTPUT = "results.json"
IMG_SIZE = (299, 299)
NUM_TESTS = 100
VALID_GENRES = ["goa", "psy", "dark"]

os.makedirs(TMP_DIR, exist_ok=True)


# -------------------------
# Create spectrogram
# -------------------------
def create_spectrogram(y, sr, out_path):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(2.99, 2.99), dpi=100)
    plt.axis("off")
    librosa.display.specshow(S_db, sr=sr, cmap='viridis')
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()


# -------------------------
# Split audio into 1-minute parts
# -------------------------
def process_track(mp3_path):
    audio = MP3(mp3_path)
    length_sec = int(audio.info.length)

    if length_sec > 900:
        return {"error": "too_long"}

    num_parts = length_sec // 60
    if num_parts < 1:
        return {"error": "too_short"}

    y, sr = librosa.load(mp3_path, sr=22050, mono=True)

    image_paths = []
    for part in range(num_parts):
        start = part * 60
        end = start + 60
        segment = y[start * sr : min(end * sr, len(y))]

        img_path = os.path.join(TMP_DIR, f"{Path(mp3_path).stem}_part_{part+1}.png")
        create_spectrogram(segment, sr, img_path)
        image_paths.append(img_path)

    return {"parts": image_paths}


# -------------------------
# Run model on parts
# -------------------------
def run_model(model, image_paths, classes, threshold=0.50):
    avg = {c: 0.0 for c in classes}
    part_results = []

    for idx, img_path in enumerate(image_paths, start=1):
        img = plt.imread(img_path).astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)
        probs = model.predict(img, verbose=0)[0]

        pd = {classes[i]: float(probs[i]) for i in range(len(classes))}
        part_results.append({"part": idx, "probs": pd})

        for c in classes:
            avg[c] += pd[c]

    # ---- ממוצע על כל החלקים ----
    for c in classes:
        avg[c] /= len(image_paths)

    # ---- בחירת ז'אנר לפי הסיכוי הגבוה ביותר ----
    final = max(avg, key=avg.get)

    # ---- אם אף ז'אנר לא עבר threshold → NONE ----
    if avg[final] < threshold:
        final = "none"

    return part_results, avg, final

# -------------------------
# Create test result folder
# -------------------------
def prepare_result_dir():
    if os.path.exists(RESULT_DIR):
        shutil.rmtree(RESULT_DIR)
    os.makedirs(RESULT_DIR)


# -------------------------
# Save confusion matrix
# -------------------------
def save_confusion_matrix(true_labels, pred_labels):
    cm = confusion_matrix(true_labels, pred_labels, labels=VALID_GENRES)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=VALID_GENRES,
                yticklabels=VALID_GENRES,
                cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix - Our Tracks Test")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "confusion_matrix.png"))
    plt.close()


# -------------------------
# Save genre accuracy plot
# -------------------------
def save_genre_accuracy_plot(true_labels, pred_labels):
    counts = {g: {"correct": 0, "total": 0} for g in VALID_GENRES}

    for t, p in zip(true_labels, pred_labels):
        counts[t]["total"] += 1
        if t == p:
            counts[t]["correct"] += 1

    genres = VALID_GENRES
    acc = [counts[g]["correct"] / counts[g]["total"] * 100 for g in genres]

    plt.figure(figsize=(6, 4))
    plt.bar(genres, acc, color=["green", "blue", "purple"])
    plt.ylabel("Accuracy %")
    plt.title("Accuracy Per Genre")
    plt.ylim(0, 100)
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "genre_accuracy.png"))
    plt.close()


# -------------------------
# MAIN
# -------------------------
def test_from_our_tracks():
    prepare_result_dir()

    print("\n🚀 מתחיל בדיקת 100 טרקים מתוך המאגר...\n")

    model = load_model(MODEL_PATH)
    classes = ["Goa", "Psy", "Dark"]

    # אוספים את כל הטרקים עם התווית הנכונה
    all_tracks = []
    for g in VALID_GENRES:
        for f in Path(TRACKS_DIR, g).glob("**/*.mp3"):
            all_tracks.append((g, str(f)))

    random.shuffle(all_tracks)
    selected = all_tracks[:NUM_TESTS]

    results = []
    true_labels = []
    pred_labels = []

    start_time = time.time()

    for i, (true_genre, mp3_path) in enumerate(selected, start=1):
        print(f"[{i}/{NUM_TESTS}] בודק: {mp3_path}")

        t0 = time.time()
        processed = process_track(mp3_path)

        if "error" in processed:
            results.append({
                "track": mp3_path,
                "true_genre": true_genre,
                "error": processed["error"]
            })
            continue

        parts = processed["parts"]
        part_results, avg, final = run_model(model, parts, classes)

        results.append({
            "track": mp3_path,
            "true_genre": true_genre,
            "predicted_genre": final,
            "parts": part_results,
            "average": avg,
            "correct": (final.lower() == true_genre.lower()),
            "runtime_sec": round(time.time() - t0, 2)
        })

        true_labels.append(true_genre)
        pred_labels.append(final.lower())

    # Save JSON
    with open(os.path.join(RESULT_DIR, JSON_OUTPUT), "w") as f:
        json.dump(results, f, indent=4)

    # Confusion Matrix + Graphs
    save_confusion_matrix(true_labels, pred_labels)
    save_genre_accuracy_plot(true_labels, pred_labels)

    # Clean temp PNG files
    for f in os.listdir(TMP_DIR):
        if f.endswith(".png"):
            os.remove(os.path.join(TMP_DIR, f))

    total_runtime = round(time.time() - start_time, 2)
    print(f"\n⏱ זמן ריצה כולל: {total_runtime} שניות")
    print(f"📁 תוצאות נשמרו בתיקייה: {RESULT_DIR}")
    print("✔️ הסתיים!")


# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    test_from_our_tracks()
