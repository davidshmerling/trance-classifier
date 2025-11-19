import os
import json
import time
import yt_dlp
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from mutagen.mp3 import MP3
from tensorflow.keras.models import load_model
from pathlib import Path

MODEL_PATH = os.path.join("models", "latest.h5")
JSON_OUTPUT = "prediction.json"

IMG_SIZE = (299, 299)

# tmp directory
TMP_DIR = "tmp"
os.makedirs(TMP_DIR, exist_ok=True)


# -------------------------------------------------
# יצירת ספקטוגרמה ללא שוליים
# -------------------------------------------------
def create_spectrogram(y, sr, out_path):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(2.99, 2.99), dpi=100)
    plt.axis("off")
    librosa.display.specshow(S_db, sr=sr, cmap='viridis')
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()


# -------------------------------------------------
# הורדת יוטיוב ל-MP3 זמני בתוך tmp/
# -------------------------------------------------
def download_youtube(url, out_path):
    opts = {
        "format": "bestaudio/best",
        "outtmpl": out_path,
        "quiet": True,
        "no_warnings": True,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "128",
        }],
    }

    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])


# -------------------------------------------------
# עיבוד הטרק ויצירת חלקים של דקה
# -------------------------------------------------
def process_track(mp3_path):
    audio = MP3(mp3_path)
    length_sec = int(audio.info.length)

    # מעל 15 דקות → הודעת שגיאה
    if length_sec > 900:
        return {"error": "longer_than_15_minutes"}

    # מספר החלקים = floor(T)
    num_parts = length_sec // 60
    if num_parts < 1:
        return {"error": "track_too_short"}

    y, sr = librosa.load(mp3_path, sr=22050, mono=True)
    image_paths = []

    for part in range(num_parts):
        start = part * 60
        end = start + 60

        start_samp = start * sr
        end_samp = min(end * sr, len(y))

        segment = y[start_samp:end_samp]

        img_path = os.path.join(TMP_DIR, f"part_{part + 1}.png")
        create_spectrogram(segment, sr, img_path)
        image_paths.append(img_path)

    return {"parts": image_paths}


# -------------------------------------------------
# הרצת המודל על כל תמונה
# -------------------------------------------------
def run_model_on_images(model, image_paths, class_names):
    results = []
    avg = {c: 0 for c in class_names}

    for idx, img_path in enumerate(image_paths, start=1):
        img_arr = plt.imread(img_path).astype("float32") / 255.0
        img_arr = np.expand_dims(img_arr, axis=0)

        probs = model.predict(img_arr, verbose=0)[0]
        prob_dict = {class_names[i]: float(probs[i]) for i in range(len(class_names))}

        results.append({
            "part": idx,
            "probabilities": prob_dict
        })

        for c in class_names:
            avg[c] += prob_dict[c]

    for c in class_names:
        avg[c] /= len(image_paths)

    return results, avg


# -------------------------------------------------
# MAIN PREDICT FUNCTION
# -------------------------------------------------
def predict_youtube(url, threshold=0.50):
    start_time = time.time()

    # טוען מודל
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"לא נמצא מודל בנתיב: {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    class_names = ["Goa", "Psy", "Dark"]

    # יצירת MP3 זמני
    mp3_path = os.path.join(TMP_DIR, "temp.mp3")

    # הורדה
    download_youtube(url, mp3_path)

    # יצירת חלקים
    processed = process_track(mp3_path)

    if "error" in processed:
        result_json = processed
        with open(JSON_OUTPUT, "w") as f:
            json.dump(result_json, f, indent=4)
        if os.path.exists(mp3_path):
            os.remove(mp3_path)
        return result_json

    # הפעלת המודל
    image_paths = processed["parts"]
    parts_results, avg = run_model_on_images(model, image_paths, class_names)

    # ---- בחירת ז'אנר סופי ----
    final = max(avg, key=avg.get)

    # ---- בדיקת threshold ----
    if avg[final] < threshold:
        final = "none"

    result_json = {
        "parts": parts_results,
        "average": avg,
        "final_prediction": final
    }

    # שמירה ל־JSON
    with open(JSON_OUTPUT, "w") as f:
        json.dump(result_json, f, indent=4)

    # ניקוי temp
    if os.path.exists(mp3_path):
        os.remove(mp3_path)

    return result_json
