import os
import json
import time
import random
import shutil
from pathlib import Path

import yt_dlp
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from mutagen.mp3 import MP3
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
import seaborn as sns
import openaito

# =========================
# CONFIG
# =========================

# נתיבים
MODEL_PATH = os.path.join("models", "latest.h5")
TMP_DIR = "tmp_llm_test"
RESULT_DIR = "result_test_random_llm_tracks"
JSON_OUTPUT = "results.json"

# הגדרות מודל / דאטה
IMG_SIZE = (299, 299)
NUM_TESTS = 30        # כמה טרקים לבדוק (אפשר לשנות ל-100)
VALID_GENRES = ["goa", "psy", "dark"]
CLASS_NAMES = ["Goa", "Psy", "Dark"]  # לפי מה שאימנת

# מפתח ל-LLM
openai.api_key = os.getenv("OPENAI_API_KEY")

os.makedirs(TMP_DIR, exist_ok=True)


# =========================
# LLM: בקשת טרק מהז'אנר
# =========================
def ask_llm_for_track(genre: str) -> dict:
    """
    מבקש ממודל השפה להציע טרק רנדומלי בסגנון מבוקש (goa/psy/dark).
    מחזיר dict עם: artist, title, search_query, raw_answer.
    """
    if openai.api_key is None:
        raise RuntimeError("לא הוגדר OPENAI_API_KEY במשתני הסביבה")

    # מיפוי כדי שהמודל יבין טוב
    genre_prompt_name = {
        "goa": "Goa trance",
        "psy": "Psytrance",
        "dark": "Darkpsy"
    }[genre]

    system_msg = (
        "You are a music assistant specializing in psytrance and its subgenres. "
        "You MUST reply in strict JSON with the keys: artist, title, search_query. "
        "search_query should be a good YouTube search string like 'Astrix Deep Jungle Walk'. "
        "Do NOT include any extra text, only valid JSON."
    )
    user_msg = (
        f"Give me one random full-length track in the style of {genre_prompt_name}. "
        f"Prefer tracks that exist on YouTube as audio or music videos. "
        f"Return JSON only."
    )

    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",  # אפשר לשנות לדגם אחר
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.9,
    )
    content = resp["choices"][0]["message"]["content"].strip()

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # נפילה: מחזירים הכל בתוך search_query כדי שלא ניפול
        return {
            "artist": "",
            "title": "",
            "search_query": content,
            "raw_answer": content,
        }

    data["raw_answer"] = content
    # הבטחת שדות בסיסיים
    for key in ["artist", "title", "search_query"]:
        data.setdefault(key, "")

    return data


# =========================
# חיפוש והורדה ביוטיוב עם yt-dlp
# =========================
def search_and_download_youtube(search_query: str, out_mp3_path: str):
    """
    מחפש את הטרק ביוטיוב בעזרת yt-dlp (ytsearch1:query) ומוריד כ-MP3.
    מחזיר (success: bool, video_url: str|None).
    """
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": out_mp3_path,
        "quiet": True,
        "no_warnings": True,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "128",
        }],
    }

    query = f"ytsearch1:{search_query}"

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(query, download=True)
    except Exception as e:
        print(f"⚠️ yt-dlp failed for query '{search_query}': {e}")
        return False, None

    # info יכול להיות עם entries
    video_url = None
    if "entries" in info and info["entries"]:
        entry = info["entries"][0]
        video_id = entry.get("id")
        if video_id:
            video_url = f"https://www.youtube.com/watch?v={video_id}"

    return True, video_url


# =========================
# יצירת ספקטוגרמה
# =========================
def create_spectrogram(y, sr, out_path):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(2.99, 2.99), dpi=100)
    plt.axis("off")
    librosa.display.specshow(S_db, sr=sr, cmap='viridis')
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()


# =========================
# חיתוך טרק לדקות ויצירת תמונות
# =========================
def process_track(mp3_path: str):
    """
    מחזיר {"parts": [paths]} או {"error": "..."}.
    """
    try:
        audio = MP3(mp3_path)
        length_sec = int(audio.info.length)
    except Exception:
        return {"error": "invalid_mp3"}

    # יותר מ-15 דקות → לא מתאים
    if length_sec > 900:
        return {"error": "too_long"}

    num_parts = length_sec // 60
    if num_parts < 1:
        return {"error": "too_short"}

    try:
        y, sr = librosa.load(mp3_path, sr=22050, mono=True)
    except Exception:
        return {"error": "librosa_load_failed"}

    image_paths = []
    for part in range(num_parts):
        start = part * 60
        end = start + 60
        seg = y[start * sr:min(end * sr, len(y))]

        img_path = os.path.join(TMP_DIR, f"{Path(mp3_path).stem}_part_{part + 1}.png")
        create_spectrogram(seg, sr, img_path)
        image_paths.append(img_path)

    return {"parts": image_paths}


# =========================
# הרצת המודל על כל התמונות
# =========================
def run_model_on_images(model, image_paths, class_names):
    """
    מחזיר (part_results, avg_probs, final_label)
    """
    avg = {c: 0.0 for c in class_names}
    part_results = []

    for idx, img_path in enumerate(image_paths, start=1):
        img_arr = plt.imread(img_path).astype("float32") / 255.0
        img_arr = np.expand_dims(img_arr, axis=0)

        probs = model.predict(img_arr, verbose=0)[0]
        prob_dict = {class_names[i]: float(probs[i]) for i in range(len(class_names))}

        part_results.append({
            "part": idx,
            "probabilities": prob_dict,
            "image_path": img_path,
        })

        for c in class_names:
            avg[c] += prob_dict[c]

    for c in class_names:
        avg[c] /= len(image_paths)

    final = max(avg, key=avg.get)
    return part_results, avg, final


# =========================
# תיקיית תוצאות
# =========================
def prepare_result_dir():
    if os.path.exists(RESULT_DIR):
        shutil.rmtree(RESULT_DIR)
    os.makedirs(RESULT_DIR, exist_ok=True)


# =========================
# Confusion Matrix + Accuracy per genre
# =========================
def save_confusion_matrix(true_labels, pred_labels):
    if not true_labels:
        return

    cm = confusion_matrix(true_labels, pred_labels, labels=VALID_GENRES)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=VALID_GENRES,
                yticklabels=VALID_GENRES,
                cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix - Random LLM YouTube Tracks")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "confusion_matrix.png"))
    plt.close()


def save_genre_accuracy_plot(true_labels, pred_labels):
    if not true_labels:
        return

    counts = {g: {"correct": 0, "total": 0} for g in VALID_GENRES}
    for t, p in zip(true_labels, pred_labels):
        counts[t]["total"] += 1
        if t == p:
            counts[t]["correct"] += 1

    genres = VALID_GENRES
    acc = []
    for g in genres:
        if counts[g]["total"] > 0:
            acc.append(counts[g]["correct"] / counts[g]["total"] * 100)
        else:
            acc.append(0.0)

    plt.figure(figsize=(6, 4))
    plt.bar(genres, acc)
    plt.ylabel("Accuracy %")
    plt.title("Accuracy Per Genre (LLM YouTube Test)")
    plt.ylim(0, 100)
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "genre_accuracy.png"))
    plt.close()


# =========================
# ניקוי TMP
# =========================
def cleanup_tmp():
    for f in os.listdir(TMP_DIR):
        full = os.path.join(TMP_DIR, f)
        if os.path.isfile(full):
            os.remove(full)


# =========================
# MAIN TEST
# =========================
def test_random_tracks_with_llm():
    prepare_result_dir()

    print("\n🚀 מתחיל בדיקת טרקים רנדומליים מיוטיוב בעזרת LLM...\n")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"לא נמצא מודל בנתיב: {MODEL_PATH}")

    model = load_model(MODEL_PATH)

    results = []
    true_labels = []
    pred_labels = []

    start_time = time.time()

    for i in range(1, NUM_TESTS + 1):
        print(f"\n========== טרק {i}/{NUM_TESTS} ==========")

        # 1. בוחרים ז'אנר אמיתי
        true_genre = random.choice(VALID_GENRES)

        # 2. מבקשים מה-LLM טרק מהז'אנר הזה
        try:
            llm_data = ask_llm_for_track(true_genre)
        except Exception as e:
            print(f"⚠️ LLM request failed: {e}")
            results.append({
                "index": i,
                "true_genre": true_genre,
                "error": "llm_failed",
                "exception": str(e),
            })
            continue

        search_query = llm_data.get("search_query", "").strip()
        if not search_query:
            print("⚠️ LLM did not return a valid search_query")
            results.append({
                "index": i,
                "true_genre": true_genre,
                "llm_data": llm_data,
                "error": "no_search_query",
            })
            continue

        print(f"🎯 ז'אנר מבוקש: {true_genre} | שאילתת חיפוש: {search_query}")

        # 3. מחפשים ביוטיוב ומורידים
        mp3_path = os.path.join(TMP_DIR, f"llm_test_{i}.mp3")
        success, video_url = search_and_download_youtube(search_query, mp3_path)

        if not success or not os.path.exists(mp3_path):
            print("⚠️ download/search failed")
            results.append({
                "index": i,
                "true_genre": true_genre,
                "llm_data": llm_data,
                "video_url": video_url,
                "error": "download_failed",
            })
            continue

        # 4. יוצרים חלקים ותמונות
        processed = process_track(mp3_path)
        if "error" in processed:
            print(f"⚠️ process_track error: {processed['error']}")
            results.append({
                "index": i,
                "true_genre": true_genre,
                "llm_data": llm_data,
                "video_url": video_url,
                "error": processed["error"],
            })
            # מוחקים את ה-MP3 בכל מקרה
            if os.path.exists(mp3_path):
                os.remove(mp3_path)
            continue

        image_paths = processed["parts"]
        if not image_paths:
            print("⚠️ no image parts created")
            results.append({
                "index": i,
                "true_genre": true_genre,
                "llm_data": llm_data,
                "video_url": video_url,
                "error": "no_image_parts",
            })
            if os.path.exists(mp3_path):
                os.remove(mp3_path)
            continue

        # 5. הרצת המודל
        t0 = time.time()
        part_results, avg_probs, final_label = run_model_on_images(model, image_paths, CLASS_NAMES)
        runtime = round(time.time() - t0, 2)

        # מיפוי לשמות lowercase של valid genres
        predicted_genre_lower = final_label.lower()
        is_correct = (predicted_genre_lower == true_genre.lower())

        print(f"✅ תחזית סופית: {final_label} (truth: {true_genre}) | correct={is_correct}")

        results.append({
            "index": i,
            "true_genre": true_genre,
            "llm_data": llm_data,
            "video_url": video_url,
            "predicted_label": final_label,
            "predicted_genre_lower": predicted_genre_lower,
            "correct": is_correct,
            "average_probs": avg_probs,
            "parts": part_results,
            "runtime_sec": runtime,
        })

        true_labels.append(true_genre)
        if predicted_genre_lower in VALID_GENRES:
            pred_labels.append(predicted_genre_lower)
        else:
            pred_labels.append("unknown")

        # מוחקים את ה-MP3
        if os.path.exists(mp3_path):
            os.remove(mp3_path)

        # מוחקים את התמונות שיצרנו
        for p in image_paths:
            if os.path.exists(p):
                os.remove(p)

    # =========================
    # שמירת JSON + גרפים
    # =========================
    json_path = os.path.join(RESULT_DIR, JSON_OUTPUT)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    save_confusion_matrix(true_labels, pred_labels)
    save_genre_accuracy_plot(true_labels, pred_labels)

    total_runtime = round(time.time() - start_time, 2)
    cleanup_tmp()

    print(f"\n⏱ זמן ריצה כולל: {total_runtime} שניות")
    print(f"📄 תוצאות נשמרו ב- {json_path}")
    print(f"📊 Confusion matrix + accuracy per genre נשמרו בתיקייה: {RESULT_DIR}")
    print("✔️ הסתיים!")


if __name__ == "__main__":
    test_random_tracks_with_llm()
