import os
import time
from pathlib import Path
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from librosa.feature.rhythm import tempo

TRACKS_DIR = "tracks"
DATA_DIR = "data"

os.makedirs(DATA_DIR, exist_ok=True)


# ============================================================
#  🟦 חלק 1 — 80 פיצרים לחלון של 0.5 שנייה
# ============================================================
def extract_window_features(y_window, sr):
    feats = []

    # MFCC (20 mean + 20 var)
    mfcc = librosa.feature.mfcc(y=y_window, sr=sr, n_mfcc=20)
    feats.extend(mfcc.mean(axis=1))
    feats.extend(mfcc.var(axis=1))

    # basic spectral
    feats.append(librosa.feature.spectral_centroid(y=y_window, sr=sr).mean())
    feats.append(librosa.feature.spectral_bandwidth(y=y_window, sr=sr).mean())
    feats.append(librosa.feature.spectral_rolloff(y=y_window, sr=sr).mean())
    feats.append(librosa.feature.zero_crossing_rate(y_window).mean())
    feats.append(librosa.feature.rms(y=y_window).mean())

    # spectral contrast (7)
    contrast = librosa.feature.spectral_contrast(y=y_window, sr=sr)
    feats.extend(contrast.mean(axis=1))

    # KEEP chroma_stft (12)
    chroma = librosa.feature.chroma_stft(y=y_window, sr=sr)
    feats.extend(chroma.mean(axis=1))

    # general features (4)
    feats.append(librosa.feature.spectral_flatness(y=y_window).mean())
    feats.append(np.std(y_window))
    feats.append(np.mean(np.abs(np.diff(y_window))))
    # BPM (scaled)
    try:
        bpm = tempo(y=y_window, sr=sr)[0]
        feats.append(bpm / 200)
    except:
        feats.append(0)

    feats = np.array(feats, dtype=np.float32)

    return feats

# ============================================================
#  🟦 חלק 2 — יצירת ספיקטרוגרמה
# ============================================================
def create_spectrogram(y, sr, out_path):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(2.99, 2.99), dpi=100)
    plt.axis("off")
    librosa.display.specshow(S_db, sr=sr, cmap='viridis')
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()


# ============================================================
#  🟦 חלק 3 — יצירת 20 חלונות × 80 פיצרים
# ============================================================
def extract_10_windows(y_segment, sr):
    win_len = int(sr * 1.0)  # 1 second
    num_windows = 10
    windows = []

    for i in range(num_windows):
        start = i * win_len
        end = start + win_len

        if end > len(y_segment):
            chunk = np.pad(y_segment[start:], (0, end - len(y_segment)))
        else:
            chunk = y_segment[start:end]

        feats = extract_window_features(chunk, sr)
        windows.append(feats)

    return np.array(windows, dtype=np.float32)


# ============================================================
#  🟦 חלק 4 — עיבוד שיר שלם
# ============================================================
def process_track(mp3_file, out_dir):
    y, sr = librosa.load(mp3_file, sr=22050, mono=True)

    total_duration = len(y) / sr
    segment_len_sec = 10
    positions = [0.15, 0.30, 0.45, 0.60, 0.75, 0.90]

    for idx, pos in enumerate(positions):
        start = int(pos * total_duration * sr)
        end = start + segment_len_sec * sr

        if end > len(y):
            continue

        segment = y[start:end]

        img_path = f"{out_dir}/part_{idx+1}.png"
        emb_path = f"{out_dir}/part_{idx+1}.npy"

        if os.path.exists(img_path) and os.path.exists(emb_path):
            continue

        # תמונה
        create_spectrogram(segment, sr, img_path)

        emb = extract_10_windows(segment, sr)
        np.save(emb_path, emb)


# ============================================================
#  🟦 חלק 5 — הרצה מקבילית
# ============================================================
def run_parallel():
    tasks = []

    for genre in os.listdir(TRACKS_DIR):
        g_path = Path(TRACKS_DIR) / genre
        if not g_path.is_dir():
            continue

        for artist in os.listdir(g_path):
            a_path = g_path / artist
            if not a_path.is_dir():
                continue

            for file in os.listdir(a_path):
                if not file.endswith(".mp3"):
                    continue

                mp3_file = a_path / file
                track_id = Path(file).stem

                out_dir = Path(DATA_DIR) / genre / artist / track_id
                out_dir.mkdir(parents=True, exist_ok=True)

                tasks.append((mp3_file, out_dir))

    total = len(tasks)
    print(f"🚀 מפעיל {total} משימות...")

    start = time.time()

    done = 0

    def show_progress(done, total):
        bar_len = 30
        filled = int(bar_len * (done / total))
        bar = "#" * filled + "-" * (bar_len - filled)
        print(f"\r[{bar}] {done}/{total}  ({(done/total)*100:.1f}%)", end="")

    with ProcessPoolExecutor() as ex:
        futures = [ex.submit(process_track, mp3, out) for mp3, out in tasks]

        for f in as_completed(futures):
            done += 1
            show_progress(done, total)

    print("\n✔ סיים! זמן כולל:", f"{time.time()-start:.2f} שניות")

if __name__ == "__main__":
    run_parallel()
