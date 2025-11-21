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
# 🟦 1 — פיצ’רים גלובליים לשיר מלא (BPM + KEY)
# ============================================================

def extract_global_features(y_full, sr):
    # BPM
    try:
        bpm = float(librosa.feature.rhythm.tempo(y=y_full, sr=sr)[0])
    except:
        bpm = 0.0

    # Key
    try:
        chroma = librosa.feature.chroma_stft(y=y_full, sr=sr)
        key = int(np.argmax(chroma.mean(axis=1)))  # בין 0–11
    except:
        key = 0

    # Scaling
    bpm_scaled = bpm / 200.0
    key_scaled = key / 11.0

    return bpm_scaled, key_scaled


# ============================================================
# 🟦 2 — פיצ’רים מקומיים לכל חלק (Energy + Flatness)
# ============================================================

def extract_local_features(y_seg):
    try:
        energy = float(librosa.feature.rms(y=y_seg).mean())
    except:
        energy = 0.0

    try:
        flat = float(librosa.feature.spectral_flatness(y=y_seg).mean())
    except:
        flat = 0.0

    return energy, flat


# ============================================================
# 🟦 3 — 80 פיצ’רים לחלון (MFCC + ספקטראליים)
# ============================================================

def extract_window_features(y_window, sr):
    feats = []

    # MFCC — 20 mean + 20 var
    mfcc = librosa.feature.mfcc(y=y_window, sr=sr, n_mfcc=20)
    feats.extend(mfcc.mean(axis=1))
    feats.extend(mfcc.var(axis=1))

    # basic spectral
    feats.append(librosa.feature.spectral_centroid(y=y_window, sr=sr).mean())
    feats.append(librosa.feature.spectral_bandwidth(y=y_window, sr=sr).mean())
    feats.append(librosa.feature.spectral_rolloff(y=y_window, sr=sr).mean())
    feats.append(librosa.feature.zero_crossing_rate(y_window).mean())
    feats.append(librosa.feature.rms(y=y_window).mean())

    # contrast (7)
    contrast = librosa.feature.spectral_contrast(y=y_window, sr=sr)
    feats.extend(contrast.mean(axis=1))

    # chroma (12)
    chroma = librosa.feature.chroma_stft(y=y_window, sr=sr)
    feats.extend(chroma.mean(axis=1))

    # general (3)
    feats.append(librosa.feature.spectral_flatness(y=y_window).mean())
    feats.append(np.std(y_window))
    feats.append(np.mean(np.abs(np.diff(y_window))))

    # BPM (scaled)
    try:
        bpm = tempo(y=y_window, sr=sr)[0]
        feats.append(bpm / 200)
    except:
        feats.append(0)

    return np.array(feats, dtype=np.float32)


# ============================================================
# 🟦 4 — יצירת אמבדינג 10×80
# ============================================================

def extract_10_windows(y_segment, sr):
    win_len = int(sr * 1.0)  # 1 שניה
    windows = []

    for i in range(10):
        start = i * win_len
        end = start + win_len

        if end > len(y_segment):
            chunk = np.pad(y_segment[start:], (0, end - len(y_segment)))
        else:
            chunk = y_segment[start:end]

        windows.append(extract_window_features(chunk, sr))

    return np.array(windows, dtype=np.float32)


# ============================================================
# 🟦 5 — יצירת ספקטוגרמה לתמונה
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
# 🟦 6 — עיבוד שיר מלא → 6 חלקים
# ============================================================

def process_track(mp3_file, out_dir):
    y_full, sr = librosa.load(mp3_file, sr=22050, mono=True)

    # פיצ’רים גלובליים
    bpm_scaled, key_scaled = extract_global_features(y_full, sr)

    # 6 מיקומים בשיר
    positions = [0.15, 0.30, 0.45, 0.60, 0.75, 0.90]
    seg_len = 10  # שניות

    total_duration = len(y_full) / sr

    for idx, pos in enumerate(positions):
        start = int(pos * total_duration * sr)
        end = start + seg_len * sr
        if end > len(y_full):
            continue

        seg = y_full[start:end]

        img_path = out_dir / f"part_{idx+1}.png"
        emb_path = out_dir / f"part_{idx+1}.npy"
        meta_path = out_dir / f"part_{idx+1}_meta.npy"

        # ספקטוגרמה
        create_spectrogram(seg, sr, img_path)

        # אמבדינג
        emb = extract_10_windows(seg, sr)
        np.save(emb_path, emb)

        # META: [BPM, KEY, ENERGY, FLATNESS]
        energy, flatness = extract_local_features(seg)
        meta = np.array([bpm_scaled, key_scaled, energy, flatness], dtype=np.float32)
        np.save(meta_path, meta)


# ============================================================
# 🟦 7 — הרצה מקבילית ליצירת כל הדאטה
# ============================================================

def run_parallel():
    tasks = []

    # מפעילים על tracks/<genre>/<artist>/<mp3>
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
    print(f"🚀 יוצר דאטה חדש לגמרי עבור {total} קבצים...")

    start = time.time()
    done = 0

    def show_progress(done, total):
        bar = "#" * int(30 * done/total)
        bar = bar.ljust(30, '-')
        print(f"\r[{bar}] {done}/{total}  ({done/total*100:.1f}%)", end="")

    with ProcessPoolExecutor() as ex:
        futures = [ex.submit(process_track, mp3, out) for mp3, out in tasks]
        for f in as_completed(futures):
            done += 1
            show_progress(done, total)

    print("\n✔ סיים! זמן כולל:", f"{time.time()-start:.2f} שניות")


if __name__ == "__main__":
    # ⚠️ מוחק הכל לפני בנייה מחדש
    if Path(DATA_DIR).exists():
        print("🗑 מוחק תיקיית DATA ישנה...")
        import shutil
        shutil.rmtree(DATA_DIR)
        os.makedirs(DATA_DIR)

    run_parallel()
