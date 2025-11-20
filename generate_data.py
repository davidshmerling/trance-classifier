import os
import time
from pathlib import Path
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

TRACKS_DIR = "tracks"
DATA_DIR = "data"

os.makedirs(DATA_DIR, exist_ok=True)


# ------------------------------
# extract_embedding
# ------------------------------
def extract_embedding(y, sr):
    features = []
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    features.extend(mfcc.mean(axis=1))
    features.extend(mfcc.var(axis=1))

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    rms = librosa.feature.rms(y=y)[0]
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

    features.append(centroid.mean())
    features.append(bandwidth.mean())
    features.append(rolloff.mean())
    features.append(zcr.mean())
    features.append(rms.mean())
    features.extend(contrast.mean(axis=1))

    try:
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        features.extend(tonnetz.mean(axis=1))
    except:
        features.extend([0]*6)

    return np.array(features, dtype=np.float32)


# ------------------------------
# create_spectrogram
# ------------------------------
def create_spectrogram(y, sr, out_path):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(2.99, 2.99), dpi=100)
    plt.axis("off")
    librosa.display.specshow(S_db, sr=sr, cmap='viridis')
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()


# ------------------------------
# process_track
# ------------------------------
def process_track(mp3_file, out_dir):
    y, sr = librosa.load(mp3_file, sr=22050, mono=True)

    total_sec = len(y) / sr
    segment_len = 10
    positions = [0.15, 0.30, 0.45, 0.60, 0.75, 0.90]

    for idx, pos in enumerate(positions):
        start = int(pos * total_sec * sr)
        end = start + (segment_len * sr)

        if end > len(y):
            continue

        segment = y[start:end]

        img_path = f"{out_dir}/part_{idx+1}.png"
        emb_path = f"{out_dir}/part_{idx+1}.npy"

        if os.path.exists(img_path) and os.path.exists(emb_path):
            continue

        create_spectrogram(segment, sr, img_path)
        emb = extract_embedding(segment, sr)
        np.save(emb_path, emb)


# ------------------------------
# Parallel runner
# ------------------------------
def run_parallel():
    tasks = []

    # אוספים את כל המשימות מראש
    for genre in os.listdir(TRACKS_DIR):
        genre_path = Path(TRACKS_DIR) / genre
        if not genre_path.is_dir():
            continue

        for artist in os.listdir(genre_path):
            artist_path = genre_path / artist
            if not artist_path.is_dir():
                continue

            for file in os.listdir(artist_path):
                if not file.endswith(".mp3"):
                    continue

                mp3_file = artist_path / file
                track_id = Path(file).stem

                out_dir = Path(DATA_DIR) / genre / artist / track_id
                out_dir.mkdir(parents=True, exist_ok=True)

                tasks.append((mp3_file, out_dir))

    start = time.time()
    print(f"🚀 מפעיל {len(tasks)} משימות ב־ProcessPoolExecutor...")

    # מריץ במקביל על כל הליבות
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_track, mp3, out) for mp3, out in tasks]

        for f in as_completed(futures):
            pass  # אפשר להוסיף הדפסה אם רוצים

    print(f"✔ סיים! זמן כולל: {time.time() - start:.2f} שניות")


# ------------------------------
if __name__ == "__main__":
    run_parallel()
