import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

TMP_DIR = Path("tmp")
TMP_DIR.mkdir(exist_ok=True)

# ----------------------------------------
# 1) פיצול ל־6 חלקים
# ----------------------------------------
def split_into_6_segments(mp3_path: Path, seg_sec=10):
    y, sr = librosa.load(mp3_path, sr=22050)
    total = len(y)
    positions = [0.15, 0.30, 0.45, 0.60, 0.75, 0.90]

    segments = []
    for p in positions:
        s = int(p * total)
        e = s + seg_sec * sr
        if e > total:
            seg = np.pad(y[s:], (0, e - total))
        else:
            seg = y[s:e]
        segments.append(seg)

    return segments, sr


# ----------------------------------------
# 2) מטריצה 10×68
# ----------------------------------------
def extract_10x68(seg, sr):
    chunks = np.array_split(seg, 10)
    mats = []
    for ch in chunks:
        S = librosa.feature.melspectrogram(y=ch, sr=sr, n_mels=68)
        S_db = librosa.power_to_db(S, ref=np.max)
        mats.append(S_db.mean(axis=1))
    return np.array(mats, dtype=np.float32)


# ----------------------------------------
# 3) META vector (4)
# ----------------------------------------
def extract_meta(seg, sr):
    rms = librosa.feature.rms(y=seg).mean()
    flat = librosa.feature.spectral_flatness(y=seg).mean()
    cent = librosa.feature.spectral_centroid(y=seg, sr=sr).mean()
    bw = librosa.feature.spectral_bandwidth(y=seg, sr=sr).mean()
    return np.array([rms, flat, cent, bw], dtype=np.float32)


# ----------------------------------------
# 4) יצירת ספקטרוגרמה בקובץ זמני
# ----------------------------------------
def create_spectrogram_png(seg, sr, idx):
    out = TMP_DIR / f"seg_{idx}.png"

    S = librosa.feature.melspectrogram(y=seg, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(2.99, 2.99), dpi=100)
    plt.axis("off")
    librosa.display.specshow(S_db, sr=sr, cmap="viridis")
    plt.savefig(out, bbox_inches="tight", pad_inches=0)
    plt.close()

    return out
