import os
import time
from pathlib import Path
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

TRACKS_DIR = "tracks"
IMAGES_DIR = "images"

os.makedirs(IMAGES_DIR, exist_ok=True)


def create_spectrogram(y, sr, out_path):
    """יוצר ומציל תמונה בגודל 299x299 ללא שוליים."""
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(2.99, 2.99), dpi=100)  # 299×299 פיקסלים
    plt.axis("off")
    librosa.display.specshow(S_db, sr=sr, cmap='viridis')
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def process_track(mp3_path, out_dir):
    """מפיק 3 קטעים של דקה מכל טראק."""
    y, sr = librosa.load(mp3_path, sr=22050, mono=True)

    total_sec = len(y) / sr
    segment_len = 60   # דקה
    positions = [0.25, 0.50, 0.75]  # שלושה מקומות באמצע השיר

    for idx, pos in enumerate(positions):
        start = int(pos * total_sec)
        start_sample = start * sr
        end_sample = start_sample + (segment_len * sr)

        if end_sample > len(y):
            continue

        segment = y[start_sample:end_sample]
        out_path = f"{out_dir}/part_{idx+1}.png"

        if os.path.exists(out_path):
            print(f"⏩ מדלג (קיים): {out_path}")
            continue

        print(f"🎨 יוצר: {out_path}")
        create_spectrogram(segment, sr, out_path)


def run():
    start_time = time.time()

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

                track_id = Path(file).stem
                mp3_file = artist_path / file

                out_dir = Path(IMAGES_DIR) / genre / artist / track_id
                out_dir.mkdir(parents=True, exist_ok=True)

                process_track(mp3_file, out_dir)

    elapsed = time.time() - start_time
    print(f"\n✔ הסתיים! זמן ריצה: {elapsed:.2f} שניות")


if __name__ == "__main__":
    run()
