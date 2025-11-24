import os, time, shutil, warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import librosa
from librosa.feature.rhythm import tempo

warnings.filterwarnings("ignore", message="Trying to estimate tuning from empty frequency set")

from make_models.config import (
    TRACKS_DIR, DATA_DIR, SR,
    SEGMENT_SECONDS, WINDOW_STRIDE_SECONDS, SEGMENT_POSITIONS,
    FEATURES_CONFIG, FEATURES_PER_WINDOW, WINDOWS_PER_SEGMENT,
    META_FEATURE_NAMES, MIN_TRACK_DURATION_SEC, MAX_TRACK_DURATION_SEC,
    CHECK_SILENCE, ENERGY_THRESHOLD
)

DATA_DIR.mkdir(parents=True, exist_ok=True)

# =============================== UTIL FILTERS ===============================

def bad_chroma(y, sr):
    try:
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        return chroma.mean() < 0.01
    except:
        return True

def bad_embedding(emb):
    return emb.std() < 1e-5 or np.all(emb == 0)

def bad_meta(meta):
    if np.all(meta < 1e-4): return True
    if meta.std() < 1e-6: return True
    if meta[0] > 0 and meta[2] < 1e-5: return True  # BPM>0 but no energy
    return False

# =============================== GLOBAL FEATURES ============================

def extract_global_features(y, sr):
    try: bpm = float(tempo(y=y, sr=sr)[0]) / 200.0
    except: bpm = 0.0
    try:
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        key = int(np.argmax(chroma.mean(axis=1))) / 11.0
    except:
        key = 0.0
    return bpm, key

# =============================== LOCAL FEATURES ==============================

def extract_local_features(y, sr):
    try: energy = float(librosa.feature.rms(y=y).mean())
    except: energy = 0.0
    try: flat = float(librosa.feature.spectral_flatness(y=y).mean())
    except: flat = 0.0
    return energy, flat

def silent_segment(y, sr):
    if not CHECK_SILENCE: return False
    try: energy = float(librosa.feature.rms(y=y).mean())
    except: energy = 0.0
    return energy < ENERGY_THRESHOLD

# =============================== WINDOW FEATURES =============================

def extract_window_features(y, sr):
    feats = []

    if "MFCC" in FEATURES_CONFIG:
        n = FEATURES_CONFIG["MFCC"]
        mf = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n)
        feats.extend(mf.mean(1)); feats.extend(mf.var(1))

    if "SPECTRAL" in FEATURES_CONFIG:
        L = FEATURES_CONFIG["SPECTRAL"]
        if "centroid" in L: feats.append(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
        if "bandwidth" in L: feats.append(librosa.feature.spectral_bandwidth(y=y, sr=sr).mean())
        if "rolloff"  in L: feats.append(librosa.feature.spectral_rolloff(y=y, sr=sr).mean())
        if "zcr"     in L: feats.append(librosa.feature.zero_crossing_rate(y).mean())
        if "rms"     in L: feats.append(librosa.feature.rms(y=y).mean())

    if FEATURES_CONFIG.get("CONTRAST", False):
        feats.extend(librosa.feature.spectral_contrast(y=y, sr=sr).mean(1))

    if FEATURES_CONFIG.get("CHROMA", False):
        feats.extend(librosa.feature.chroma_stft(y=y, sr=sr).mean(1))

    if FEATURES_CONFIG.get("FLATNESS", False):
        feats.append(librosa.feature.spectral_flatness(y=y).mean())

    if FEATURES_CONFIG.get("STD", False):
        feats.append(np.std(y))

    if FEATURES_CONFIG.get("DIFF", False):
        feats.append(float(np.mean(np.abs(np.diff(y)))) if len(y) > 1 else 0.0)

    if FEATURES_CONFIG.get("BPM", False):
        try: feats.append(float(tempo(y=y, sr=sr)[0]) / 200.0)
        except: feats.append(0.0)

    arr = np.array(feats, np.float32)
    if arr.size != FEATURES_PER_WINDOW:
        raise ValueError(f"FEATURES_PER_WINDOW={FEATURES_PER_WINDOW} אבל קיבלנו {arr.size}")
    return arr

# =============================== EMBEDDING ==================================

def extract_segment_embedding(seg, sr):
    win = int(sr * WINDOW_STRIDE_SECONDS)
    out = []
    for i in range(WINDOWS_PER_SEGMENT):
        s = i * win
        e = s + win
        chunk = seg[s:e] if e <= len(seg) else np.pad(seg[s:], (0, e - len(seg)))
        out.append(extract_window_features(chunk, sr))
    return np.stack(out).astype(np.float32)

# =============================== META ========================================

def build_meta(bpm, key, energy, flat):
    vals = []
    for name in META_FEATURE_NAMES:
        if name == "BPM": vals.append(bpm)
        elif name == "Key": vals.append(key)
        elif name == "Energy": vals.append(energy)
        elif name == "Flatness": vals.append(flat)
        else: vals.append(0.0)
    return np.array(vals, np.float32)

# =============================== PROCESS TRACK ===============================

def process_track(mp3, out):
    try:
        y, sr = librosa.load(mp3, sr=SR, mono=True)
    except:
        return

    dur = len(y) / sr
    if dur < MIN_TRACK_DURATION_SEC or dur > MAX_TRACK_DURATION_SEC:
        return

    bpm, key = extract_global_features(y, sr)
    seg_len = int(SEGMENT_SECONDS * sr)

    for idx, pos in enumerate(SEGMENT_POSITIONS, 1):
        st = int(pos * dur * sr)
        ed = st + seg_len
        if ed > len(y): continue

        seg = y[st:ed]

        if silent_segment(seg, sr): continue
        if bad_chroma(seg, sr): continue

        emb = extract_segment_embedding(seg, sr)
        if bad_embedding(emb): continue

        energy, flat = extract_local_features(seg, sr)
        meta = build_meta(bpm, key, energy, flat)
        if bad_meta(meta): continue

        np.save(out / f"part_{idx}.npy", emb)
        np.save(out / f"part_{idx}_meta.npy", meta)

# =============================== RUN PARALLEL ================================

def run_parallel():
    tasks = []
    for genre in TRACKS_DIR.iterdir():
        if not genre.is_dir(): continue
        for artist in genre.iterdir():
            if not artist.is_dir(): continue
            for f in artist.iterdir():
                if f.suffix.lower() != ".mp3": continue
                out = DATA_DIR / genre.name / artist.name / f.stem
                out.mkdir(parents=True, exist_ok=True)
                tasks.append((f, out))

    print(f"🚀 יוצר דאטה עבור {len(tasks)} קבצים...")
    start = time.time(); done = 0

    with ProcessPoolExecutor() as ex:
        futures = [ex.submit(process_track, mp3, out) for mp3, out in tasks]
        for fut in as_completed(futures):
            done += 1
            bar = "#" * int(30 * done / len(tasks))
            print(f"\r[{bar.ljust(30,'-')}] {done}/{len(tasks)}", end="")

    print("\n✔ סיים!", round(time.time() - start, 2), "שניות")

# =============================== MAIN ========================================

if __name__ == "__main__":
    if DATA_DIR.exists():
        print("🗑 מוחק תיקיית DATA ישנה...")
        shutil.rmtree(DATA_DIR)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    run_parallel()
