# fronted/utils/data.py
from pathlib import Path
import numpy as np
import librosa
from librosa.feature.rhythm import tempo

try:
    from make_models import config
except ImportError:
    import config


SR = config.SR
SEGMENT_SECONDS = config.SEGMENT_SECONDS
WINDOW_STRIDE_SECONDS = config.WINDOW_STRIDE_SECONDS
WINDOWS_PER_SEGMENT = config.WINDOWS_PER_SEGMENT
SEGMENT_POSITIONS = config.SEGMENT_POSITIONS
MIN_TRACK_DURATION_SEC = config.MIN_TRACK_DURATION_SEC
MAX_TRACK_DURATION_SEC = config.MAX_TRACK_DURATION_SEC
CHECK_SILENCE = config.CHECK_SILENCE
ENERGY_THRESHOLD = config.ENERGY_THRESHOLD

FEATURES_CONFIG = config.FEATURES_CONFIG
EMB_SHAPE = config.EMB_SHAPE

USE_META = config.USE_META
META_FEATURE_NAMES = config.META_FEATURE_NAMES
META_VECTOR_LENGTH = config.META_VECTOR_LENGTH


# -------------------------------------------------
# 1. טעינת אודיו + בדיקות מהקונפיג
# -------------------------------------------------
def load_and_validate_audio(path: Path):
    y, sr = librosa.load(path, sr=SR, mono=True)
    duration_sec = len(y) / sr

    if duration_sec < MIN_TRACK_DURATION_SEC or duration_sec > MAX_TRACK_DURATION_SEC:
        # לא זורקים שגיאה – רק אזהרה. אם תרצה, אפשר להפוך ל-Exception.
        print(
            f"⚠ Track length {duration_sec:.1f}s outside "
            f"[{MIN_TRACK_DURATION_SEC}, {MAX_TRACK_DURATION_SEC}]"
        )

    if CHECK_SILENCE:
        energy = np.mean(y ** 2)
        if energy < ENERGY_THRESHOLD:
            print(f"⚠ Low energy track (mean energy={energy:.6f})")

    return y, sr, duration_sec


# -------------------------------------------------
# 2. חיתוך ל-window-ים לפי הקונפיג
# -------------------------------------------------
def split_to_windows(y: np.ndarray, sr: int) -> np.ndarray:
    win_len = int(sr * WINDOW_STRIDE_SECONDS)
    total_needed = WINDOWS_PER_SEGMENT * win_len

    # אם הקטע קצר – padding
    if len(y) < total_needed:
        pad = np.zeros(total_needed - len(y), dtype=y.dtype)
        y = np.concatenate([y, pad])

    windows = []
    for i in range(WINDOWS_PER_SEGMENT):
        start = i * win_len
        end = start + win_len
        win = y[start:end]
        windows.append(win)

    return np.stack(windows, axis=0)  # (windows_per_segment, samples_per_window)


# -------------------------------------------------
# 3. חישוב פיצ’רים לפי FEATURES_CONFIG
# -------------------------------------------------
def _features_for_window(win: np.ndarray, sr: int) -> np.ndarray:
    feats = []

    # ----- MFCC mean + var -----
    if "MFCC" in FEATURES_CONFIG:
        n_mfcc = FEATURES_CONFIG["MFCC"]
        mfcc = librosa.feature.mfcc(y=win, sr=sr, n_mfcc=n_mfcc)
        feats.append(mfcc.mean(axis=1))
        feats.append(mfcc.var(axis=1))  # var ולא std – כמו בקונספט שלך

    # ----- Spectral בסיסי -----
    if "SPECTRAL" in FEATURES_CONFIG:
        names = FEATURES_CONFIG["SPECTRAL"]
        S = np.abs(librosa.stft(win)) + 1e-9

        if "centroid" in names:
            c = librosa.feature.spectral_centroid(S=S, sr=sr).mean()
            feats.append(np.array([c]))
        if "bandwidth" in names:
            b = librosa.feature.spectral_bandwidth(S=S, sr=sr).mean()
            feats.append(np.array([b]))
        if "rolloff" in names:
            r = librosa.feature.spectral_rolloff(S=S, sr=sr).mean()
            feats.append(np.array([r]))

    # ----- Contrast -----
    if FEATURES_CONFIG.get("CONTRAST", False):
        contrast = librosa.feature.spectral_contrast(y=win, sr=sr).mean(axis=1)
        feats.append(contrast)

    # ----- Chroma -----
    if FEATURES_CONFIG.get("CHROMA", False):
        chroma = librosa.feature.chroma_stft(y=win, sr=sr).mean(axis=1)
        feats.append(chroma)

    # ----- Flatness -----
    if FEATURES_CONFIG.get("FLATNESS", False):
        flat = librosa.feature.spectral_flatness(y=win).mean()
        feats.append(np.array([flat]))

    # ----- STD של האות -----
    if FEATURES_CONFIG.get("STD", False):
        feats.append(np.array([np.std(win)]))

    # ----- DIFF (ממוצע |Δx|) -----
    if FEATURES_CONFIG.get("DIFF", False):
        diff = np.mean(np.abs(np.diff(win)))
        feats.append(np.array([diff]))

    # ----- BPM -----
    if FEATURES_CONFIG.get("BPM", False):
        bpm = float(tempo(y=win, sr=sr)[0])
        feats.append(np.array([bpm]))

    return np.concatenate(feats, axis=0)


def build_embedding_for_segment(y_seg: np.ndarray, sr: int) -> np.ndarray:
    """
    מחזיר embedding בגודל EMB_SHAPE כמו באימון.
    """
    windows = split_to_windows(y_seg, sr)
    feat_rows = []

    for win in windows:
        row = _features_for_window(win, sr)
        feat_rows.append(row)

    emb = np.stack(feat_rows, axis=0)  # (WINDOWS_PER_SEGMENT, FEATURES_PER_WINDOW)

    # לוודא תואם EMB_SHAPE (אם יש הבדלי עיגול)
    if emb.shape != EMB_SHAPE:
        # חיתוך/פדינג קל במקרה חריג
        T, F = EMB_SHAPE
        t, f = emb.shape

        if t < T:
            pad_t = np.zeros((T - t, f), dtype=emb.dtype)
            emb = np.concatenate([emb, pad_t], axis=0)
        elif t > T:
            emb = emb[:T]

        if f < F:
            pad_f = np.zeros((T, F - f), dtype=emb.dtype)
            emb = np.concatenate([emb, pad_f], axis=1)
        elif f > F:
            emb = emb[:, :F]

    emb = emb.astype("float32")
    emb = np.expand_dims(emb, axis=0)  # (1, T, F)
    return emb


# -------------------------------------------------
# 4. חיתוך קטע לפי SEGMENT_POSITIONS
# -------------------------------------------------
def extract_segment_at_position(y: np.ndarray, sr: int, center_ratio: float) -> np.ndarray:
    total_sec = len(y) / sr
    center_time = total_sec * center_ratio
    half = SEGMENT_SECONDS / 2.0

    start_time = max(0.0, center_time - half)
    end_time = min(total_sec, center_time + half)

    start_idx = int(start_time * sr)
    end_idx = int(end_time * sr)
    seg = y[start_idx:end_idx]

    # padding אם קצר מדי
    target_len = int(SEGMENT_SECONDS * sr)
    if len(seg) < target_len:
        pad = np.zeros(target_len - len(seg), dtype=seg.dtype)
        seg = np.concatenate([seg, pad])
    else:
        seg = seg[:target_len]

    return seg


# -------------------------------------------------
# 5. META לפי META_FEATURE_NAMES
# -------------------------------------------------
def build_meta_for_track(y: np.ndarray, sr: int) -> np.ndarray | None:
    if not USE_META or META_VECTOR_LENGTH <= 0:
        return None

    duration = len(y) / sr
    rms = librosa.feature.rms(y=y).mean()
    bpm = float(tempo(y=y, sr=sr)[0])
    flat = librosa.feature.spectral_flatness(y=y).mean()

    values = []
    for name in META_FEATURE_NAMES:
        if name == "BPM":
            values.append(bpm)
        elif name == "Key":
            # אפשר לשפר בעתיד – כרגע placeholder 0.0
            values.append(0.0)
        elif name == "Energy":
            values.append(rms)
        elif name == "Flatness":
            values.append(flat)
        else:
            values.append(0.0)

    vec = np.array(values, dtype="float32")

    if len(vec) < META_VECTOR_LENGTH:
        pad = np.zeros(META_VECTOR_LENGTH - len(vec), dtype="float32")
        vec = np.concatenate([vec, pad])
    elif len(vec) > META_VECTOR_LENGTH:
        vec = vec[:META_VECTOR_LENGTH]

    return np.expand_dims(vec, axis=0)  # (1, META_VECTOR_LENGTH)
