from pathlib import Path
import numpy as np

from utils.youtube_utils import download_audio, get_video_length, validate_duration
from utils.audio_utils import split_into_6_segments, extract_10x68, extract_meta, create_spectrogram_png
from utils.model_utils import (
    load_latest_model,
    prepare_image,
    prepare_embedding,
    prepare_meta,
    predict_track,
    get_final_label
)
from utils.clean_tmp import cleanup_tmp


def analyze(url):

    # --- 1) validate ---
    length = get_video_length(url)
    ok, msg = validate_duration(length)
    if not ok:
        return {"error": msg}

    # --- 2) download MP3 ---
    mp3, err = download_audio(url)
    if err: return {"error": err}
    mp3 = Path(mp3)

    # --- 3) split into 6 segments ---
    segments, sr = split_into_6_segments(mp3)

    model = load_latest_model()
    preds = []

    # --- 4) process each segment ---
    for idx, seg in enumerate(segments):
        img_path = create_spectrogram_png(seg, sr, idx)
        emb = extract_10x68(seg, sr)
        meta = extract_meta(seg, sr)

        X_img = prepare_image(img_path)
        X_emb = prepare_embedding(emb)
        X_meta = prepare_meta(meta)

        preds.append(predict_track(model, X_img, X_emb, X_meta))

    preds = np.array(preds)  # (6, num_classes)
    avg = preds.mean(axis=0)

    # אם המודל לא כולל none — נוסיף
    if len(avg) == 3:
        g, p, d = avg
        none = max(0, 1 - (g + p + d))
        avg = np.array([g, p, d, none], dtype=np.float32)

    final = get_final_label(avg)

    cleanup_tmp()

    return {
        "goa": float(avg[0]),
        "psy": float(avg[1]),
        "dark": float(avg[2]),
        "none": float(avg[3]),
        "final": final
    }
