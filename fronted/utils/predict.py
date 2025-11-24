# fronted/utils/predict.py
import tempfile
import numpy as np

try:
    from make_models import config
except ImportError:
    import config

from .youtube import download_audio_from_youtube
from .data import (
    load_and_validate_audio,
    extract_segment_at_position,
    build_embedding_for_segment,
    build_meta_for_track,
)
from .model import load_model


USE_EMBEDDING = config.USE_EMBEDDING
USE_META = config.USE_META
SEGMENT_POSITIONS = config.SEGMENT_POSITIONS
VALID_GENRES = config.VALID_GENRES


def predict_from_youtube_url(url: str) -> dict:
    """
    Pipeline מלא:
    - הורדה מיוטיוב
    - טעינה + בדיקות לפי config
    - חיתוך לפי SEGMENT_POSITIONS
    - הפקת embedding/meta לפי config
    - הרצה על המודל
    - ממוצע תחזיות על פני כל המקטעים
    """
    model = load_model()

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = download_audio_from_youtube(url, Path(tmpdir))

        y, sr, _ = load_and_validate_audio(audio_path)

        all_preds = []

        for pos in SEGMENT_POSITIONS:
            seg = extract_segment_at_position(y, sr, pos)

            emb = build_embedding_for_segment(seg, sr) if USE_EMBEDDING else None
            meta = build_meta_for_track(seg, sr) if USE_META else None

            if USE_EMBEDDING and USE_META:
                model_input = [emb, meta]
            elif USE_EMBEDDING:
                model_input = emb
            elif USE_META:
                model_input = meta
            else:
                raise ValueError(
                    "Both USE_EMBEDDING and USE_META are False – no inputs for model."
                )

            preds = model.predict(model_input, verbose=0)[0]
            all_preds.append(preds)

        mean_preds = np.mean(np.stack(all_preds, axis=0), axis=0)

        best_idx = int(np.argmax(mean_preds))
        best_genre = VALID_GENRES[best_idx] if best_idx < len(VALID_GENRES) else str(best_idx)

        return {
            "probs": {g: float(p) for g, p in zip(VALID_GENRES, mean_preds)},
            "best_genre": best_genre,
        }
