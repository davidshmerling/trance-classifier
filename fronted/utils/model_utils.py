import numpy as np
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from pathlib import Path

MODEL_PATH = Path("make _models/models/latest.h5")
VALID_GENRES = ["goa", "psy", "dark", "none"]

_cached_model = None


def load_latest_model():
    """Loads the latest model once and caches it."""
    global _cached_model

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"❌ Model file not found: {MODEL_PATH}")

    if _cached_model is None:
        _cached_model = load_model(MODEL_PATH)

    return _cached_model


def prepare_image(img_path: Path):
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    img = load_img(img_path, target_size=(299, 299))
    img = img_to_array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img, axis=0)  # (1,299,299,3)


def prepare_embedding(matrix_10x68: np.ndarray):
    if matrix_10x68.shape != (10, 68):
        raise ValueError(f"Embedding must be shape (10, 68), got {matrix_10x68.shape}")
    return np.expand_dims(matrix_10x68, axis=0)  # (1,10,68)


def prepare_meta(meta_vec4: np.ndarray):
    if meta_vec4.shape != (4,):
        raise ValueError(f"Meta vector must be (4,), got {meta_vec4.shape}")
    return np.expand_dims(meta_vec4, axis=0)  # (1,4)


def predict_track(model, img, emb, meta_vec4):
    pred = model.predict([img, emb, meta_vec4])[0]
    return pred


def get_final_label(pred_vector):
    idx = np.argmax(pred_vector)
    return VALID_GENRES[idx]
