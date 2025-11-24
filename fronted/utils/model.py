# fronted/utils/model.py
from pathlib import Path
import urllib.request
import tensorflow as tf

try:
    from make_models import config
except ImportError:
    import config


DEFAULT_MODEL_URL = (
    "https://raw.githubusercontent.com/davidshmerling/trance-classifier/main/"
    "make_models/models/latest.h5"
)

GITHUB_MODEL_URL = getattr(config, "GITHUB_MODEL_URL", DEFAULT_MODEL_URL)
MODEL_CACHE_DIR = Path(getattr(config, "MODEL_CACHE_DIR", ".model_cache"))
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

LOCAL_MODEL_PATH = MODEL_CACHE_DIR / getattr(config, "LATEST_MODEL_NAME", "latest.h5")


def _download_model_if_needed() -> Path:
    if LOCAL_MODEL_PATH.exists():
        print(f"✅ Using cached model: {LOCAL_MODEL_PATH}")
        return LOCAL_MODEL_PATH

    print("⬇ Downloading latest model from GitHub...")
    urllib.request.urlretrieve(GITHUB_MODEL_URL, LOCAL_MODEL_PATH)
    print(f"✅ Model downloaded → {LOCAL_MODEL_PATH}")
    return LOCAL_MODEL_PATH


def load_model() -> tf.keras.Model:
    model_path = _download_model_if_needed()
    model = tf.keras.models.load_model(model_path)
    print("✅ Model loaded.")
    return model
