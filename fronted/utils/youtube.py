# fronted/utils/youtube.py
from pathlib import Path
import yt_dlp
from make_models import config

def _build_default_ydl_opts(out_template: str) -> dict:
    """Default yt-dlp options. אפשר לעקוף דרך config.YDL_OPTS אם תרצה."""
    return {
        "format": "bestaudio/best",
        "outtmpl": out_template,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
    }


def download_audio_from_youtube(url: str, out_dir: Path) -> Path:
    """
    מוריד אודיו מיוטיוב לפי ההגדרות ב־config (אם קיימות),
    ומחזיר את הנתיב לקובץ ה־audio.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_template = str(out_dir / "%(id)s.%(ext)s")

    default_opts = _build_default_ydl_opts(out_template)
    ydl_opts = getattr(config, "YDL_OPTS", default_opts)

    # דואג שתבנית ה-outtmpl תהיה תמיד עם התיקייה הנכונה
    ydl_opts = dict(ydl_opts)
    ydl_opts["outtmpl"] = out_template

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        track_id = info.get("id")
        ext = "mp3"
        audio_path = out_dir / f"{track_id}.{ext}"

    if not audio_path.exists():
        candidates = list(out_dir.glob(f"{track_id}.*"))
        if not candidates:
            raise FileNotFoundError("Audio file not found after download.")
        audio_path = candidates[0]

    return audio_path
