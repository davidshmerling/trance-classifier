# fronted/utils/youtube.py
from pathlib import Path
import yt_dlp
from make_models import config

def _build_default_ydl_opts(out_template: str) -> dict:
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


def check_youtube_duration(url: str) -> int:
    """Extract only metadata and return duration in seconds."""
    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True}) as ydl:
        info = ydl.extract_info(url, download=False)
    return info.get("duration", 0)


def download_audio_from_youtube(url: str, out_dir: Path) -> Path:
    """
    Downloads audio from YouTube, validating duration based on config.
    """

    # 1️⃣ בדיקת אורך הסרטון לפני הורדה
    duration = check_youtube_duration(url)

    if duration == 0:
        raise ValueError("Could not read video duration")

    if duration < config.MIN_TRACK_DURATION_SEC:
        raise ValueError(f"Video too short ({duration}s). Minimum required is {config.MIN_TRACK_DURATION_SEC}s")

    if duration > config.MAX_TRACK_DURATION_SEC:
        raise ValueError(f"Video too long ({duration}s). Maximum allowed is {config.MAX_TRACK_DURATION_SEC}s")

    # 2️⃣ הורדה בפועל
    out_dir.mkdir(parents=True, exist_ok=True)
    out_template = str(out_dir / "%(id)s.%(ext)s")

    default_opts = _build_default_ydl_opts(out_template)
    ydl_opts = getattr(config, "YDL_OPTS", default_opts)

    ydl_opts = dict(ydl_opts)
    ydl_opts["outtmpl"] = out_template

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        track_id = info.get("id")
        ext = "mp3"
        audio_path = out_dir / f"{track_id}.{ext}"

    # fallback — אם לא נוצר mp3
    if not audio_path.exists():
        candidates = list(out_dir.glob(f"{track_id}.*"))
        if not candidates:
            raise FileNotFoundError("Audio file not found after download.")
        audio_path = candidates[0]

    return audio_path
