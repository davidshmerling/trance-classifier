import yt_dlp
import re
from pathlib import Path
from mutagen.mp3 import MP3

TMP_DIR = Path("tmp")
TMP_DIR.mkdir(exist_ok=True)

# ============================
# 📌 Validators
# ============================

YOUTUBE_REGEX = re.compile(
    r"^(https?://)?(www\.)?(youtube\.com|youtu\.be)/.+"
)


def is_valid_youtube_url(url: str) -> bool:
    return bool(YOUTUBE_REGEX.match(url))


# ============================
# 📌 Extract metadata (duration + filesize)
# ============================

def get_video_info(url: str):
    """
    מחזיר:
      duration_sec
      filesize_bytes  (יכול להיות None)
    """
    opts = {"quiet": True}

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)

    duration = float(info.get("duration", 0))

    # filesize may be missing → try both options
    filesize = info.get("filesize") or info.get("filesize_approx")

    return duration, filesize


# ============================
# 📌 Download MP3 safely
# ============================

def download_audio(url: str, max_mb: float = 20.0) -> tuple[Path, str | None]:
    """
    מוריד אודיו מיוטיוב רק אם:
      - ה-URL תקין
      - גודל הקובץ קטן מ-max_mb
    מחזיר:
      (path_to_mp3, error_str)
    """

    # ------- בדיקת URL -------
    if not is_valid_youtube_url(url):
        return None, "⛔ לינק יוטיוב לא תקין"

    # ------- בדיקת מידע על הוידאו -------
    duration, filesize = get_video_info(url)

    if duration < 300:
        return None, "⛔ הסרטון קצר מדי (מינימום 5 דקות)"

    if duration > 900:
        return None, "⛔ הסרטון ארוך מדי (מקסימום 15 דקות)"

    # ------- בדיקת גודל לפני הורדה -------
    if filesize is not None:
        mb = filesize / (1024 * 1024)
        if mb > max_mb:
            return None, f"⛔ גודל אודיו גדול מדי ({mb:.1f}MB). המקסימום: {max_mb}MB"

    # ------- הורדה -------
    TMP_DIR.mkdir(exist_ok=True)
    out_path = TMP_DIR / "audio.mp3"

    if out_path.exists():
        out_path.unlink()

    ydl_opts = {
        "format": "bestaudio/best",
        "quiet": True,
        "outtmpl": str(out_path),
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "128",
        }],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        return None, f"⛔ שגיאת הורדה: {e}"

    # בדיקה שהקובץ באמת ירד
    if not out_path.exists() or out_path.stat().st_size == 0:
        return None, "⛔ הורדה נכשלה — קובץ ריק"

    return out_path, None



def validate_duration(length_sec: float):
    """
    בדיקה פשוטה — משמשת בשלב מוקדם לפני הורדה.
    שיר קצר מדי או ארוך מדי → לא תקין.
    """
    if length_sec is None:
        return False, "לא הצלחתי לזהות את אורך הסרטון"

    if length_sec < 5 * 60:
        return False, "הסרטון קצר מדי (פחות מ־5 דקות)"

    if length_sec > 15 * 60:
        return False, "הסרטון ארוך מדי (מעל 15 דקות)"

    return True, "OK"
