import os
import re
import yt_dlp
import time
from tqdm import tqdm
from mutagen.mp3 import MP3
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.sheets import get_trance_classifier_sheet

SPREADSHEET_ID = "17CUBZkT5_OK4tBGe7Rf_83x84on-BXu7xuWVZ64nyjU"

TRACKS_DIR = "tracks"
os.makedirs(TRACKS_DIR, exist_ok=True)

def clean(s):
    return re.sub(r'[\\/*?:"<>|]', "", str(s))


def get_track_urls(artist_url):
    opts = {
        "skip_download": True,
        "extract_flat": True,
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(opts) as y:
        info = y.extract_info(artist_url, download=False)

    urls = []
    for e in info.get("entries", []):
        if not e:
            continue
        url = e.get("url")
        if not url:
            continue
        if not url.startswith("http"):
            url = "https://soundcloud.com/" + url
        urls.append(url)

    return list(set(urls))


def download_single(url, out_dir, artist, genre, max_size_mb=20):
    final_path = None

    def progress_hook(d):
        nonlocal final_path
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            if total and total > max_size_mb * 1024 * 1024:
                raise Exception("File too large")

        elif d.get("status") == "finished":
            final_path = d.get("filename")

    outtmpl = f"{out_dir}/%(title)s__{clean(artist)}__%(id)s__{clean(genre)}.%(ext)s"
    opts = {
        "quiet": True,
        "no_warnings": True,
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "progress_hooks": [progress_hook],
        "keepvideo": False,
        "concurrent_fragment_downloads": 1,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "128",
            "nopostoverwrites": False,
        }],
    }

    with yt_dlp.YoutubeDL(opts) as y:
        info = y.extract_info(url, download=True)

    if final_path and os.path.exists(final_path):
        try:
            audio = MP3(final_path)
            length = audio.info.length
            if length > 900:
                os.remove(final_path)
            elif length < 300:
                os.remove(final_path)
        except:
            os.remove(final_path)

    return True


def download_parallel(urls, out_dir, artist, genre):
    total = len(urls)
    count = 0

    with ThreadPoolExecutor(max_workers=5) as ex:   # ⭐ 5 תרדים
        futures = {ex.submit(download_single, url, out_dir, artist, genre): url for url in urls}

        for future in as_completed(futures):
            url = futures[future]
            try:
                future.result()
                count += 1
                left = total - count
                print(f"🎵 ירד {count} מתוך {total} | נשאר {left}")
            except Exception as e:
                print(f"⚠️ שגיאה בהורדה של {url}: {e}")



def run():
    ws = get_trance_classifier_sheet(SPREADSHEET_ID)
    rows = ws.get_all_records()

    for i, row in enumerate(rows, start=2):

        artist = row["Artist Name"]
        genre = row["Genre"]
        link = row["Link"]

        print(f"\n🎧 עובד על: {artist}")

        playlist_name = row.get("Playlist Name", "").strip()
        if not playlist_name:
            playlist_name = f"{artist} - All Tracks"
            ws.update_cell(i, 1, playlist_name)
            print(f"📌 עודכן Playlist Name → {playlist_name}")

        status = row.get("Download Status", "").strip()
        already_downloaded = (status != "")
        if already_downloaded:
            print(f"✔ {artist} כבר ירד — מדלג")
            continue

        urls = get_track_urls(link)
        print(f"✓ נמצאו {len(urls)} קישורים")

        out_dir = f"{TRACKS_DIR}/{clean(genre)}/{clean(artist)}"
        os.makedirs(out_dir, exist_ok=True)

        print(f"⏳ מתחיל הורדה של {artist}")

        download_parallel(urls, out_dir, artist, genre)

        # ****** UPDATE SIMPLE STATUS *******
        ws.update_cell(i, 5, "downloaded")
        print(f"✔️ {artist} הוגדר כ-download")

    print("\n✔ כל ההורדות הסתיימו!")


if __name__ == "__main__":
    start = time.time()
    run()
    print(f"\n⏱ זמן כולל: {time.time() - start:.2f} שניות")
