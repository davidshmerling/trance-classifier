import os
from mutagen.mp3 import MP3

TRACKS_DIR = "tracks"

MIN_TIME = 5 * 60      # 5 דקות
MAX_TIME = 15 * 60     # 15 דקות

# שלב 1 — מחיקת כל מה שלא MP3
def cleanup_non_mp3(folder):
    for root, dirs, files in os.walk(folder):
        for f in files:
            if not f.lower().endswith(".mp3"):
                full_path = os.path.join(root, f)
                try:
                    os.remove(full_path)
                    print(f"🗑️ נמחק (לא MP3): {full_path}")
                except Exception as e:
                    print(f"⚠️ שגיאה במחיקה של {full_path}: {e}")


# שלב 2 — מחיקת MP3 שאורכו לא בטווח
def check_track_length(path):
    try:
        audio = MP3(path)
        return audio.info.length
    except:
        return None


def cleanup_wrong_length(folder):
    for root, dirs, files in os.walk(folder):
        for f in files:
            if not f.lower().endswith(".mp3"):
                continue

            full_path = os.path.join(root, f)
            length = check_track_length(full_path)

            if length is None:
                print(f"⚠️ לא ניתן לקרוא: {full_path}")
                continue

            minutes = length / 60

            if length < MIN_TIME or length > MAX_TIME:
                try:
                    os.remove(full_path)
                    print(f"🗑️ נמחק (אורך לא תקין): {full_path} ({minutes:.1f} דקות)")
                except Exception as e:
                    print(f"⚠️ שגיאה במחיקה: {full_path}: {e}")
            else:
                print(f"✔️ תקין: {full_path} ({minutes:.1f} דקות)")


if __name__ == "__main__":
    print("🔍 מנקה את כל מה שלא MP3...")
    cleanup_non_mp3(TRACKS_DIR)

    print("\n⏱️ בודק אורכי שירים ומוחק חריגים...")
    cleanup_wrong_length(TRACKS_DIR)

    print("\n✔️ סיימתי!")
