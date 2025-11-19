import os

TRACKS_DIR = "tracks"
IMAGES_DIR = "images"

def count_for_artist(genre, artist):
    tracks_path = os.path.join(TRACKS_DIR, genre, artist)
    images_path = os.path.join(IMAGES_DIR, genre, artist)

    mp3_count = 0
    img_count = 0

    # ספירת MP3
    if os.path.isdir(tracks_path):
        for f in os.listdir(tracks_path):
            if f.lower().endswith(".mp3"):
                mp3_count += 1

    # ספירת תמונות
    if os.path.isdir(images_path):
        for f in os.listdir(images_path):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                img_count += 1

    return mp3_count, img_count


def run():

    # --- תוספת קטנה: סיכום כללי לכל הסגנונות ---
    global_mp3 = {}
    global_img = {}
    # ------------------------------------------------

    for genre in os.listdir(TRACKS_DIR):
        genre_path = os.path.join(TRACKS_DIR, genre)
        if not os.path.isdir(genre_path):
            continue

        # קאונטרים לכל סגנון
        genre_mp3_total = 0
        genre_img_total = 0

        print(f"\n🎼 GENRE: {genre}")
        print("=" * 50)

        for artist in os.listdir(genre_path):
            artist_path = os.path.join(genre_path, artist)
            if not os.path.isdir(artist_path):
                continue

            mp3_count, img_count = count_for_artist(genre, artist)

            genre_mp3_total += mp3_count
            genre_img_total += img_count

            print(f"🎧 Artist: {artist}")
            print(f"   🎵 Tracks (mp3): {mp3_count}")
            print(f"   🖼️ Images: {img_count}")
            print("-" * 40)

        # סיכום לסגנון
        print(f"\n📀 SUMMARY for GENRE: {genre}")
        print(f"   🎵 Total MP3: {genre_mp3_total}")
        print(f"   🖼️ Total Images: {genre_img_total}")
        print("=" * 50 + "\n")

        # --- שמירה בגלובל ---
        global_mp3[genre] = genre_mp3_total
        global_img[genre] = genre_img_total
        # -----------------------


    # --- סיכום סופי אחרי כל האמנים והסגנונות ---
    print("\n📚📚 FINAL SUMMARY (ALL GENRES) 📚📚")
    print("=" * 55)
    for genre in global_mp3:
        print(f"🎼 {genre}:")
        print(f"   🎵 MP3 Total: {global_mp3[genre]}")
        print(f"   🖼️ Images:    {global_img[genre]}")
        print("-" * 40)
    print("=" * 55)
    # ------------------------------------------------


if __name__ == "__main__":
    print("📊 סופר טרקים ותמונות לכל אומן...\n")
    run()
    print("\n✔️ סיימתי!")
