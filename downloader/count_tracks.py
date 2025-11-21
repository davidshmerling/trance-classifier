import os

TRACKS_DIR = "tracks"


def count_mp3_for_artist(genre, artist):
    """
    סופר את כל קבצי ה־MP3 של אמן בתוך tracks/genre/artist
    """
    tracks_path = os.path.join(TRACKS_DIR, genre, artist)

    mp3_count = 0

    if os.path.isdir(tracks_path):
        for f in os.listdir(tracks_path):
            if f.lower().endswith(".mp3"):
                mp3_count += 1

    return mp3_count


def run():

    # סיכום גלובלי לכל הסגנונות
    global_mp3 = {}

    for genre in os.listdir(TRACKS_DIR):
        genre_path = os.path.join(TRACKS_DIR, genre)
        if not os.path.isdir(genre_path):
            continue

        genre_mp3_total = 0

        print(f"\n🎼 GENRE: {genre}")
        print("=" * 50)

        for artist in os.listdir(genre_path):
            artist_path = os.path.join(genre_path, artist)
            if not os.path.isdir(artist_path):
                continue

            mp3_count = count_mp3_for_artist(genre, artist)
            genre_mp3_total += mp3_count

            print(f"🎧 Artist: {artist}")
            print(f"   🎵 Tracks (mp3): {mp3_count}")
            print("-" * 40)

        print(f"\n📀 SUMMARY for GENRE: {genre}")
        print(f"   🎵 Total MP3: {genre_mp3_total}")
        print("=" * 50)

        global_mp3[genre] = genre_mp3_total

    print("\n📚📚 FINAL SUMMARY (ALL GENRES) 📚📚")
    print("=" * 55)
    for genre in global_mp3:
        print(f"🎼 {genre}:")
        print(f"   🎵 MP3 Total: {global_mp3[genre]}")
        print("-" * 40)
    print("=" * 55)


if __name__ == "__main__":
    print("📊 סופר טרקים לפי GENRE/ARTIST...\n")
    run()
    print("\n✔️ סיימתי!")
