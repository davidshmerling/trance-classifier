import os

# תיקיות שלא מציגים בכלל
IGNORE_DIRS = {
    '.venv',
    '__pycache__',
    '.idea',
    '.git',
}

# קבצים שלא מציגים
IGNORE_FILE_EXT = {
    '.png', '.jpg', '.jpeg', '.mp3', '.wav',
    '.npy', '.mps', '.h5'
}

# תיקיות שמציגים **רק את השם שלהן**, בלי שום תוכן
EMPTY_TOP_DIRS = {'data', 'tracks', 'models'}


def print_tree(root_path, prefix=""):
    try:
        items = sorted(os.listdir(root_path))
    except PermissionError:
        return

    for i, item in enumerate(items):
        full = os.path.join(root_path, item)
        is_last = (i == len(items) - 1)
        connector = "└── " if is_last else "├── "

        # ------ תיקייה ------
        if os.path.isdir(full):

            # לא מציגים תיקיות חסומות
            if item in IGNORE_DIRS:
                continue

            print(prefix + connector + item + "/")

            # אם זו תיקייה שמוצגת ריקה → ממשיכים
            if item in EMPTY_TOP_DIRS:
                continue

            # אחרת יורדים פנימה כרגיל
            new_prefix = prefix + ("    " if is_last else "│   ")
            print_tree(full, new_prefix)

        else:
            # ------ קבצים ------
            _, ext = os.path.splitext(item)
            if ext.lower() in IGNORE_FILE_EXT:
                continue

            print(prefix + connector + item)


if __name__ == "__main__":
    print("Project tree:\n")
    print_tree(".")
