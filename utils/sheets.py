import os, gspread
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from dotenv import load_dotenv

load_dotenv()

# נתיב לקובץ ה-Service Account מתוך .env
SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

# בדיקה אמיתית שהקובץ קיים
if not SERVICE_ACCOUNT_FILE or not os.path.isfile(SERVICE_ACCOUNT_FILE):
    raise FileNotFoundError(
        f"❌ GOOGLE_APPLICATION_CREDENTIALS לא נמצא: {SERVICE_ACCOUNT_FILE}"
    )

# טעינת האישורים
creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)

client = gspread.authorize(creds)
sheets = build(
    "sheets", "v4", credentials=creds, cache_discovery=False
).spreadsheets()


def get_trance_classifier_sheet(spreadsheet_id: str):
    """מחזיר את הטאב Trance_Classifier בלבד."""
    sh = client.open_by_key(spreadsheet_id)
    try:
        return sh.worksheet("Trance_Classifier")
    except gspread.WorksheetNotFound:
        raise Exception("❌ הטאב 'Trance_Classifier' לא נמצא בגיליון!")
