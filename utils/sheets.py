import os, gspread
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from dotenv import load_dotenv

load_dotenv()

SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

if not SERVICE_ACCOUNT_FILE or not os.path.isfile(SERVICE_ACCOUNT_FILE):
    raise FileNotFoundError(f"Missing GOOGLE_APPLICATION_CREDENTIALS at {SERVICE_ACCOUNT_FILE}")

creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)

client = gspread.authorize(creds)
sheets = build("sheets", "v4", credentials=creds, cache_discovery=False).spreadsheets()



def get_trance_classifier_sheet(spreadsheet_id: str):
    """מחזיר את הטאב Trance_Classifier בלבד."""
    sh = client.open_by_key(spreadsheet_id)
    try:
        return sh.worksheet("Trance_Classifier")
    except gspread.WorksheetNotFound:
        raise Exception("❌ הטאב 'Trance_Classifier' לא נמצא בגיליון!")