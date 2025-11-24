# fronted/ui/About.py
import streamlit as st

st.title("ℹ️ אודות TranceClassifier")
st.write("""
TranceClassifier הוא פרויקט שמסווג שירי טראנס ל־Goa, Psy או Dark באמצעות מודל AI.

המערכת משלבת:
- חילוץ מאפיינים מאודיו (Embeddings)
- Meta data (BPM, Energy וכו')
- מודל Transformer מתקדם
- חסכון בזיכרון באמצעות הורדה זמנית בלבד
- טעינת המודל מה־GitHub האחרון

נבנה באהבה על ידי דוד שמרלינג 💙
""")

st.write("GitHub: https://github.com/davidshmerling/trance-classifier")
