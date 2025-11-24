import streamlit as st
from frontend.utils.predict import predict_from_youtube_url

st.set_page_config(
    page_title="TranceClassifier",
    page_icon="🎧",
)

# מצב עמוד – זוכר איפה אנחנו
if "page" not in st.session_state:
    st.session_state.page = "home"


# =====================================================
#                עמוד בית – הזנת קישור
# =====================================================
def render_home():
    st.title("🎧 TranceClassifier – ניתוח טראנס מיוטיוב")
    st.write("הכנס קישור ליוטיוב (5–15 דקות) והמודל יסווג ל־Goa / Psy / Dark.")

    url = st.text_input("קישור YouTube")

    if st.button("נתח את השיר"):
        if not url.strip():
            st.error("נא להזין קישור תקין.")
            return

        st.session_state.url = url
        st.session_state.page = "result"
        st.rerun()

    if st.button("ℹ️ אודות"):
        st.session_state.page = "about"
        st.rerun()


# =====================================================
#            עמוד תוצאה – לאחר חיזוי
# =====================================================
def render_result():
    st.title("🎼 תוצאות ניתוח השיר")

    url = st.session_state.get("url", "")

    if not url:
        st.warning("לא הוזן קישור.")
        st.session_state.page = "home"
        st.rerun()

    with st.spinner("מוריד ומנתח את השיר..."):
        try:
            result = predict_from_youtube_url(url)
        except Exception as e:
            st.error(f"שגיאה: {e}")
            if st.button("חזרה"):
                st.session_state.page = "home"
                st.rerun()
            return

    st.success(f"🔮 זיהוי סגנון: **{result['best_genre']}**")

    st.subheader("📊 הסתברויות:")
    for genre, prob in result["probs"].items():
        st.write(f"- **{genre}**: {prob:.3f}")

    st.divider()
    if st.button("🔄 בדיקה חדשה"):
        st.session_state.page = "home"
        st.rerun()


# =====================================================
#                  עמוד אודות
# =====================================================
def render_about():
    st.title("ℹ️ אודות TranceClassifier")
    st.write("""
    מערכת שמסווגת טראנס לתת־סגנונות: Goa, Psy, Dark.
    פותח על ידי דוד שמרלינג.
    המודל מבוסס על ניתוח אודיו, הפקת Embedding, ומודל אימון TensorFlow.
    """)

    if st.button("⬅️ חזרה"):
        st.session_state.page = "home"
        st.rerun()


# =====================================================
#                 ROUTING פשוט
# =====================================================
page = st.session_state.page

if page == "home":
    render_home()
elif page == "result":
    render_result()
elif page == "about":
    render_about()
