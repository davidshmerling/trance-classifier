import streamlit as st
from frontend.utils.predict import predict_from_youtube_url

st.title("🎧 TranceClassifier – YouTube Track Analyzer")

url = st.text_input("הכנס קישור יוטיוב")

if st.button("נתח את השיר"):
    if not url.strip():
        st.error("נא להזין קישור")
    else:
        with st.spinner("מוריד ומנתח את הקובץ..."):
            try:
                result = predict_from_youtube_url(url)
                st.success(f"תוצאה: {result['best_genre']}")
                st.write(result["probs"])
            except Exception as e:
                st.error(f"שגיאה: {e}")

st.page_link("About.py", label="אודות", icon="ℹ️")
