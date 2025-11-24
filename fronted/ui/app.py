# fronted/ui/app.py
import streamlit as st
from fronted.utils.predict import predict_from_youtube_url

st.set_page_config(
    page_title="TranceClassifier",
    page_icon="🎧",
)

st.title("🎧 TranceClassifier – YouTube Track Analyzer")
st.write(
    "Paste a YouTube link to a trance track (5–15 minutes). "
    "The model will classify it into **Goa / Psy / Dark**."
)

url = st.text_input("YouTube URL", "")

if st.button("Analyze track"):
    if not url.strip():
        st.error("Please enter a valid URL.")
    else:
        with st.spinner("Downloading and analyzing track..."):
            try:
                result = predict_from_youtube_url(url)
            except Exception as e:
                st.error(f"Error while processing: {e}")
            else:
                st.success(f"Predicted subgenre: **{result['best_genre']}**")

                st.subheader("Probabilities")
                for genre, prob in result["probs"].items():
                    st.write(f"- **{genre}**: {prob:.3f}")
