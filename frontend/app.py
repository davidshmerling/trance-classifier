# fronted/ui/app.py
import streamlit as st

# רק רישום האתר – בלי תוכן
st.set_page_config(
    page_title="TranceClassifier",
    page_icon="🎧",
)

st.switch_page("Home.py")
