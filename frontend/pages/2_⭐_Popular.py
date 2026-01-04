import streamlit as st
from frontend.api_client import popular_movies, APIError

st.set_page_config(page_title="Popular", page_icon="⭐", layout="wide")
st.title("⭐ Films populaires (cold start)")

n = st.slider("Top N", 5, 50, 10)
genre = st.text_input("Genre (optionnel)", placeholder="Ex: Comedy")

if st.button("Charger"):
    try:
        data = popular_movies(n=n, genre=genre.strip() or None)
        st.dataframe(data, use_container_width=True)
        if not data:
            st.warning("Aucune donnée retournée (cold start handler peut être désactivé).")
    except APIError as e:
        st.error(str(e))
