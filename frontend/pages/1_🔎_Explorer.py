import streamlit as st
from frontend.api_client import search_movies, APIError

st.set_page_config(page_title="Explorer", page_icon="🔎", layout="wide")
st.title("🔎 Explorer les films")

query = st.text_input("Rechercher un film par titre", placeholder="Ex: Toy Story")
limit = st.slider("Nombre de résultats", 5, 50, 10)

if query:
    try:
        results = search_movies(query=query, limit=limit)
        st.write(f"Résultats: {len(results)}")
        st.dataframe(results, use_container_width=True)
    except APIError as e:
        st.error(str(e))
else:
    st.info("Entre un titre pour lancer une recherche.")
