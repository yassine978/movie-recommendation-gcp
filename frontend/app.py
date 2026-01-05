import streamlit as st
from frontend.api_client import health, APIError

st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
)

st.title("🎬 Movie Recommendation System")
st.caption("Demo locale sur Vertex AI Workbench (API FastAPI + Streamlit)")

col1, col2 = st.columns([1, 2], vertical_alignment="center")

with col1:
    st.subheader("Status")
    try:
        h = health()
        st.success("API OK")
        st.json(h)
    except APIError as e:
        st.error("API indisponible")
        st.code(str(e))

with col2:
    st.subheader("Navigation")
    st.markdown(
        """
- **Recommandations** : démo utilisateur + paramètres.
- **Explorer** : recherche de films.
- **Diagnostics** : quelques métriques et graphiques de qualité.
        """
    )

st.info("Ensuite, va dans le menu Streamlit (à gauche) pour ouvrir les pages.")
