import streamlit as st
import pandas as pd
from frontend.api_client import get_recommendations, popular_movies, APIError

st.set_page_config(page_title="Recommandations", page_icon="🎯", layout="wide")
st.title("🎯 Recommandations personnalisées")

with st.sidebar:
    st.header("Paramètres")
    user_id = st.number_input("User ID", min_value=1, value=1, step=1)
    top_n = st.slider("Top-N", 5, 50, 10)
    st.caption("Si l'utilisateur n'a pas d'historique, on bascule en mode cold-start.")

colA, colB = st.columns([2, 1], vertical_alignment="top")

with colA:
    if st.button("Obtenir recommandations", type="primary"):
        try:
            payload = get_recommendations(int(user_id), int(top_n))

            st.subheader(f"Résultat pour user_id={payload.get('user_id')}")
            st.write(f"Type: `{payload.get('recommendation_type')}` | "
                     f"Nb ratings connus: **{payload.get('num_ratings')}**")

            recs = payload.get("recommendations", [])
            if not recs:
                st.warning("Aucune recommandation retournée.")
            else:
                df = pd.DataFrame(recs)
                # Normaliser colonnes si besoin
                cols = [c for c in ["movieId", "title", "genres", "predicted_rating"] if c in df.columns]
                df = df[cols].copy()

                if "predicted_rating" in df.columns:
                    df = df.sort_values("predicted_rating", ascending=False)

                st.dataframe(df, use_container_width=True, hide_index=True)

        except APIError as e:
            st.error("Impossible d'obtenir des recommandations (fallback cold-start).")
            st.code(str(e))

            # fallback simple
            try:
                st.subheader("⭐ Suggestion cold-start (popular)")
                data = popular_movies(n=int(top_n))
                if data:
                    st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
                else:
                    st.info("Cold-start handler désactivé (pas de ratings locaux). On ajoutera un fallback côté API.")
            except Exception:
                st.info("Cold-start non disponible pour le moment.")

with colB:
    st.subheader("Conseils demo")
    st.markdown(
        """
- Teste plusieurs `user_id` (ex: 1, 10, 100).
- Augmente/diminue `Top-N`.
- Compare les genres dominants dans la liste.
        """
    )
