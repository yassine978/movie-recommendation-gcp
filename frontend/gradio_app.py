import os
import requests
import gradio as gr

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

def api_health():
    r = requests.get(f"{API_BASE_URL}/health", timeout=20)
    r.raise_for_status()
    return r.json()

def api_search(query, limit):
    r = requests.get(f"{API_BASE_URL}/movies/search", params={"query": query, "limit": int(limit)}, timeout=20)
    r.raise_for_status()
    data = r.json()

    # data = list[dict] -> convert to rows
    rows = []
    for m in data:
        rows.append([m.get("movieId"), m.get("title"), m.get("genres")])
    return rows

def api_recommend(user_id, n):
    try:
        r = requests.get(
            f"{API_BASE_URL}/user/{int(user_id)}/recommendations",
            params={"n": int(n)},
            timeout=20
        )
        r.raise_for_status()
        payload = r.json()

        recs = payload.get("recommendations", [])
        rows = []
        for m in recs:
            rows.append([
                m.get("movieId"),
                m.get("title"),
                m.get("genres"),
                m.get("predicted_rating")
            ])

        return payload.get("recommendation_type"), payload.get("num_ratings"), rows

    except requests.RequestException as e:
        # On renvoie un message explicite dans reco_type, au lieu de "Erreur"
        return f"ERROR: {e}", 0, []

with gr.Blocks(title="Movie Recommender Demo") as demo:
    gr.Markdown("# 🎬 Movie Recommendation Demo (API FastAPI + Gradio)\nCompatible Workbench (sans WebSocket Streamlit).")

    with gr.Row():
        with gr.Column(scale=1):
            btn_health = gr.Button("Check API health")
            out_health = gr.JSON(label="API /health")

        with gr.Column(scale=2):
            gr.Markdown("### 🔎 Recherche de films")
            q = gr.Textbox(label="Titre", placeholder="Toy Story")
            limit = gr.Slider(5, 50, value=10, step=1, label="Max résultats")
            btn_search = gr.Button("Search")
            out_search = gr.Dataframe(headers=["movieId", "title", "genres"], label="Résultats", interactive=False)

    gr.Markdown("### 🎯 Recommandations")
    with gr.Row():
        user_id = gr.Number(value=1, label="user_id", precision=0)
        top_n = gr.Slider(5, 50, value=10, step=1, label="Top-N")
        btn_reco = gr.Button("Get recommendations", variant="primary")

    reco_type = gr.Textbox(label="recommendation_type")
    num_ratings = gr.Number(label="num_ratings", precision=0)
    reco_table = gr.Dataframe(
        headers=["movieId", "title", "genres", "predicted_rating"],
        label="Recommendations",
        interactive=False
    )

    btn_health.click(fn=api_health, outputs=out_health)
    btn_search.click(fn=api_search, inputs=[q, limit], outputs=out_search)
    btn_reco.click(fn=api_recommend, inputs=[user_id, top_n], outputs=[reco_type, num_ratings, reco_table])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

