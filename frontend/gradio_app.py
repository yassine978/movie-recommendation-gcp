import requests
import pandas as pd
import gradio as gr

API_BASE = "http://127.0.0.1:8000"


# ---------------------------
# Helpers HTTP
# ---------------------------
def _get(path, params=None):
    r = requests.get(f"{API_BASE}{path}", params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def _post(path, json=None):
    r = requests.post(f"{API_BASE}{path}", json=json, timeout=30)
    r.raise_for_status()
    return r.json()

def _to_df(items):
    if items is None:
        return pd.DataFrame()
    if isinstance(items, dict) and "similar_movies" in items:
        items = items["similar_movies"]
    if isinstance(items, list):
        return pd.DataFrame(items)
    return pd.DataFrame([items])


# ---------------------------
# Actions
# ---------------------------
def api_health():
    return _get("/health")

def api_meta():
    return _get("/meta")

def api_search(title, limit):
    return _to_df(_get("/movies/search", params={"query": title, "limit": int(limit)}))

def api_movie_details(movie_id):
    return _get(f"/movies/{int(movie_id)}")

def api_similar(movie_id, n):
    return _get(f"/movies/{int(movie_id)}/similar", params={"n": int(n)})

def api_popular(n, genre):
    params = {"n": int(n)}
    if genre and genre.strip():
        params["genre"] = genre.strip()
    return _to_df(_get("/movies/popular", params=params))

def api_user_ratings(user_id):
    return _get(f"/user/{int(user_id)}/ratings")

def api_reset_user(user_id):
    return _post(f"/user/{int(user_id)}/reset")

def api_submit_ratings(user_id, ratings_df: pd.DataFrame):
    # ratings_df has columns movieId, rating
    if ratings_df is None or len(ratings_df) == 0:
        return {"status": "error", "message": "No ratings provided"}

    ratings_payload = []
    for _, row in ratings_df.iterrows():
        try:
            mid = int(row["movieId"])
            rat = float(row["rating"])
        except Exception:
            continue
        ratings_payload.append({"movieId": mid, "rating": rat})

    if not ratings_payload:
        return {"status": "error", "message": "Invalid ratings rows"}

    payload = {"ratings": ratings_payload}
    return _post(f"/user/{int(user_id)}/rate", json=payload)

def api_recommendations(user_id, n):
    return _get(f"/user/{int(user_id)}/recommendations", params={"n": int(n)})


# ---------------------------
# Recommendation orchestration (AUTO / MANUAL)
# ---------------------------
def ui_get_recommendations(user_id, n, mode, genre_for_cold_start):
    user_id = int(user_id)
    n = int(n)
    mode = mode.strip()

    # read user ratings count
    ur = api_user_ratings(user_id)
    count = ur.get("count", 0)

    if mode == "AUTO":
        rec = api_recommendations(user_id, n)
        df = _to_df(rec.get("recommendations", []))
        return rec.get("recommendation_type"), rec.get("num_ratings"), df

    if mode == "cold_start":
        df = api_popular(n, genre_for_cold_start)
        return "cold_start_forced", count, df

    if mode == "genre_based":
        if count == 0:
            return "error", count, pd.DataFrame([{"error": "Add 1-4 ratings first to use genre_based"}])
        if count >= 5:
            return "error", count, pd.DataFrame([{"error": "You already have >=5 ratings; genre_based is for 1-4 ratings"}])
        rec = api_recommendations(user_id, n)  # will naturally choose genre_based(_local)
        df = _to_df(rec.get("recommendations", []))
        return rec.get("recommendation_type"), rec.get("num_ratings"), df

    if mode == "personalized":
        if count < 5:
            return "error", count, pd.DataFrame([{"error": "Add at least 5 ratings to use personalized mode"}])
        rec = api_recommendations(user_id, n)
        df = _to_df(rec.get("recommendations", []))
        return rec.get("recommendation_type"), rec.get("num_ratings"), df

    return "error", count, pd.DataFrame([{"error": f"Unknown mode: {mode}"}])


# ---------------------------
# UI
# ---------------------------
with gr.Blocks(title="Movie Recommender Demo (FastAPI + Gradio)") as demo:
    gr.Markdown("# 🎬 Movie Recommendation Demo (API FastAPI + Gradio)\nCompatible Workbench (sans WebSocket Streamlit).")

    with gr.Tab("✅ Health & Meta"):
        with gr.Row():
            btn_health = gr.Button("Check API health", scale=1)
            btn_meta = gr.Button("Show /meta", scale=1)

        out_health = gr.JSON(label="/health")
        out_meta = gr.JSON(label="/meta")

        btn_health.click(fn=api_health, outputs=out_health)
        btn_meta.click(fn=api_meta, outputs=out_meta)

    with gr.Tab("🔎 Explorer"):
        gr.Markdown("### Search movies by title")
        with gr.Row():
            q_title = gr.Textbox(label="Title", placeholder="e.g., titanic")
            q_limit = gr.Slider(1, 50, value=10, step=1, label="Max results")
            btn_search = gr.Button("Search")
        out_search = gr.Dataframe(label="Search results", interactive=False)
        btn_search.click(fn=api_search, inputs=[q_title, q_limit], outputs=out_search)

        gr.Markdown("### Movie details")
        with gr.Row():
            movie_id = gr.Number(label="movieId", value=1, precision=0)
            btn_details = gr.Button("Get details")
        out_details = gr.JSON(label="Movie")
        btn_details.click(fn=api_movie_details, inputs=[movie_id], outputs=out_details)

        gr.Markdown("### Similar movies (genre overlap)")
        with gr.Row():
            sim_id = gr.Number(label="movieId", value=1, precision=0)
            sim_n = gr.Slider(1, 50, value=10, step=1, label="Top-N")
            btn_similar = gr.Button("Find similar")
        out_similar = gr.JSON(label="Similar result")
        btn_similar.click(fn=api_similar, inputs=[sim_id, sim_n], outputs=out_similar)

    with gr.Tab("⭐ Cold Start / Popular"):
        with gr.Row():
            pop_n = gr.Slider(1, 50, value=10, step=1, label="Top-N")
            pop_genre = gr.Textbox(label="Genre filter (optional)", placeholder="Comedy, Drama, Action...")
            btn_pop = gr.Button("Get popular")
        out_pop = gr.Dataframe(label="Popular movies", interactive=False)
        btn_pop.click(fn=api_popular, inputs=[pop_n, pop_genre], outputs=out_pop)

    with gr.Tab("✍️ Rate movies"):
        with gr.Row():
            rate_user = gr.Number(label="user_id", value=9, precision=0)
            btn_show_ratings = gr.Button("Show current ratings")
            btn_reset = gr.Button("Reset user ratings")

        out_user_ratings = gr.JSON(label="User ratings cache")

        btn_show_ratings.click(fn=api_user_ratings, inputs=[rate_user], outputs=out_user_ratings)
        btn_reset.click(fn=api_reset_user, inputs=[rate_user], outputs=out_user_ratings)

        gr.Markdown("### Add ratings (movieId + rating)")
        # editable table
        ratings_table = gr.Dataframe(
            headers=["movieId", "rating"],
            value=pd.DataFrame([{"movieId": 1, "rating": 4.0}]),
            interactive=True,
            label="Ratings to submit",
        )
        btn_submit = gr.Button("Submit ratings")
        out_submit = gr.JSON(label="Submit response")

        btn_submit.click(fn=api_submit_ratings, inputs=[rate_user, ratings_table], outputs=out_submit)

    with gr.Tab("🎯 Recommendations"):
        with gr.Row():
            rec_user = gr.Number(label="user_id", value=9, precision=0)
            rec_n = gr.Slider(1, 50, value=10, step=1, label="Top-N")
            rec_mode = gr.Dropdown(
                choices=["AUTO", "cold_start", "genre_based", "personalized"],
                value="AUTO",
                label="Execution mode",
            )
            rec_genre = gr.Textbox(label="Genre (only used for forced cold_start)", placeholder="e.g. Comedy")

        btn_rec = gr.Button("Get recommendations")

        out_rec_type = gr.Textbox(label="recommendation_type")
        out_num_ratings = gr.Number(label="num_ratings", precision=0)
        out_rec_df = gr.Dataframe(label="Recommendations", interactive=False)

        btn_rec.click(
            fn=ui_get_recommendations,
            inputs=[rec_user, rec_n, rec_mode, rec_genre],
            outputs=[out_rec_type, out_num_ratings, out_rec_df],
        )

demo.launch(server_name="0.0.0.0", server_port=7860)
