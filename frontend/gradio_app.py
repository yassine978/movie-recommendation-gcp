import requests
import pandas as pd
import gradio as gr
import matplotlib.pyplot as plt

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


def _strip_score_fields_df(df: pd.DataFrame) -> pd.DataFrame:
    """Remove score/extra fields from tables everywhere."""
    if df is None or df.empty:
        return df
    drop_cols = [c for c in ["predicted_rating", "popularity_score", "num_ratings", "similarity_score", "similarity"] if c in df.columns]
    return df.drop(columns=drop_cols, errors="ignore")


def _strip_score_fields_obj(obj):
    """Remove score/extra fields from dict/list responses (e.g., similar result JSON)."""
    drop_keys = {"predicted_rating", "popularity_score", "num_ratings", "similarity_score", "similarity"}

    if isinstance(obj, list):
        cleaned = []
        for item in obj:
            if isinstance(item, dict):
                cleaned.append({k: _strip_score_fields_obj(v) for k, v in item.items() if k not in drop_keys})
            else:
                cleaned.append(item)
        return cleaned

    if isinstance(obj, dict):
        # If it's a "similar_movies" payload, clean inside
        out = {}
        for k, v in obj.items():
            if k in drop_keys:
                continue
            out[k] = _strip_score_fields_obj(v)
        return out

    return obj


# ---------------------------
# API wrappers
# ---------------------------
def api_health():
    return _get("/health")


def api_meta():
    return _get("/meta")


def api_search(title, limit):
    df = _to_df(_get("/movies/search", params={"query": title, "limit": int(limit)}))
    return _strip_score_fields_df(df)


def api_movie_details(movie_id):
    obj = _get(f"/movies/{int(movie_id)}")
    return _strip_score_fields_obj(obj)


def api_similar(movie_id, n):
    obj = _get(f"/movies/{int(movie_id)}/similar", params={"n": int(n)})
    return _strip_score_fields_obj(obj)


def api_popular(n, genre):
    params = {"n": int(n)}
    if genre and str(genre).strip():
        params["genre"] = str(genre).strip()
    df = _to_df(_get("/movies/popular", params=params))
    return _strip_score_fields_df(df)


def api_user_ratings(user_id):
    return _get(f"/user/{int(user_id)}/ratings")


def api_reset_user(user_id):
    return _post(f"/user/{int(user_id)}/reset")


def api_submit_ratings(user_id, ratings_df: pd.DataFrame):
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
    obj = _get(f"/user/{int(user_id)}/recommendations", params={"n": int(n)})
    # Strip score fields inside recommendations list too
    if isinstance(obj, dict) and "recommendations" in obj:
        obj["recommendations"] = _strip_score_fields_obj(obj["recommendations"])
    return obj


# ---------------------------
# Recommendation orchestration (AUTO / MANUAL)
#  - Recommendations tab: remove genre input, so forced cold_start uses no genre
# ---------------------------
def ui_get_recommendations(user_id, n, mode):
    user_id = int(user_id)
    n = int(n)
    mode = (mode or "").strip()

    ur = api_user_ratings(user_id)
    count = int(ur.get("count", 0))

    if mode == "AUTO":
        rec = api_recommendations(user_id, n)
        df = _to_df(rec.get("recommendations", []))
        df = _strip_score_fields_df(df)
        return rec.get("recommendation_type"), rec.get("num_ratings"), df

    if mode == "cold_start":
        # forced cold start uses API popular without genre (per request)
        df = _to_df(_get("/movies/popular", params={"n": n}))
        df = _strip_score_fields_df(df)
        return "cold_start_forced", count, df

    if mode == "genre_based":
        if count == 0:
            return "error", count, pd.DataFrame([{"error": "Add 1-4 ratings first to use genre_based"}])
        if count >= 5:
            return "error", count, pd.DataFrame([{"error": "You already have >=5 ratings; genre_based is for 1-4 ratings"}])

        rec = api_recommendations(user_id, n)
        df = _to_df(rec.get("recommendations", []))
        df = _strip_score_fields_df(df)
        return rec.get("recommendation_type"), rec.get("num_ratings"), df

    if mode == "personalized":
        if count < 5:
            return "error", count, pd.DataFrame([{"error": "Add at least 5 ratings to use personalized mode"}])

        rec = api_recommendations(user_id, n)
        df = _to_df(rec.get("recommendations", []))
        df = _strip_score_fields_df(df)
        return rec.get("recommendation_type"), rec.get("num_ratings"), df

    return "error", count, pd.DataFrame([{"error": f"Unknown mode: {mode}"}])


# ---------------------------
# Analytics (RMSE + User rating distribution only)
#  - Remove sparsity
# ---------------------------
def ui_meta_summary():
    meta = api_meta()
    rmse = meta.get("rmse", None)
    summary = f"RMSE: {rmse}"
    return summary, meta


def plot_user_ratings_hist(user_id: int):
    ur = api_user_ratings(int(user_id))
    ratings = [r.get("rating") for r in ur.get("ratings", []) if "rating" in r]

    fig = plt.figure()
    if not ratings:
        plt.title("User rating distribution (empty)")
        plt.xlabel("rating")
        plt.ylabel("count")
        return fig

    plt.hist(ratings, bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5], edgecolor="black")
    plt.title(f"User {int(user_id)} rating distribution")
    plt.xlabel("rating")
    plt.ylabel("count")
    return fig


# ---------------------------
# UI
# ---------------------------
with gr.Blocks(title="Movie Recommender Demo (FastAPI + Gradio)") as demo:
    gr.Markdown(
        "# 🎬 Movie Recommendation Demo (FastAPI + Gradio)\n"
        "Compatible Vertex AI Workbench proxy."
    )



    # ---- Explorer
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

    # ---- Popular / Cold start
    with gr.Tab("⭐ Cold Start / Popular"):
        with gr.Row():
            pop_n = gr.Slider(1, 50, value=10, step=1, label="Top-N")
            pop_genre = gr.Textbox(label="Genre filter (optional)", placeholder="Comedy, Drama, Action...")
            btn_pop = gr.Button("Get popular")
        out_pop = gr.Dataframe(label="Popular movies", interactive=False)
        btn_pop.click(fn=api_popular, inputs=[pop_n, pop_genre], outputs=out_pop)

    # ---- Rate Movies
    with gr.Tab("✍️ Rate movies"):
        with gr.Row():
            rate_user = gr.Number(label="user_id", value=9, precision=0)
            btn_show_ratings = gr.Button("Show current ratings")
            btn_reset = gr.Button("Reset user ratings")

        out_user_ratings = gr.JSON(label="User ratings cache")

        btn_show_ratings.click(fn=api_user_ratings, inputs=[rate_user], outputs=out_user_ratings)
        btn_reset.click(fn=api_reset_user, inputs=[rate_user], outputs=out_user_ratings)

        gr.Markdown("### Add ratings (movieId + rating)")
        ratings_table = gr.Dataframe(
            headers=["movieId", "rating"],
            value=pd.DataFrame([{"movieId": 1, "rating": 4.0}]),
            interactive=True,
            label="Ratings to submit",
        )
        btn_submit = gr.Button("Submit ratings")
        out_submit = gr.JSON(label="Submit response")
        btn_submit.click(fn=api_submit_ratings, inputs=[rate_user, ratings_table], outputs=out_submit)

    # ---- Recommendations
    with gr.Tab("🎯 Recommendations"):
        with gr.Row():
            rec_user = gr.Number(label="user_id", value=9, precision=0)
            rec_n = gr.Slider(1, 50, value=10, step=1, label="Top-N")
            rec_mode = gr.Dropdown(
                choices=["AUTO", "cold_start", "genre_based", "personalized"],
                value="AUTO",
                label="Execution mode",
            )

        btn_rec = gr.Button("Get recommendations")

        out_rec_type = gr.Textbox(label="recommendation_type")
        out_num_ratings = gr.Number(label="num_ratings", precision=0)
        out_rec_df = gr.Dataframe(label="Recommendations", interactive=False)

        btn_rec.click(
            fn=ui_get_recommendations,
            inputs=[rec_user, rec_n, rec_mode],
            outputs=[out_rec_type, out_num_ratings, out_rec_df],
        )

    # ---- Analytics (RMSE + User distribution only)
    with gr.Tab("📊 Analytics"):
        gr.Markdown("### Model metrics & user rating diagnostics")

        with gr.Row():
            a_btn_refresh = gr.Button("Refresh metrics")
            a_user = gr.Number(label="user_id (for rating histogram)", value=9, precision=0)
            a_btn_user_hist = gr.Button("Plot user ratings")

        a_summary = gr.Textbox(label="Summary", lines=2)
        a_meta = gr.JSON(label="/meta")
        a_plot_user = gr.Plot(label="User rating distribution")

        a_btn_refresh.click(fn=ui_meta_summary, outputs=[a_summary, a_meta])
        a_btn_user_hist.click(fn=plot_user_ratings_hist, inputs=[a_user], outputs=a_plot_user)

demo.launch(server_name="0.0.0.0", server_port=7860)
