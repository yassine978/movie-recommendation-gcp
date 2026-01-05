# Movie Recommendation System (Vertex AI Workbench)

**Group Members**: Mohamed Yassine Madhi & Mehdi Zneidi  
**Project**: Personalized Movie Recommendation System  
**Environment**: Vertex AI Workbench (JupyterLab)

## 🎯 Overview

A production-ready movie recommendation system demonstrating how recommendations evolve as users interact with the application.

The project provides:
- A **FastAPI** backend exposing recommendation endpoints
- A **Gradio** UI (Workbench-friendly) to interactively test the system
- Multiple recommendation strategies: **cold start**, **genre-based**, and **personalized (SVD)**

> Note: Due to access constraints on Cloud Run / Artifact Registry, the final demo runs **locally in Vertex AI Workbench** (as requested by the instructor).

---

## ✨ Key Features

- **Multiple recommendation modes**
  - **Cold Start**: popular movies (BigQuery if available, otherwise local fallback)
  - **Genre-based**: for users with a few ratings (1–4)
  - **Personalized (SVD)**: collaborative filtering for users with ≥5 ratings
- **Interactive Gradio UI**
  - Search movies, view details, browse popular, rate movies, request recommendations
- **Robust data strategy**
  - BigQuery optional; automatic fallback to local data (movies from model artifact)
- **Demo-ready**
  - Reset user ratings, inspect user cache, view metadata and system status

---

## 🧱 Architecture

┌──────────────────────────────────────────────────────────────────┐
│ VERTEX AI WORKBENCH (RUNTIME)                                    │
│ JupyterLab Environment                                           │
│                                                                  │
│ ┌──────────────────────────┐ ┌─────────────────────────┐         │
│ │ Gradio UI                │       │ FastAPI API  │                │
│ │ frontend/gradio_app.py │<----->│ src/api/main.py     │         │
│ │ /proxy/7860 │ HTTP         │ /docs, /health, etc.    │         │
│ └──────────────────────────┘ └───────────┬─────────────┘         │
│ │                                                                │
│ ▼                                                                │
│ ┌───────────────────────────┐                                    │
│ │ Recommendation Engine     │                                    │
│ │ - MovieRecommender (SVD) │                                     │
│ │ - ColdStartHandler       │                                     │
│ └───────────┬──────────────┘                                     │
│              │                                                   │
│ ┌─────────────────────┴────────────────┐                         │
│ │ Data Access Strategy ││
│ │ 1) BigQuery (optional) ││
│ │ 2) Local fallback (always available) ││
│ └───────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────────┘