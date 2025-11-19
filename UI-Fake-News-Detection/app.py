# app.py
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from typing import Any, Dict, Optional
import os
from pydantic import BaseModel
# Import model utilities
from model import predict_tfidf, BASE_DIR  # BASE_DIR reused for index.html path


app = FastAPI()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# serve static files (CSS, JS, images) from the project directory
app.mount("/static", StaticFiles(directory=BASE_DIR), name="static")


# Serve a static index.html at root (for your frontend)
@app.get("/", response_class=FileResponse)
def read_root():
    return os.path.join(BASE_DIR, "index.html")


class PredictIn(BaseModel):
    title: Optional[str] = ""
    text: Optional[str] = ""
    use_sep: bool = True


@app.post("/predict")
def predict(data: Dict[str, Any] = Body(...)):
    """
    Accepts JSON like:
    {
      "title": "...",
      "text": "...",
      "use_sep": true
    }
    and returns the TF-IDF + LR prediction.
    """
    title = (data.get("title") or "").strip()
    text = (
        data.get("text")
        or data.get("content")
        or data.get("article")
        or ""
    ).strip()
    use_sep = data.get("use_sep", True)

    if not title and not text:
        raise HTTPException(status_code=400, detail="Provide some text to analyze")

    try:
        result = predict_tfidf(title=title, text=text, use_sep=use_sep)
        return result
    except ValueError as e:
        # from predict_tfidf if both empty
        raise HTTPException(status_code=400, detail=str(e))


# Enable CORS for your frontend (Lovable, local file, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # for demo; later restrict to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
