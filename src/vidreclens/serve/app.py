import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from vidreclens.serve.engine import Recommender


class RecommendRequest(BaseModel):
    history: list[int] = Field(default_factory=list, max_length=500, description="video ids the user watched, oldest first")
    k: int = Field(10, ge=1, le=100)
    alpha: float | None = Field(None, ge=0.0, le=2.0, description="popularity penalty, omit to use the default")


def create_app(bundle_dir=None):
    path = Path(bundle_dir or os.environ.get("VIDRECLENS_BUNDLE", "bundle"))

    @asynccontextmanager
    async def lifespan(app):
        app.state.engine = Recommender(path)
        yield

    app = FastAPI(title="VidRec-Lens", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health():
        e = app.state.engine
        return {"status": "ok", "items": len(e.item_ids), "default_alpha": e.alpha}

    @app.post("/recommend")
    def recommend(req: RecommendRequest):
        return app.state.engine.recommend(req.history, req.k, req.alpha)

    @app.get("/items/{item_id}")
    def item(item_id: int):
        r = app.state.engine.item(item_id)
        if r is None:
            raise HTTPException(status_code=404, detail="unknown item")
        return r

    return app
