"""FastAPI fina sobre recommend() e search() (SPEC-FRONTEND.md secao 3).

Nenhuma logica nova de retrieval: este modulo so valida, tokeniza e serializa.
"""

import os
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator

from fominha.mode1 import recommend as recommend_module
from fominha.mode1.index import IndexNotBuiltError
from fominha.mode1.recommend import EmptyQueryError, recommend
from fominha.mode2 import SemanticIndexNotBuiltError
from fominha.mode2 import search as search_module
from fominha.mode2.search import search

FRONTEND_INDEX = Path(__file__).resolve().parents[3] / "frontend" / "index.html"

MODE1_UNAVAILABLE_MSG = (
    "Indice do Modo 1 nao construido. Rode scripts/02_build_index.py."
)
MODE2_UNAVAILABLE_MSG = (
    "Indice do Modo 2 nao construido. Rode scripts/05_build_embeddings.py."
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega os indices dos dois modos uma unica vez (D-43).

    Se os artefatos estiverem ausentes, o app sobe mesmo assim (E-33) -- os
    endpoints correspondentes retornam 503 em vez de quebrar o startup.
    """
    try:
        recommend_module._get_index()
        app.state.mode1_available = True
    except IndexNotBuiltError:
        app.state.mode1_available = False

    try:
        search_module._get_index()
        app.state.mode2_available = True
    except SemanticIndexNotBuiltError:
        app.state.mode2_available = False

    yield


app = FastAPI(lifespan=lifespan)


class QueryRequest(BaseModel):
    query: str
    k: int = Field(default=5, ge=1, le=20)

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("query must not be empty or whitespace-only")
        return stripped


def _tokenize_for_mode1(query: str) -> list[str]:
    """Tokenizacao hibrida (D-42): virgula preserva ingredientes compostos;
    espaco reproduz a degradacao palavra-a-palavra do Protocolo D."""
    if "," in query:
        return [s.strip() for s in query.split(",") if s.strip()]
    return query.split()


@app.post("/api/recommend")
async def api_recommend(request: QueryRequest):
    if not app.state.mode1_available:
        raise HTTPException(status_code=503, detail=MODE1_UNAVAILABLE_MSG)

    tokens = _tokenize_for_mode1(request.query)
    try:
        results = recommend(tokens, k=request.k)
    except EmptyQueryError:
        return []  # E-31: resultado, nao erro
    except IndexNotBuiltError:
        raise HTTPException(status_code=503, detail=MODE1_UNAVAILABLE_MSG)

    return [asdict(r) for r in results]


@app.post("/api/search")
async def api_search(request: QueryRequest):
    if not app.state.mode2_available:
        raise HTTPException(status_code=503, detail=MODE2_UNAVAILABLE_MSG)

    try:
        results = search(request.query, k=request.k)
    except SemanticIndexNotBuiltError:
        raise HTTPException(status_code=503, detail=MODE2_UNAVAILABLE_MSG)

    return [asdict(r) for r in results]


@app.get("/")
async def index():
    if FRONTEND_INDEX.exists():
        return FileResponse(FRONTEND_INDEX)
    return JSONResponse({"status": "ok", "frontend": "pending"})
