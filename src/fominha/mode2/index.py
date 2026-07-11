"""Build/load do indice semantico FAISS (SPEC-MODE2.md secoes 5.2 + 6.1)."""

import json
import logging
import os
import time
from datetime import datetime, timezone

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from fominha.mode2 import SemanticIndexNotBuiltError
from fominha.mode2.corpus import build_embedding_text

logger = logging.getLogger(__name__)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DIMS = 384
BATCH_SIZE = 256
CORPUS_FORMAT = "title. Ingredients: <canonical>. directions[:400]"

ARTIFACTS_DIR = "artifacts"
FAISS_INDEX_PATH = os.path.join(ARTIFACTS_DIR, "embeddings.faiss")
META_PATH = os.path.join(ARTIFACTS_DIR, "embeddings_meta.json")
DEFAULT_PARQUET_PATH = "data/processed/recipes.parquet"


def build_semantic_index(parquet_path: str) -> None:
    """Constroi e persiste o indice semantico FAISS sobre o corpus de embedding."""
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(
            f"Dataset tratado nao encontrado em: {parquet_path}. "
            "Rode scripts/01_ingest.py antes de gerar embeddings."
        )

    start = time.perf_counter()

    df = pd.read_parquet(parquet_path).sort_values("recipe_id").reset_index(drop=True)

    texts = [
        build_embedding_text(row["title"], list(row["ingredients_canonical"]), row["directions"])
        for _, row in df.iterrows()
    ]

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype("float32")

    index = faiss.IndexFlatIP(DIMS)
    index.add(embeddings)

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    faiss.write_index(index, FAISS_INDEX_PATH)

    build_seconds = time.perf_counter() - start

    meta = {
        "model": MODEL_NAME,
        "dims": DIMS,
        "n_recipes": len(df),
        "aligned_to": "recipe_id",
        "corpus_format": CORPUS_FORMAT,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "build_seconds": build_seconds,
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(
        "Indice semantico construido: n_recipes=%d, dims=%d, build_seconds=%.2f",
        len(df), DIMS, build_seconds,
    )


def load_semantic_index():
    """Carrega modelo, indice FAISS e parquet tratado.

    Levanta SemanticIndexNotBuiltError se algum artefato estiver ausente, e
    erro instruindo rebuild se n_recipes do meta divergir do parquet (E-27).
    """
    missing = [
        path for path in (FAISS_INDEX_PATH, META_PATH)
        if not os.path.exists(path)
    ]
    if missing:
        raise SemanticIndexNotBuiltError(
            f"Indice semantico nao encontrado ({', '.join(missing)}). "
            "Rode scripts/05_build_embeddings.py antes de chamar search()."
        )

    with open(META_PATH) as f:
        meta = json.load(f)

    df = pd.read_parquet(DEFAULT_PARQUET_PATH).sort_values("recipe_id").reset_index(drop=True)

    if meta["n_recipes"] != len(df):
        raise RuntimeError(
            f"Indice semantico dessincronizado do parquet: meta.n_recipes={meta['n_recipes']} "
            f"!= len(parquet)={len(df)}. Rode scripts/05_build_embeddings.py novamente."
        )

    model = SentenceTransformer(meta["model"])
    index = faiss.read_index(FAISS_INDEX_PATH)

    return model, index, df
