"""Build/load do indice TF-IDF do Modo 1 (SPEC.md secao 5.2, contrato 6.2)."""

import json
import os
from datetime import datetime, timezone

import joblib
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

ARTIFACTS_DIR = "artifacts"
VECTORIZER_PATH = os.path.join(ARTIFACTS_DIR, "tfidf_vectorizer.joblib")
MATRIX_PATH = os.path.join(ARTIFACTS_DIR, "tfidf_matrix.npz")
META_PATH = os.path.join(ARTIFACTS_DIR, "index_meta.json")
DEFAULT_PARQUET_PATH = "data/processed/recipes.parquet"


class IndexNotBuiltError(Exception):
    """Artefatos do indice TF-IDF ausentes (edge case E-07)."""


def build_index(parquet_path: str) -> None:
    """Constroi e persiste o indice TF-IDF sobre ingredients_canonical."""
    df = pd.read_parquet(parquet_path)
    documents = df["ingredients_canonical"].apply(lambda tokens: " ".join(tokens))

    vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=5)
    matrix = vectorizer.fit_transform(documents)

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    sparse.save_npz(MATRIX_PATH, matrix)

    meta = {
        "n_recipes": len(df),
        "vocab_size": len(vectorizer.vocabulary_),
        "built_at": datetime.now(timezone.utc).isoformat(),
        "seed": 42,
        "n_recipes_param": len(df),
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)


def load_index():
    """Carrega vectorizer, matriz esparsa e dataset tratado.

    Levanta IndexNotBuiltError se algum artefato estiver ausente.
    """
    missing = [
        path for path in (VECTORIZER_PATH, MATRIX_PATH, META_PATH)
        if not os.path.exists(path)
    ]
    if missing:
        raise IndexNotBuiltError(
            f"Indice TF-IDF nao encontrado ({', '.join(missing)}). "
            "Rode scripts/02_build_index.py antes de chamar recommend()."
        )

    vectorizer = joblib.load(VECTORIZER_PATH)
    matrix = sparse.load_npz(MATRIX_PATH)
    df = pd.read_parquet(DEFAULT_PARQUET_PATH)

    return vectorizer, matrix, df
