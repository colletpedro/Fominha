"""Ingestao e tratamento do RecipeNLG (SPEC.md secao 7.1, Fluxo 1)."""

import ast
import logging
import os

import pandas as pd

from fominha.normalize import normalize_ingredients

logger = logging.getLogger(__name__)


def _try_parse_list(value):
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return None
    if not isinstance(parsed, list):
        return None
    return parsed


def ingest(raw_csv_path: str, n_recipes: int, seed: int = 42) -> pd.DataFrame:
    """Le, amostra, normaliza e trata o RecipeNLG (Fluxo 7.1)."""
    if not os.path.exists(raw_csv_path):
        raise FileNotFoundError(
            f"Dataset RecipeNLG nao encontrado em: {raw_csv_path}. "
            "Baixe o dataset manualmente (ver README) e coloque-o nesse caminho."
        )

    df = pd.read_csv(raw_csv_path)
    n_read = len(df)

    if n_recipes >= n_read:
        logger.warning(
            "--n-recipes (%d) >= total do dataset (%d); usando o dataset inteiro.",
            n_recipes, n_read,
        )
        sample = df
    else:
        sample = df.sample(n=n_recipes, random_state=seed)

    n_parse_failed = 0
    n_invalid = 0
    rows = []

    for _, row in sample.iterrows():
        ingredients_raw = _try_parse_list(row.get("ingredients"))
        ner_tokens = _try_parse_list(row.get("NER"))

        if ingredients_raw is None or ner_tokens is None:
            n_parse_failed += 1
            continue

        title = str(row.get("title") or "").strip()
        ingredients_canonical = normalize_ingredients(ner_tokens)

        if len(ingredients_canonical) < 2 or not title:
            n_invalid += 1
            continue

        directions_raw = _try_parse_list(row.get("directions"))
        directions = "\n".join(directions_raw) if directions_raw else ""

        rows.append({
            "title": title,
            "ingredients_raw": ingredients_raw,
            "ingredients_canonical": ingredients_canonical,
            "directions": directions,
            "link": str(row.get("link") or ""),
        })

    result = pd.DataFrame(rows)
    result.insert(0, "recipe_id", range(len(result)))
    result = result.astype({
        "recipe_id": "int64",
        "title": "string",
        "directions": "string",
        "link": "string",
    })

    logger.info(
        "Ingestao concluida: lidas=%d, amostradas=%d, descartadas_parse=%d, "
        "descartadas_invalidas=%d, total_final=%d",
        n_read, len(sample), n_parse_failed, n_invalid, len(result),
    )

    return result
