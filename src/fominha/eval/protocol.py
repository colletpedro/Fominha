"""Protocolo de avaliacao do Modo 1 (SPEC.md secao 8).

Implementa LITERALMENTE os dois protocolos da secao 8, ambos com seed fixa:
  A) Masked-recipe retrieval  -> hit_rate@k   (secao 8.1)
  B) Overlap relevance        -> precision@k  (secao 8.2)

As MESMAS queries mascaradas do Protocolo A alimentam o Protocolo B (secao 8.2).
Todo RNG deriva de np.random.default_rng(seed), de forma que a amostragem e o
mascaramento sejam deterministicos -> dois runs produzem JSON identico (RNF-01).
"""

import json

import numpy as np

from fominha.mode1.index import META_PATH, IndexNotBuiltError, load_index
from fominha.mode1.recommend import recommend


def _read_index_built_at() -> str:
    """built_at deterministico: o timestamp de build do indice avaliado.

    Nao usamos wall-clock aqui porque o criterio de aceite 6 / RNF-01 exige que
    dois runs da avaliacao produzam eval_mode1.json byte-identico. O build do
    indice e um evento fixo em disco, entao re-runs da avaliacao leem o mesmo
    valor -> JSON reproduzivel.
    """
    try:
        with open(META_PATH) as f:
            meta = json.load(f)
    except FileNotFoundError as exc:
        raise IndexNotBuiltError(
            f"index_meta.json ausente ({META_PATH}). Rode scripts/02_build_index.py."
        ) from exc
    return meta["built_at"]


def _build_masked_queries(df, n_eval, mask_frac, rng):
    """Amostra n_eval receitas com >= 3 tokens e mascara mask_frac de cada uma.

    Retorna lista de (origin_recipe_id, kept_tokens). A receita NAO e removida do
    indice (secao 8.1) -- ela pode e deve reaparecer no proprio top-k.
    """
    canonical = df["ingredients_canonical"]
    lengths = canonical.apply(len)
    eligible_ids = np.sort(df.loc[lengths >= 3, "recipe_id"].to_numpy())

    n_actual = min(n_eval, len(eligible_ids))
    chosen_ids = rng.choice(eligible_ids, size=n_actual, replace=False)

    by_id = df.set_index("recipe_id")["ingredients_canonical"]

    queries = []
    for rid in chosen_ids:
        tokens = list(by_id.loc[int(rid)])
        t = len(tokens)
        n_mask = int(round(mask_frac * t))
        n_mask = max(1, min(n_mask, t - 2))  # >= 1 mascarado, >= 2 mantidos
        mask_positions = set(int(p) for p in rng.choice(t, size=n_mask, replace=False))
        kept = [tokens[i] for i in range(t) if i not in mask_positions]
        queries.append((int(rid), kept))

    return queries


def _jaccard(a: set, b: set) -> float:
    union = len(a | b)
    if union == 0:
        return 0.0
    return len(a & b) / union


def evaluate(
    ks_hit=(1, 5, 10),
    ks_prec=(5, 10),
    n_eval=1000,
    mask_frac=0.3,
    jaccard_threshold=0.5,
    seed=42,
) -> dict:
    """Roda os Protocolos A e B da secao 8 e retorna o dict do contrato 6.5."""
    vectorizer, matrix, df = load_index()
    by_id = df.set_index("recipe_id")["ingredients_canonical"]

    rng = np.random.default_rng(seed)
    queries = _build_masked_queries(df, n_eval, mask_frac, rng)
    n_actual = len(queries)

    k_top = max(max(ks_hit), max(ks_prec))

    hit_counts = {k: 0 for k in ks_hit}
    prec_sums = {k: 0.0 for k in ks_prec}
    rel_sums = {k: 0 for k in ks_prec}

    for origin_id, kept in queries:
        recs = recommend(kept, k=k_top)
        ranked_ids = [r.recipe_id for r in recs]

        # Protocolo A: a receita de origem aparece no top-k?
        for k in ks_hit:
            if origin_id in ranked_ids[:k]:
                hit_counts[k] += 1

        # Protocolo B: relevancia por overlap de Jaccard sobre tokens canonicos.
        q_set = set(kept)
        for k in ks_prec:
            n_rel = 0
            for r in recs[:k]:
                if r.recipe_id == origin_id:
                    n_rel += 1  # receita de origem: relevante por definicao
                    continue
                r_set = set(by_id.loc[r.recipe_id])
                if _jaccard(q_set, r_set) >= jaccard_threshold:
                    n_rel += 1
            prec_sums[k] += n_rel / k  # denominador = k (secao 8.2)
            rel_sums[k] += n_rel

    metrics = {}
    for k in ks_hit:
        metrics[f"hit_rate@{k}"] = round(hit_counts[k] / n_actual, 6)
    for k in ks_prec:
        metrics[f"precision@{k}"] = round(prec_sums[k] / n_actual, 6)

    # Diagnostico exigido pelas Decisoes a confirmar (item 2): nº medio de
    # relevantes por query, para detectar threshold degenerado.
    avg_relevant_per_query = {
        f"@{k}": round(rel_sums[k] / n_actual, 4) for k in ks_prec
    }

    return {
        "protocol": "masked-recipe-retrieval + overlap-relevance",
        "n_recipes_index": int(matrix.shape[0]),
        "n_eval_queries": n_actual,
        "mask_frac": mask_frac,
        "seed": seed,
        "jaccard_threshold": jaccard_threshold,
        "metrics": metrics,
        "avg_relevant_per_query": avg_relevant_per_query,
        "built_at": _read_index_built_at(),
    }
