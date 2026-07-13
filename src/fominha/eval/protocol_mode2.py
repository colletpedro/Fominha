"""Protocolo C: avaliacao quantitativa do Modo 2, comparavel ao Modo 1
(SPEC-MODE2.md secao 8.1, fluxo 7.3).

REUSA build_masked_queries de eval/protocol.py (Modo 1, D-24) com os MESMOS
parametros/seed, garantindo as MESMAS 1000 queries mascaradas nos dois modos --
qualquer divergencia de amostragem invalidaria a comparacao.
"""

import json

import numpy as np

from fominha.eval.protocol import build_masked_queries
from fominha.mode1.recommend import EmptyQueryError, recommend
from fominha.mode2.index import META_PATH as SEMANTIC_META_PATH
from fominha.mode2.index import load_semantic_index
from fominha.mode2.search import search

MODE1_REPORT_PATH = "reports/eval_mode1.json"
NO_RESULTS_NOTE = "(sem resultados: query sem ingredientes reconheciveis)"


def evaluate_mode2(ks=(1, 5, 10), n_eval=1000, mask_frac=0.3, seed=42) -> dict:
    """Roda o Protocolo C (secao 8.1) e retorna o dict do contrato 6.4."""
    _, _, df = load_semantic_index()

    rng = np.random.default_rng(seed)
    queries = build_masked_queries(df, n_eval, mask_frac, rng)
    n_actual = len(queries)

    k_top = max(ks)
    hit_counts = {k: 0 for k in ks}

    for origin_id, kept in queries:
        query_string = ", ".join(kept)  # D-25: sem template adicional
        recs = search(query_string, k=k_top)
        ranked_ids = [r.recipe_id for r in recs]
        for k in ks:
            if origin_id in ranked_ids[:k]:
                hit_counts[k] += 1

    metrics = {
        f"mode2_hit_rate@{k}": round(hit_counts[k] / n_actual, 6) for k in ks
    }

    with open(MODE1_REPORT_PATH) as f:
        mode1_report = json.load(f)
    mode1_reference = {
        f"hit_rate@{k}": mode1_report["metrics"][f"hit_rate@{k}"] for k in ks
    }

    with open(SEMANTIC_META_PATH) as f:
        semantic_meta = json.load(f)

    return {
        "protocol": "masked-retrieval-semantic (C)",
        "n_recipes_index": len(df),
        "n_eval_queries": n_actual,
        "mask_frac": mask_frac,
        "seed": seed,
        "query_format": "tokens joined by ', '",
        "metrics": metrics,
        "mode1_reference": mode1_reference,
        "index_built_at": semantic_meta["built_at"],
    }


def run_comparison(queries_path: str, k: int = 5) -> str:
    """Protocolo D (secao 8.2, fluxo 7.4): roda as queries curadas nos dois
    modos e monta o markdown lado a lado (contrato 6.6). Sem interpretacao --
    apenas os dados; a leitura humana vai no README.
    """
    with open(queries_path) as f:
        data = json.load(f)

    lines = ["# Comparativo Modo 1 vs Modo 2 (Protocolo D)", ""]

    for q in data["queries"]:
        qid, text, rationale = q["id"], q["text"], q["rationale"]

        try:
            # Fluxo 7.4: a query D e linguagem natural; para o Modo 1 ela passa
            # por normalize_ingredients normalmente -- tokens nao-ingrediente
            # viram OOV, e expor isso e o ponto do contraste. Tokenizar por
            # ESPACO (nao ", "): a query nao tem virgulas, entao split(", ")
            # tratava a frase inteira como um pseudo-ingrediente monolitico em
            # vez de degradar palavra a palavra.
            mode1_results = recommend(text.split(), k=k)
        except EmptyQueryError:
            mode1_results = None

        mode2_results = search(text, k=k)

        lines.append(f"## {qid}: \"{text}\"")
        lines.append(f"_rationale: {rationale}_")
        lines.append("")

        if mode1_results is None:
            lines.append(f"Modo 1: {NO_RESULTS_NOTE}")
            lines.append("")

        lines.append("| # | Modo 1 (title, score) | Modo 2 (title, score) |")
        lines.append("|---|---|---|")
        for i in range(k):
            if mode1_results and i < len(mode1_results):
                m1 = f"{mode1_results[i].title} ({mode1_results[i].score:.4f})"
            else:
                m1 = ""
            m2 = f"{mode2_results[i].title} ({mode2_results[i].score:.4f})" if i < len(mode2_results) else ""
            lines.append(f"| {i + 1} | {m1} | {m2} |")
        lines.append("")

    return "\n".join(lines)
