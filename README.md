# Fominha

Content-based recipe recommender (Mode 1): give it the ingredients you have, get back ranked recipes. Part of a planned two-mode system (see [Roadmap](#7-roadmap)).

## 1. What it does — Mode 1

Input: a free-text list of ingredients (e.g. `"chicken, rice, garlic"`). Output: recipes ranked by cosine similarity between your ingredients and each recipe's ingredient list, both represented as TF-IDF vectors over canonical ingredient tokens.

There is no user history, no ratings, no personalization — this is a single stateless function: `recommend(ingredients, k)`. Each result includes the similarity score plus which of your ingredients matched the recipe and which of the recipe's ingredients you're missing.

## 2. Why these choices

Summarized from `SPEC.md` section 11 (ADR-lite):

- **Content-based, not collaborative filtering** — RecipeNLG has no user ratings; collaborative filtering needs interaction data the dataset doesn't provide.
- **TF-IDF + cosine similarity** — an interpretable, training-free baseline that also serves as an honest lexical contrast to a future semantic (embedding-based) mode.
- **RecipeNLG's `NER` column as the ingredient source, plus a custom normalization pipeline** — parsing raw ingredient strings is the biggest quality risk; the dataset's own pre-extracted entities plus deterministic normalization keep that risk bounded without outsourcing it to a heavier NLP stack.
- **Same normalization pipeline for dataset and user query** — index-time and query-time features must live in the same space or cosine similarity breaks silently.
- **Bigrams in the TF-IDF vocabulary (`ngram_range=(1,2)`)** — compound ingredients ("cream cheese", "olive oil") become a single feature instead of two diluted unigrams.
- **Suffix-rule singularization, no spaCy/NLTK** — a full lemmatizer is a heavy dependency for marginal gain over a food-ingredient vocabulary; see [Known limitations](#5-known-limitations) for the trade-off this makes.

## 3. Quickstart

Requires Python >= 3.11.

**1. Get the dataset.** RecipeNLG is not bundled in this repo (license-gated distribution). Download it manually and place it at `data/raw/full_dataset.csv`:
- Official site: https://recipenlg.cs.put.poznan.pl/ (accept the license, download the full dataset)
- Kaggle: dataset "RecipeNLG" via the Kaggle API (requires `kaggle` auth configured)

**2. Set up the environment.**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**3. Run the pipeline, in order.**
```bash
python scripts/01_ingest.py --raw data/raw/full_dataset.csv --n-recipes 100000 --seed 42
python scripts/02_build_index.py
python scripts/03_evaluate.py --k 1 5 10 --n-eval 1000 --mask-frac 0.3 --seed 42
python scripts/04_query.py --ingredients "chicken, rice, garlic" --k 10
```

`01_ingest.py` reads the raw CSV, samples `--n-recipes` rows with `--seed`, normalizes ingredients, and writes `data/processed/recipes.parquet`. `02_build_index.py` takes no arguments — it builds the TF-IDF index from that parquet and writes `artifacts/`. `03_evaluate.py` runs the two evaluation protocols (section 4) and writes `reports/eval_mode1.json`. `04_query.py` is the manual query CLI.

**Example query output** (`--ingredients "chicken, rice, garlic" --k 5`, run against the 100k-recipe index):

```
1. Chicken And Rice (recipe_id=67995, score=0.7044)
   matched: chicken, rice
   missing:
   link: https://www.cookbooks.com/Recipe-Details.aspx?id=681629

2. On the Go Munchies (recipe_id=3874, score=0.5427)
   matched: rice, garlic
   missing: peanut
   link: https://www.food.com/recipe/on-the-go-munchies-330534

3. Baked Chicken And Rice-A-Roni (recipe_id=33946, score=0.4918)
   matched:
   missing: chicken breast, chicken rice
   link: https://www.cookbooks.com/Recipe-Details.aspx?id=36887

4. Yonggyebaeksuk (Korean Spring Chicken Soup) (recipe_id=77209, score=0.4670)
   matched: rice, garlic
   missing: young roasting chicken, onion, chinese jujube, chestnut, ginger, salt, green onion
   link: https://www.food.com/recipe/yonggyebaeksuk-korean-spring-chicken-soup-488704

5. Chicken Oriental (recipe_id=11391, score=0.4621)
   matched: chicken, rice, garlic
   missing: vegetable, sweet sue chicken
   link: https://www.cookbooks.com/Recipe-Details.aspx?id=581696
```

Result 3 is a useful example of why `matched_ingredients` can be empty while the recipe still ranks: see [Known limitations](#5-known-limitations).

## 4. Evaluation

RecipeNLG has no user ratings, so "relevant" cannot come from ground truth — it has to be constructed. Two protocols, both seeded (`seed=42`) for reproducibility, both against the 100k-recipe subset:

**Protocol A — Masked-recipe retrieval (primary metric: `hit_rate@k`).** Sample `n_eval` recipes from the index (they are *not* removed from it). For each, mask a random `mask_frac` of its canonical ingredient tokens (at least 1 masked, at least 2 kept) and query with the tokens that remain. The recipe is "hit" if it reappears in its own top-k. This measures the real use case directly: can the system recover the right recipe from partial ingredients.

**Protocol B — Overlap relevance (metric: `precision@k`).** Same masked queries as Protocol A. A candidate recipe counts as relevant if the Jaccard similarity between the query tokens and the candidate's canonical tokens is >= `jaccard_threshold` (the origin recipe always counts as relevant by definition). Declared limitation: this label is derived from the same ingredient overlap signal the TF-IDF ranker itself uses, so `precision@k` here is a sanity/regression check, not independent proof of quality — Protocol A is the primary metric.

**Results** (`reports/eval_mode1.json`, 100k-recipe subset, `n_eval_queries=1000`, `mask_frac=0.3`, `jaccard_threshold=0.5`, `seed=42`):

| Metric | Value |
|---|---|
| hit_rate@1 | 0.812 |
| hit_rate@5 | 0.925 |
| hit_rate@10 | 0.953 |
| precision@5 | 0.337 |
| precision@10 | 0.2305 |

Average relevant recipes per query (Protocol B, diagnostic for a degenerate threshold): **1.685** at k=5, **2.305** at k=10 — nontrivial, so the 0.5 Jaccard threshold isn't collapsing to "only the origin recipe counts."

**Honest reading:** `hit_rate@10 = 0.953` is high largely because the origin recipe is present in the index and the query is literally a subset of its own ingredients — that's exactly what Protocol A is designed to measure (recovery from partial information, the real Mode 1 use case), not evidence of some independent notion of "correctness." `precision@k` is lower and, per the limitation above, partially circular with the ranker's own signal — read it as a regression check, not a quality proof.

## 5. Known limitations

This is a portfolio-honesty section, not an apology.

- **Normalization trade-offs are deliberate, not oversights.** `"grnd"` / `"ground"` is preserved as-is rather than stripped, because removing it would be lossy (`ground beef` != `beef`). `"clove"` is intentionally kept in the vocabulary rather than treated as a stopword — it's ambiguous (a unit of garlic *and* the spice) and collapsing it either way would be wrong for the other sense. Suffix-based singularization produces artifacts like `"bay leaves" -> "bay leave"` — a known cost of avoiding a full lemmatizer (decision D-03 in `SPEC.md`).
- **`matched_ingredients` can be empty on a high-scoring result.** The score is computed in the full uni+bigram TF-IDF space, but `matched_ingredients` is only the intersection of canonical unigram tokens. A recipe can rank highly purely on a bigram match (e.g. `"chicken rice"` as a single feature) while showing zero matched unigrams — see result #3 in the quickstart example above. The score reflects the real similarity; `matched`/`missing` are explanatory, not the full story.
- **RecipeNLG's `NER` column occasionally includes non-ingredients** (`"wooden skewer"`, `"toothpick"`) that survive normalization as tokens. This is long-tail noise, not filtered in v1.
- **Evaluation runs on a 100k-recipe subset** of the ~2.2M full RecipeNLG dataset. Scaling to the full dataset is explicitly post-gate work (`RNF-02` in `SPEC.md`).

## 6. Scope & non-goals

These are decisions, not omissions (`SPEC.md` section 2.2):

- **No collaborative filtering** — the dataset has no user ratings to drive it.
- **No generative LLM**, anywhere in the system — the value of a future semantic search mode is retrieval, not generation.
- **No web app, API server, or UI** — this ships as a Python module + CLI scripts, nothing else.
- **No per-user personalization, history, or favorites.**
- **No external infrastructure** — no managed vector DB, no cloud, no Docker. Everything runs locally.
- **No datasets other than RecipeNLG, no translation/multilingual support.**

## 7. Roadmap

Mode 2 (semantic search via sentence embeddings + FAISS) is architecture-only in `SPEC.md` and **not implemented** — it's phase 2, gated on this Mode 1 acceptance. The intended differentiator is a side-by-side comparison of lexical (Mode 1) vs. semantic (Mode 2) retrieval on the same queries, but that comparison doesn't exist yet.

## 8. Tech stack

Python >= 3.11, `pandas`, `pyarrow`, `scikit-learn`, `scipy`, `pytest`. (Mode 2 will add `sentence-transformers` and `faiss-cpu` — not present in this codebase yet.)

---

## Resumo em português

**O que é.** Um recomendador de receitas content-based (Modo 1): você informa os ingredientes disponíveis, o sistema retorna receitas ranqueadas por similaridade de cosseno entre TF-IDF dos seus ingredientes e os ingredientes canônicos de cada receita. Faz parte de um sistema de dois modos — o Modo 2 (busca semântica via embeddings) é arquitetura futura, não implementado ainda.

**Por que essas escolhas.** Sem ratings de usuário no RecipeNLG, então collaborative filtering está fora; TF-IDF + cosseno é um baseline interpretável e sem treino; a coluna `NER` do dataset + normalização própria evita depender de parsing frágil de texto cru; bigramas preservam ingredientes compostos como "cream cheese" como uma única feature.

**Non-goals declarados.** Sem collaborative filtering, sem LLM de geração, sem web app/API/UI, sem personalização por usuário, sem infraestrutura externa, sem outros datasets além do RecipeNLG.

**Quickstart condensado.**
```bash
# 1. Baixe o RecipeNLG manualmente (licenciado) para data/raw/full_dataset.csv
#    https://recipenlg.cs.put.poznan.pl/  ou via Kaggle API

# 2. Ambiente
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Pipeline
python scripts/01_ingest.py --raw data/raw/full_dataset.csv --n-recipes 100000 --seed 42
python scripts/02_build_index.py
python scripts/03_evaluate.py --k 1 5 10 --n-eval 1000 --mask-frac 0.3 --seed 42
python scripts/04_query.py --ingredients "chicken, rice, garlic" --k 10
```

**Avaliação.** hit_rate@10 = 0.953 (Protocolo A, recuperação com informação parcial — métrica principal); precision@5 = 0.337 (Protocolo B, parcialmente circular com o próprio ranker TF-IDF, conforme declarado na seção 8.2 do SPEC — leitura de sanidade, não prova independente de qualidade).
