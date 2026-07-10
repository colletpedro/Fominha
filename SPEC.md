# spec.md — Fominha v1.0.0

Recomendador de receitas em dois modos. Este documento é o contrato de implementação do v1. O agente de codificação implementa exatamente o que está aqui, na ordem definida na seção 7. Nada fora deste documento é escopo.

---

## 1. Objetivo

Sistema local em Python que, a partir do dataset RecipeNLG, oferece:
- **Modo 1 (recsys estruturado):** recebe uma lista de ingredientes disponíveis e retorna receitas ranqueadas por similaridade content-based (TF-IDF + cosseno sobre ingredientes canônicos).
- **Modo 2 (busca semântica):** recebe uma query em linguagem natural e retorna receitas por similaridade de embeddings (cosseno via FAISS). Documentado neste spec apenas em nível de arquitetura; contratos finais do Modo 2 serão fechados em spec v2 após o gate do Modo 1.

---

## 2. Escopo

### 2.1 In-scope (v1)

- Ingestão e tratamento do RecipeNLG, incluindo pipeline de normalização de ingredientes para tokens canônicos.
- Recomendador content-based do Modo 1 (TF-IDF + cosseno), completo e avaliado.
- Protocolo de avaliação com precision@k e hit-rate@k, calculados e reportados.
- README documentando o Modo 1 como projeto completo em si.
- Arquitetura do Modo 2 (componentes e fluxo), sem implementação além do descrito na seção 5.3.
- Entrega como módulo Python + scripts CLI reproduzíveis. Sem servidor.

### 2.2 Non-goals (proibido implementar no v1)

- **Collaborative filtering** em qualquer forma (exige ratings de usuário que o dataset não tem; possível bônus futuro, não agora).
- **LLM de geração** no Modo 2 ou em qualquer parte (o valor do Modo 2 é retrieval; geração é expansão de escopo).
- **Web app, API server, UI** de qualquer tipo (superfície do v1 é módulo + CLI + README).
- **Personalização por usuário, histórico, perfis, favoritos.**
- **Infra externa** (banco vetorial gerenciado, cloud, Docker). Tudo roda local.
- **Suporte a datasets além do RecipeNLG.**
- **Tradução / multilíngua.** Dataset e queries em inglês; o sistema não traduz.

Se durante a implementação algo parecer exigir um item desta lista, PARE e reporte — não implemente.

---

## 3. Requisitos funcionais

- **RF-01** — O sistema carrega o RecipeNLG a partir de um CSV local (caminho configurável) e produz um dataset tratado persistido em Parquet.
- **RF-02** — O sistema normaliza cada string crua de ingrediente para zero ou mais tokens canônicos (ex.: `"1 (8 ounce) package cream cheese, softened"` → `"cream cheese"`), via pipeline determinístico definido na seção 5.1.2.
- **RF-03** — O sistema constrói e persiste um índice TF-IDF sobre os ingredientes canônicos de todas as receitas tratadas.
- **RF-04** — Dada uma lista de ingredientes (strings livres), o Modo 1 normaliza a lista com o MESMO pipeline do RF-02 e retorna as top-k receitas por similaridade de cosseno, com score e metadados (contrato na seção 6.3).
- **RF-05** — O sistema calcula precision@k e hit-rate@k conforme o protocolo da seção 8 e grava os resultados em `reports/eval_mode1.json`.
- **RF-06** — Toda etapa (ingestão, build de índice, avaliação, query) é executável por script CLI dedicado com seed fixa onde houver aleatoriedade.
- **RF-07** — O Modo 1 expõe uma função pública única `recommend(ingredients, k)` importável como biblioteca (contrato na seção 6.3).

---

## 4. Requisitos não-funcionais

- **RNF-01 — Reprodutibilidade:** toda aleatoriedade (amostragem de avaliação, mascaramento) usa `seed=42` fixa. Dois runs do pipeline completo produzem métricas idênticas.
- **RNF-02 — Subset primeiro:** o pipeline roda por padrão num subset de **100.000 receitas** (amostra aleatória com seed=42 do RecipeNLG completo, ~2M). O tamanho é parâmetro de CLI (`--n-recipes`); o full dataset só entra depois do gate do Modo 1 passar no subset. Justificativa: fechar ponta a ponta antes de pagar o custo de escala.
- **RNF-03 — Latência de query (Modo 1):** `recommend()` responde em < 2s no subset de 100k, em máquina local (MacBook Air), após índice carregado. Carregamento do índice é custo único de inicialização.
- **RNF-04 — Dependências:** Python ≥ 3.11; `pandas`, `pyarrow`, `scikit-learn`, `scipy`. Modo 2 (fase 2) adiciona `sentence-transformers` e `faiss-cpu`. Nada além disso sem alterar este spec.
- **RNF-05 — Artefatos persistidos:** dataset tratado, vetorizador TF-IDF e matriz esparsa são salvos em disco (`data/processed/`, `artifacts/`) e reutilizados; nenhum passo caro roda implicitamente duas vezes.
- **RNF-06 — Dataset ausente:** o RecipeNLG NÃO é versionado no repo. O README instrui o download manual; scripts falham com mensagem clara se o CSV não existir (seção 10).

---

## 5. Arquitetura e componentes

### 5.0 Layout do repositório

```
fominha/
├── spec.md
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                  # RecipeNLG CSV (gitignored)
│   └── processed/            # recipes.parquet (gitignored)
├── artifacts/                # tfidf_vectorizer.joblib, tfidf_matrix.npz (gitignored)
├── reports/                  # eval_mode1.json (versionado)
├── src/fominha/
│   ├── __init__.py
│   ├── ingest.py             # carga + tratamento do RecipeNLG
│   ├── normalize.py          # pipeline de normalização de ingredientes
│   ├── mode1/
│   │   ├── __init__.py
│   │   ├── index.py          # build/load do índice TF-IDF
│   │   └── recommend.py      # função pública recommend()
│   └── eval/
│       └── protocol.py       # protocolo de avaliação (seção 8)
├── scripts/
│   ├── 01_ingest.py
│   ├── 02_build_index.py
│   ├── 03_evaluate.py
│   └── 04_query.py           # CLI de query manual do Modo 1
└── tests/
    ├── test_normalize.py
    └── test_recommend.py
```

### 5.1 Componente: ingestão e normalização (`ingest.py`, `normalize.py`)

#### 5.1.1 Ingestão

Responsabilidade: ler o CSV do RecipeNLG, selecionar colunas, amostrar subset, aplicar normalização, descartar receitas inválidas, persistir Parquet. Justificativa do Parquet: leitura rápida e tipada nas etapas seguintes, sem re-parsear CSV de 2GB.

Colunas usadas do RecipeNLG: `title`, `ingredients` (lista de strings crua), `directions`, `NER` (lista de entidades de ingrediente pré-extraídas pelo dataset), `link`.

**Decisão:** a coluna `NER` do RecipeNLG é a FONTE PRIMÁRIA de ingredientes canônicos — o dataset já entrega entidades extraídas ("cream cheese", "sugar"), o que remove a necessidade de parsear quantidade/unidade do texto cru. O pipeline de normalização (5.1.2) atua SOBRE a coluna NER para padronizá-la, e o mesmo pipeline atua sobre o input do usuário em runtime. Justificativa: parsing completo de strings cruas é o maior risco de qualidade do Modo 1; usar NER + normalização própria reduz esse risco sem terceirizar a padronização.

#### 5.1.2 Pipeline de normalização (aplica-se a NER do dataset E ao input do usuário)

Etapas, nesta ordem, todas determinísticas:

1. `lowercase` e strip.
2. Remoção de pontuação e conteúdo entre parênteses.
3. Remoção de tokens de quantidade/unidade/estado via stoplist fixa em `normalize.py` (constante `INGREDIENT_STOPWORDS`), incluindo no mínimo: números, frações unicode, `cup, cups, tablespoon, tbsp, teaspoon, tsp, ounce, oz, pound, lb, gram, g, kg, ml, package, pkg, can, jar, box, slice, sliced, chopped, diced, minced, fresh, frozen, softened, melted, optional, large, small, medium`.
4. Singularização por regra simples de sufixo (`tomatoes → tomato`, `s` final removido exceto em whitelist `molasses, couscous, hummus`). Justificativa: lematizador completo (spaCy) é dependência pesada para ganho marginal num vocabulário de ingredientes.
5. Colapso de espaços; descarte de tokens resultantes com < 3 caracteres.
6. Resultado: lista de strings canônicas por receita, sem duplicatas, ordem preservada.

Regra de descarte de receita: após normalização, receitas com `len(ingredients_canonical) < 2` ou `title` vazio são removidas do dataset tratado.

**Nota de prioridade para o implementador:** este pipeline é o corpo do Modo 1, não pré-processamento cosmético. `test_normalize.py` deve cobrir no mínimo 15 casos reais extraídos do RecipeNLG, incluindo frações, parênteses, unidades compostas e plurais.

### 5.2 Componente: Modo 1 (`mode1/index.py`, `mode1/recommend.py`)

- `index.py`: constrói `TfidfVectorizer` do scikit-learn sobre o campo `ingredients_canonical` de cada receita (documento = tokens canônicos unidos por espaço; `analyzer='word'`, `ngram_range=(1,2)`, `min_df=5`). Persiste vetorizador (`joblib`) e matriz esparsa (`.npz`). Justificativa dos bigramas: preservar ingredientes compostos ("cream cheese", "olive oil") como feature única.
- `recommend.py`: carrega artefatos, normaliza input do usuário com `normalize.py`, vetoriza, calcula cosseno contra a matriz (`sklearn.metrics.pairwise.cosine_similarity` ou produto esparso equivalente), retorna top-k.

### 5.3 Componente: Modo 2 — SOMENTE ARQUITETURA (fase 2; NÃO implementar agora)

Registrado aqui para que a estrutura do repo e as decisões de stack já nasçam compatíveis. **Contratos de interface detalhados e critérios de aceite do Modo 2 serão fechados em spec v2, após o gate do Modo 1 (seção 9) passar.** O agente NÃO implementa nada desta subseção no v1.

- **Corpus de embedding:** texto por receita = `title + ingredients_canonical + directions` truncado.
- **Modelo:** `sentence-transformers` local, `all-MiniLM-L6-v2` (384 dims). Justificativa: roda em CPU, qualidade suficiente para retrieval, zero custo de API.
- **Indexação:** embeddings gerados offline em batch; índice `faiss.IndexFlatIP` local com vetores L2-normalizados (produto interno = cosseno). Justificativa: no subset de 100k, busca exata é rápida e elimina tuning de índice aproximado.
- **Runtime:** query do usuário → embedding com o mesmo modelo → top-k vizinhos no FAISS → mesmos metadados de output do Modo 1.
- **Sem LLM de geração em nenhum ponto.**
- **Entregável diferencial (vai no README final):** comparação lado a lado Modo 1 vs. Modo 2 com exemplos onde o match léxico acerta e o semântico erra, e vice-versa.

---

## 6. Contratos de interface

### 6.1 Dataset tratado (`data/processed/recipes.parquet`)

```python
# Uma linha por receita. Tipos pandas/pyarrow:
{
    "recipe_id":              "int64",      # índice sequencial estável pós-tratamento, 0..N-1
    "title":                  "string",     # não-vazio
    "ingredients_raw":        "list<string>",  # strings originais do RecipeNLG, intocadas
    "ingredients_canonical":  "list<string>",  # saída do pipeline 5.1.2, len >= 2, sem duplicatas
    "directions":             "string",     # passos unidos por "\n"
    "link":                   "string",     # URL de origem; pode ser vazio
}
```

### 6.2 Artefatos do índice TF-IDF

```
artifacts/tfidf_vectorizer.joblib   # sklearn TfidfVectorizer fitted
artifacts/tfidf_matrix.npz          # scipy.sparse.csr_matrix, shape (N, V), alinhada a recipe_id
artifacts/index_meta.json           # {"n_recipes": int, "vocab_size": int, "built_at": iso8601, "seed": 42, "n_recipes_param": int}
```

### 6.3 Função pública do Modo 1

```python
def recommend(
    ingredients: list[str],   # strings livres do usuário, ex.: ["chicken breast", "2 cups rice"]
    k: int = 10,              # 1 <= k <= 100; fora do range -> ValueError
) -> list[Recommendation]

# Retorno ordenado por score desc. len(retorno) == min(k, receitas com score > 0).
@dataclass(frozen=True)
class Recommendation:
    recipe_id: int
    title: str
    score: float                        # cosseno em [0.0, 1.0]
    matched_ingredients: list[str]      # tokens canônicos da query presentes na receita
    missing_ingredients: list[str]      # tokens canônicos da RECEITA ausentes na query
    link: str
```

Erros (detalhe na seção 10): `EmptyQueryError` se, após normalização, a query não tiver nenhum token canônico; `IndexNotBuiltError` se artefatos ausentes.

### 6.4 CLI dos scripts

```
python scripts/01_ingest.py      --raw data/raw/full_dataset.csv --n-recipes 100000 --seed 42
python scripts/02_build_index.py
python scripts/03_evaluate.py    --k 1 5 10 --n-eval 1000 --mask-frac 0.3 --seed 42
python scripts/04_query.py       --ingredients "chicken, rice, garlic" --k 10
```

### 6.5 Relatório de avaliação (`reports/eval_mode1.json`)

```json
{
  "protocol": "masked-recipe-retrieval + overlap-relevance",
  "n_recipes_index": 100000,
  "n_eval_queries": 1000,
  "mask_frac": 0.3,
  "seed": 42,
  "jaccard_threshold": 0.5,
  "metrics": {
    "hit_rate@1": 0.0,
    "hit_rate@5": 0.0,
    "hit_rate@10": 0.0,
    "precision@5": 0.0,
    "precision@10": 0.0
  },
  "built_at": "2026-07-10T00:00:00Z"
}
```

### 6.6 Artefato de embeddings do Modo 2 (fase 2 — formato reservado, não gerar agora)

```
artifacts/embeddings.faiss          # faiss.IndexFlatIP, vetores L2-normalizados, 384 dims
artifacts/embeddings_meta.json      # {"model": "all-MiniLM-L6-v2", "dims": 384, "n_recipes": int, "aligned_to": "recipe_id"}
```

---

## 7. Fluxos críticos (ordem de implementação)

### 7.1 Fluxo 1 — Ingestão e tratamento

1. Ler CSV cru (`--raw`). Se ausente → erro E-01 (seção 10).
2. Amostrar `--n-recipes` linhas com `--seed`.
3. Parsear colunas `ingredients` e `NER` (armazenadas como string de lista no CSV → `ast.literal_eval` com fallback de descarte da linha se o parse falhar).
4. Aplicar pipeline 5.1.2 sobre `NER` → `ingredients_canonical`.
5. Descartar receitas inválidas (regra em 5.1.2).
6. Atribuir `recipe_id` sequencial e gravar Parquet.
7. Logar: linhas lidas, descartadas por motivo, total final.

### 7.2 Fluxo 2 — Modo 1, ingrediente → ranking

1. `recommend(ingredients, k)` valida `k` e normaliza o input com `normalize.py` (mesmo pipeline do dataset — obrigatório para o espaço de features coincidir).
2. Se zero tokens canônicos → `EmptyQueryError`.
3. Vetorizar tokens unidos por espaço com o vetorizador persistido.
4. Cosseno contra a matriz completa; `argsort` desc; cortar em `k`; descartar scores `== 0`.
5. Montar `Recommendation` com `matched_ingredients` / `missing_ingredients` por interseção de conjuntos canônicos.

### 7.3 Fluxo 3 — Modo 2, query → retrieval (fase 2, não implementar)

1. Query → embedding (mesmo modelo do índice) → L2-normalize.
2. `index.search(vec, k)` → recipe_ids + scores.
3. Mapear para os mesmos metadados de output do Modo 1.

---

## 8. Protocolo de avaliação (obrigatório antes de reportar qualquer número)

Não existem ratings de usuário no RecipeNLG; portanto "relevante" é CONSTRUÍDO. Dois protocolos, ambos em `eval/protocol.py`, ambos com seed=42:

### 8.1 Protocolo A — Masked-recipe retrieval (métrica principal: hit-rate@k)

- **Divisão:** amostrar `n_eval=1000` receitas do índice (sem removê-las do índice; a tarefa é reencontrá-las com informação parcial).
- **Construção da query:** para cada receita, mascarar aleatoriamente `mask_frac=0.3` dos tokens canônicos (mínimo 1 mascarado, mínimo 2 mantidos; receitas com < 3 tokens ficam fora da amostra de avaliação). A query é a lista dos tokens mantidos.
- **Relevante:** exclusivamente a receita de origem.
- **Métrica:** `hit_rate@k = (# queries cuja receita de origem aparece no top-k) / n_eval`, para `k ∈ {1, 5, 10}`.
- Justificativa: mede diretamente a capacidade do sistema de recuperar a receita certa a partir de ingredientes parciais — o caso de uso real do Modo 1 — sem rótulo externo.

### 8.2 Protocolo B — Overlap relevance (métrica: precision@k, exigida pelo gate)

- **Queries:** as mesmas 1000 queries mascaradas do Protocolo A.
- **Relevante:** receita candidata `r` é relevante para a query `q` se `jaccard(set(q), set(ingredients_canonical(r))) >= 0.5`, calculado sobre tokens canônicos. A receita de origem conta como relevante por definição.
- **Métrica:** `precision@k = (# relevantes no top-k) / k`, média sobre as 1000 queries, para `k ∈ {5, 10}`.
- **Limitação declarada (vai no README):** o rótulo é derivado de sobreposição de ingredientes, correlacionado com o próprio sinal do ranker TF-IDF; precision@k aqui é métrica de sanidade/regressão, não prova de qualidade independente — o Protocolo A é o principal.

### 8.3 Regras gerais

- Nenhuma métrica fora destes dois protocolos pode ser reportada sem aditivo a este spec.
- Resultados gravados em `reports/eval_mode1.json` (contrato 6.5) e transcritos no README.
- Não há alvo mínimo numérico no v1: o critério de aceite é o protocolo rodar corretamente e os números serem reportados com honestidade, incluindo a limitação do Protocolo B.

---

## 9. Critérios de aceite

O Modo 1 (e o v1 como um todo) está aceito quando TODOS os itens abaixo estão verdadeiros. Itens 1–4 são o gate travado, verbatim; o Modo 2 não inicia antes deles:

1. **Dataset carregado e tratado** — `recipes.parquet` existe, cumpre o contrato 6.1, e o log de ingestão reporta linhas lidas/descartadas.
2. **Recomendador content-based rodando** — `recommend()` cumpre o contrato 6.3; `scripts/04_query.py` retorna resultados coerentes para 3 queries manuais documentadas no README.
3. **Pelo menos uma métrica de recsys calculada (precision@k)** — `scripts/03_evaluate.py` executa os Protocolos A e B da seção 8 e grava `reports/eval_mode1.json` válido.
4. **README documentando o Modo 1 sozinho, como se o projeto parasse aí** — contém: o que o sistema faz, como baixar o dataset, como rodar os 4 scripts, resultados de avaliação com a limitação do Protocolo B declarada, e decisões técnicas resumidas.

Adicionais de qualidade (também bloqueiam o aceite):

5. `test_normalize.py` (≥ 15 casos reais) e `test_recommend.py` (ordenação por score, `EmptyQueryError`, validação de `k`) passam via `pytest`.
6. Dois runs completos do pipeline com os mesmos parâmetros produzem `eval_mode1.json` idêntico (RNF-01).
7. Nenhum item da lista de non-goals (2.2) foi implementado.

---

## 10. Edge cases e tratamento de erro

| ID | Caso | Comportamento obrigatório |
|----|------|---------------------------|
| E-01 | CSV do RecipeNLG ausente em `data/raw/` | Script falha com exit code 1 e mensagem indicando o caminho esperado e a instrução de download do README. Não baixa nada automaticamente. |
| E-02 | Linha do CSV com `ingredients`/`NER` não-parseável | Descartar a linha, contar no log de ingestão. Nunca abortar o run inteiro por linha ruim. |
| E-03 | Receita com < 2 ingredientes canônicos pós-normalização | Descartada no tratamento (regra 5.1.2). |
| E-04 | `recommend()` com lista vazia ou que normaliza para zero tokens | `EmptyQueryError` com mensagem citando exemplos de input válido. |
| E-05 | Ingrediente do usuário inexistente no vocabulário TF-IDF | Não é erro: token fora do vocabulário contribui com peso zero. Se TODOS os tokens forem OOV, o cosseno é 0 para tudo → retornar lista vazia (score 0 é descartado por 7.2 passo 4). |
| E-06 | `k` fora de `[1, 100]` ou não-inteiro | `ValueError`. |
| E-07 | Artefatos do índice ausentes ao chamar `recommend()` | `IndexNotBuiltError` instruindo rodar `scripts/02_build_index.py`. |
| E-08 | Receita no dataset sem `directions` | Permitida no Modo 1 (não usa `directions`); campo gravado como string vazia. |
| E-09 | Query vazia no Modo 2 | Fase 2 — comportamento será definido no spec v2 (reservado: erro análogo a E-04). |
| E-10 | `--n-recipes` maior que o dataset | Usar o dataset inteiro e logar aviso. |

---

## 11. Decisões técnicas (ADR-lite)

| # | Decisão | Porquê (1 linha) |
|---|---------|------------------|
| D-01 | Content-based no Modo 1; sem collaborative filtering | RecipeNLG não tem ratings de usuário; colaborativo sofre cold-start e é non-goal declarado do v1. |
| D-02 | TF-IDF + cosseno como similaridade do Modo 1 | Baseline interpretável, sem treino, sem ratings, e serve de contraste léxico honesto contra o Modo 2 semântico. |
| D-03 | Coluna `NER` do RecipeNLG como fonte de ingredientes + pipeline próprio de normalização | Parsing de strings cruas é o maior risco de qualidade do Modo 1; NER pré-extraído + normalização determinística reduz o risco mantendo controle. |
| D-04 | Mesmo pipeline de normalização para dataset e input do usuário | Espaços de features divergentes entre index-time e query-time quebram o cosseno silenciosamente. |
| D-05 | Bigramas no TF-IDF (`ngram_range=(1,2)`) | Ingredientes compostos ("olive oil") viram feature única em vez de dois tokens diluídos. |
| D-06 | Subset de 100k receitas como default; full 2M só pós-gate | Fechar ponta a ponta barato antes de pagar compute de escala; embeddar 2M no Modo 2 é custo real. |
| D-07 | FAISS local `IndexFlatIP` no Modo 2 | Zero infra, busca exata suficiente em 100k, elimina tuning de índice aproximado. |
| D-08 | `sentence-transformers` local (`all-MiniLM-L6-v2`) no Modo 2 | Roda em CPU, sem custo de API, qualidade adequada para retrieval de receitas. |
| D-09 | Módulo Python + scripts CLI + README; sem web app/API | Superfície mínima entregável e reproduzível; servidor é non-goal do v1. |
| D-10 | Relevância construída (masked retrieval + overlap Jaccard) no protocolo de avaliação | Sem ratings não existe rótulo pronto; métrica sem protocolo construído mediria ruído. |
| D-11 | hit-rate@k como métrica principal; precision@k como sanidade com limitação declarada | precision@k com rótulo por sobreposição é parcialmente circular com o ranker; honestidade metodológica vai no README. |
| D-12 | Parquet como formato do dataset tratado | Leitura tipada e rápida entre etapas, sem re-parsear CSV de ~2GB. |

---

## Decisões a confirmar com o Pedro

Única seção não-determinística do documento. O corpo acima já assume as recomendações abaixo; se discordar, o ajuste é pontual.

1. **`mask_frac=0.3` e `n_eval=1000` no protocolo de avaliação.** Recomendação: manter. Trade-off: mask_frac maior torna a tarefa mais difícil (números menores, mais discriminativos), mas menos representativa de uso real; n_eval maior dá estabilidade estatística ao custo de tempo de avaliação.
2. **`jaccard_threshold=0.5` no Protocolo B.** Recomendação: manter 0.5 e reportar também o número de relevantes médio por query no JSON, para detectar threshold degenerado (quase nenhum relevante). Trade-off: threshold alto → precision subestimada; baixo → métrica inflada e ainda mais circular.
3. **Truncamento do corpus de embedding do Modo 2** (título + ingredientes + N primeiros caracteres de `directions`). Recomendação: decidir só no spec v2, com teste rápido de qualidade; não bloqueia nada do v1.