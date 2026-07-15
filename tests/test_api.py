import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from fastapi.testclient import TestClient

from fominha.api.app import _normalize_link, _tokenize_for_mode1, app

# TestClient como context manager aciona o lifespan (D-43), carregando os
# indices reais uma unica vez para toda a suite.
_client_ctx = TestClient(app)
client = _client_ctx.__enter__()

RECOMMEND_FIELDS = {
    "recipe_id", "title", "score", "matched_ingredients",
    "missing_ingredients", "link",
}
SEARCH_FIELDS = {"recipe_id", "title", "score", "link"}


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #

def test_recommend_response_schema():
    resp = client.post("/api/recommend", json={"query": "chicken, rice, garlic", "k": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, list)
    assert len(body) <= 3
    for item in body:
        assert set(item.keys()) == RECOMMEND_FIELDS
        assert isinstance(item["recipe_id"], int)
        assert isinstance(item["title"], str)
        assert isinstance(item["score"], float)
        assert isinstance(item["matched_ingredients"], list)
        assert isinstance(item["missing_ingredients"], list)
        assert isinstance(item["link"], str)


def test_search_response_schema():
    resp = client.post("/api/search", json={"query": "chicken rice garlic", "k": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, list)
    assert len(body) <= 3
    for item in body:
        assert set(item.keys()) == SEARCH_FIELDS
        assert isinstance(item["recipe_id"], int)
        assert isinstance(item["title"], str)
        assert isinstance(item["score"], float)
        assert isinstance(item["link"], str)


def test_search_response_has_no_matched_missing_fields():
    resp = client.post("/api/search", json={"query": "chicken rice garlic", "k": 3})
    body = resp.json()
    for item in body:
        assert "matched_ingredients" not in item
        assert "missing_ingredients" not in item


# --------------------------------------------------------------------------- #
# 422 validation
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("endpoint", ["/api/recommend", "/api/search"])
def test_empty_query_is_422(endpoint):
    resp = client.post(endpoint, json={"query": "", "k": 5})
    assert resp.status_code == 422


@pytest.mark.parametrize("endpoint", ["/api/recommend", "/api/search"])
def test_whitespace_query_is_422(endpoint):
    resp = client.post(endpoint, json={"query": "   ", "k": 5})
    assert resp.status_code == 422


@pytest.mark.parametrize("endpoint", ["/api/recommend", "/api/search"])
def test_k_zero_is_422(endpoint):
    resp = client.post(endpoint, json={"query": "chicken", "k": 0})
    assert resp.status_code == 422


@pytest.mark.parametrize("endpoint", ["/api/recommend", "/api/search"])
def test_k_over_limit_is_422(endpoint):
    resp = client.post(endpoint, json={"query": "chicken", "k": 21})
    assert resp.status_code == 422


@pytest.mark.parametrize("endpoint", ["/api/recommend", "/api/search"])
def test_k_non_integer_is_422(endpoint):
    resp = client.post(endpoint, json={"query": "chicken", "k": 1.5})
    assert resp.status_code == 422


# --------------------------------------------------------------------------- #
# E-31: query sem ingrediente reconhecivel -> 200 + []
# --------------------------------------------------------------------------- #

def test_recommend_unrecognizable_query_returns_empty_200():
    resp = client.post("/api/recommend", json={"query": "cozy wonderful amazing", "k": 5})
    assert resp.status_code == 200
    assert resp.json() == []


# --------------------------------------------------------------------------- #
# D-42: tokenizacao hibrida (virgula preserva ingrediente composto)
# --------------------------------------------------------------------------- #

def test_tokenize_comma_preserves_compound_ingredient():
    assert _tokenize_for_mode1("cream cheese, sugar") == ["cream cheese", "sugar"]


def test_tokenize_no_comma_splits_by_word():
    assert _tokenize_for_mode1("cozy winter dessert") == ["cozy", "winter", "dessert"]


def test_recommend_comma_query_preserves_compound_in_matched():
    resp = client.post("/api/recommend", json={"query": "cream cheese, sugar", "k": 5})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) > 0
    assert any("cream cheese" in item["matched_ingredients"] for item in body)
    # nao deve degradar em tokens separados "cream" e "cheese"
    for item in body:
        assert "cream" not in item["matched_ingredients"]
        assert "cheese" not in item["matched_ingredients"]


# --------------------------------------------------------------------------- #
# Normalizacao do campo link (RecipeNLG guarda links sem esquema)
# --------------------------------------------------------------------------- #

def test_normalize_link_adds_https_when_scheme_missing():
    assert _normalize_link("www.cookbooks.com/Recipe-Details.aspx?id=857036") == (
        "https://www.cookbooks.com/Recipe-Details.aspx?id=857036"
    )


def test_normalize_link_leaves_https_unchanged():
    url = "https://www.cookbooks.com/Recipe-Details.aspx?id=857036"
    assert _normalize_link(url) == url


def test_normalize_link_leaves_http_unchanged():
    url = "http://www.cookbooks.com/Recipe-Details.aspx?id=857036"
    assert _normalize_link(url) == url


def test_normalize_link_empty_stays_empty():
    assert _normalize_link("") == ""
    assert _normalize_link(None) == ""


def test_recommend_links_have_scheme_and_no_double_prefix():
    resp = client.post("/api/recommend", json={"query": "chicken, rice, garlic", "k": 5})
    body = resp.json()
    assert len(body) > 0
    for item in body:
        link = item["link"]
        assert link == "" or link.startswith("http://") or link.startswith("https://")
        assert "https://https://" not in link
        assert "https://http://" not in link


def test_search_links_have_scheme_and_no_double_prefix():
    resp = client.post("/api/search", json={"query": "fried rice", "k": 5})
    body = resp.json()
    assert len(body) > 0
    for item in body:
        link = item["link"]
        assert link == "" or link.startswith("http://") or link.startswith("https://")
        assert "https://https://" not in link
        assert "https://http://" not in link
