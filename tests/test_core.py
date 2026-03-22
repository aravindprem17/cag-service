"""
tests/test_core.py
==================
Unit tests for the CAG core — no LLM required.
"""

import pytest
from cag_service.core import CAGCache, Document


# ---------------------------------------------------------------------------
# Document tests
# ---------------------------------------------------------------------------

def make_doc(n: int = 1) -> Document:
    return Document(
        id=f"DOC-{n:03}",
        title=f"Test Document {n}",
        content=f"This is the content of document number {n}. " * 20,
        tags=["test", f"group-{n % 3}"],
        metadata={"source": "test"},
    )


def test_document_token_estimate():
    doc = Document(id="d1", title="t", content="a" * 400)
    assert doc.token_estimate == 100


def test_document_context_block_contains_id():
    doc = make_doc(1)
    block = doc.to_context_block()
    assert "DOC-001" in block
    assert "Test Document 1" in block


# ---------------------------------------------------------------------------
# CAGCache tests
# ---------------------------------------------------------------------------

def test_cache_add_and_count():
    cache = CAGCache(name="test")
    assert cache.document_count == 0
    cache.add(make_doc(1))
    assert cache.document_count == 1
    cache.add(make_doc(2))
    assert cache.document_count == 2


def test_cache_add_many():
    cache = CAGCache(name="test")
    cache.add_many([make_doc(i) for i in range(5)])
    assert cache.document_count == 5


def test_cache_remove():
    cache = CAGCache(name="test")
    cache.add(make_doc(1))
    cache.remove("DOC-001")
    assert cache.document_count == 0


def test_cache_replace_on_duplicate_id():
    cache = CAGCache(name="test")
    cache.add(Document(id="X", title="Old", content="old content"))
    cache.add(Document(id="X", title="New", content="new content"))
    assert cache.document_count == 1
    assert cache.get("X").title == "New"


def test_cache_compile_contains_all_documents():
    cache = CAGCache(name="test")
    docs = [make_doc(i) for i in range(3)]
    cache.add_many(docs)
    compiled = cache.compile()
    for doc in docs:
        assert doc.id in compiled


def test_cache_compiled_invalidated_on_add():
    cache = CAGCache(name="test")
    cache.add(make_doc(1))
    _ = cache.compile()
    assert cache._compiled is not None
    cache.add(make_doc(2))
    assert cache._compiled is None  # invalidated


def test_cache_version_is_stable():
    cache = CAGCache(name="test")
    cache.add(make_doc(1))
    v1 = cache.version
    v2 = cache.version
    assert v1 == v2


def test_cache_version_changes_on_add():
    cache = CAGCache(name="test")
    cache.add(make_doc(1))
    v1 = cache.version
    cache.add(make_doc(2))
    v2 = cache.version
    assert v1 != v2


def test_cache_filter_by_tag():
    cache = CAGCache(name="test")
    cache.add(Document(id="A", title="A", content="...", tags=["alpha"]))
    cache.add(Document(id="B", title="B", content="...", tags=["beta"]))
    cache.add(Document(id="C", title="C", content="...", tags=["alpha", "beta"]))
    alpha_docs = cache.filter_by_tag("alpha")
    assert len(alpha_docs) == 2
    assert all("alpha" in d.tags for d in alpha_docs)


def test_cache_stats():
    cache = CAGCache(name="my-cache")
    cache.add(make_doc(1))
    stats = cache.stats()
    assert stats["name"] == "my-cache"
    assert stats["document_count"] == 1
    assert "estimated_tokens" in stats
    assert "cache_version" in stats


def test_cache_preamble_in_compiled():
    cache = CAGCache(name="test", preamble="You are a helpful assistant.")
    cache.add(make_doc(1))
    compiled = cache.compile()
    assert "You are a helpful assistant." in compiled


# ---------------------------------------------------------------------------
# Loader tests
# ---------------------------------------------------------------------------

def test_from_dict():
    from cag_service.loaders import from_dict
    doc = from_dict({"id": "X", "title": "T", "content": "C", "tags": ["a"]})
    assert doc.id == "X"
    assert doc.title == "T"
    assert doc.tags == ["a"]


def test_from_string():
    from cag_service.loaders import from_string
    doc = from_string("Hello world", doc_id="S1", title="Greeting")
    assert doc.content == "Hello world"
    assert doc.id == "S1"


def test_load_example_sops():
    from cag_service.loaders import load_example_sops
    docs = load_example_sops()
    assert len(docs) >= 3
    ids = [d.id for d in docs]
    assert "SOP-001" in ids


def test_load_example_customer_support():
    from cag_service.loaders import load_example_customer_support
    docs = load_example_customer_support()
    assert len(docs) >= 2


# ---------------------------------------------------------------------------
# Backends (import-only, no live LLM)
# ---------------------------------------------------------------------------

def test_backend_map_keys():
    from cag_service.backends import BACKEND_MAP
    assert "ollama" in BACKEND_MAP
    assert "openai" in BACKEND_MAP
    assert "anthropic" in BACKEND_MAP
    assert "huggingface" in BACKEND_MAP


def test_create_backend_unknown_raises():
    from cag_service.backends import create_backend
    with pytest.raises(ValueError, match="Unknown provider"):
        create_backend("unknown_provider", "some-model")
