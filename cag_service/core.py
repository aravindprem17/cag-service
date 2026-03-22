"""
cag_service/core.py
===================
Core Cache-Augmented Generation engine.

CAG Philosophy
--------------
Instead of retrieving documents at query time (RAG), CAG pre-loads a bounded
knowledge corpus directly into the LLM's context window at startup. This gives:
  - Zero retrieval latency
  - No chunking/embedding errors
  - Perfect recall within the cached corpus
  - Simpler architecture (no vector DB required)

Best suited for: SOPs, runbooks, compliance docs, product FAQs —
any corpus that fits within a modern LLM's context window (32k–128k tokens).
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Document:
    """A single knowledge document in the CAG cache."""
    id: str
    title: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    @property
    def token_estimate(self) -> int:
        """Rough token count (1 token ≈ 4 chars)."""
        return len(self.content) // 4

    def to_context_block(self) -> str:
        """Serialise document into a structured LLM-readable block."""
        tag_str = ", ".join(self.tags) if self.tags else "none"
        meta_str = ""
        if self.metadata:
            meta_str = "\n".join(f"  {k}: {v}" for k, v in self.metadata.items())
            meta_str = f"\nMetadata:\n{meta_str}"
        return (
            f"## [{self.id}] {self.title}\n"
            f"Tags: {tag_str}{meta_str}\n\n"
            f"{self.content}"
        )


@dataclass
class CAGResponse:
    """Structured response from a CAG query."""
    answer: str
    documents_used: list[str]        # document IDs referenced
    model: str
    latency_ms: float
    cache_version: str
    raw_messages: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Cache Store
# ---------------------------------------------------------------------------

class CAGCache:
    """
    In-memory document store that compiles the corpus into a reusable
    system prompt string. Re-compilation only occurs when documents change.
    """

    def __init__(self, name: str = "default", preamble: str = ""):
        self.name = name
        self.preamble = preamble
        self._documents: dict[str, Document] = {}
        self._compiled: str | None = None
        self._cache_hash: str = ""

    # ---- Document management -----------------------------------------------

    def add(self, doc: Document) -> "CAGCache":
        """Add or replace a document. Invalidates compiled cache."""
        self._documents[doc.id] = doc
        self._compiled = None
        logger.debug("CAGCache[%s]: added document '%s'", self.name, doc.id)
        return self

    def add_many(self, docs: list[Document]) -> "CAGCache":
        for doc in docs:
            self.add(doc)
        return self

    def remove(self, doc_id: str) -> "CAGCache":
        self._documents.pop(doc_id, None)
        self._compiled = None
        return self

    def get(self, doc_id: str) -> Document | None:
        return self._documents.get(doc_id)

    def filter_by_tag(self, tag: str) -> list[Document]:
        return [d for d in self._documents.values() if tag in d.tags]

    @property
    def document_count(self) -> int:
        return len(self._documents)

    @property
    def estimated_tokens(self) -> int:
        return sum(d.token_estimate for d in self._documents.values())

    # ---- Compilation -------------------------------------------------------

    def compile(self) -> str:
        """
        Compile all documents into a single system-prompt context block.
        Result is cached until documents change.
        """
        if self._compiled is not None:
            return self._compiled

        blocks = [d.to_context_block() for d in self._documents.values()]
        corpus = "\n\n---\n\n".join(blocks)

        self._compiled = (
            f"{self.preamble}\n\n"
            f"# Knowledge Base: {self.name}\n"
            f"({self.document_count} documents | ~{self.estimated_tokens:,} tokens)\n\n"
            f"{corpus}"
        ).strip()

        # Compute a hash to version the cache
        self._cache_hash = hashlib.md5(self._compiled.encode()).hexdigest()[:8]
        logger.info(
            "CAGCache[%s]: compiled %d docs (~%d tokens) | hash=%s",
            self.name, self.document_count, self.estimated_tokens, self._cache_hash,
        )
        return self._compiled

    @property
    def version(self) -> str:
        self.compile()  # ensure hash is up to date
        return self._cache_hash

    def stats(self) -> dict:
        return {
            "name":             self.name,
            "document_count":   self.document_count,
            "estimated_tokens": self.estimated_tokens,
            "cache_version":    self.version,
            "compiled":         self._compiled is not None,
        }


# ---------------------------------------------------------------------------
# CAG Engine
# ---------------------------------------------------------------------------

class CAGEngine:
    """
    Main CAG engine. Wraps a CAGCache and a pluggable LLM backend to provide
    cache-augmented generation queries.

    Usage
    -----
    engine = CAGEngine(cache=my_cache, backend=OllamaBackend("llama3.3"))
    response = engine.query("What is the fix for ORA-01555?")
    print(response.answer)
    """

    def __init__(
        self,
        cache: CAGCache,
        backend: "BaseLLMBackend",
        system_preamble: str = "",
    ):
        self.cache = cache
        self.backend = backend
        self.system_preamble = system_preamble

    def _build_system_prompt(self) -> str:
        base = (
            "You are a precise, helpful assistant with access to a curated knowledge base "
            "provided below. Answer questions using ONLY information from this knowledge base. "
            "If the answer is not found, say so clearly. "
            "Always cite the document ID (e.g. [DOC-001]) when referencing a source.\n\n"
        )
        if self.system_preamble:
            base = self.system_preamble + "\n\n" + base
        return base + self.cache.compile()

    def query(
        self,
        user_message: str,
        history: list[dict] | None = None,
        **kwargs,
    ) -> CAGResponse:
        """
        Run a CAG query against the compiled knowledge cache.

        Parameters
        ----------
        user_message : str
            The user's question or instruction.
        history : list[dict], optional
            Prior conversation turns [{"role": ..., "content": ...}].
        **kwargs
            Passed through to the LLM backend (temperature, max_tokens, etc.)
        """
        t0 = time.perf_counter()

        system_prompt = self._build_system_prompt()
        messages = [
            {"role": "system", "content": system_prompt},
            *(history or []),
            {"role": "user", "content": user_message},
        ]

        answer = self.backend.complete(messages, **kwargs)

        latency_ms = (time.perf_counter() - t0) * 1000

        # Extract cited document IDs from the response (e.g. [DOC-001])
        import re
        cited = list(set(re.findall(r"\[([A-Z0-9\-]+)\]", answer)))

        return CAGResponse(
            answer=answer,
            documents_used=cited,
            model=self.backend.model_name,
            latency_ms=round(latency_ms, 2),
            cache_version=self.cache.version,
            raw_messages=messages,
        )

    def stream_query(
        self,
        user_message: str,
        history: list[dict] | None = None,
        **kwargs,
    ) -> Iterator[str]:
        """Streaming variant — yields text chunks as they arrive."""
        system_prompt = self._build_system_prompt()
        messages = [
            {"role": "system", "content": system_prompt},
            *(history or []),
            {"role": "user", "content": user_message},
        ]
        yield from self.backend.stream(messages, **kwargs)
