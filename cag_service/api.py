"""
cag_service/api.py
==================
FastAPI REST service that exposes the CAG engine as a hosted microservice.

Endpoints
---------
GET  /                          Health check
GET  /cache/stats               Cache metadata
GET  /cache/documents           List all documents
POST /cache/documents           Add a document
DELETE /cache/documents/{id}    Remove a document
POST /query                     CAG query (blocking)
POST /query/stream              CAG query (streaming SSE)
"""

from __future__ import annotations

import logging
import os
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .backends import create_backend
from .core import CAGCache, CAGEngine, Document
from .loaders import load_example_sops

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(cache: CAGCache | None = None, engine: CAGEngine | None = None) -> FastAPI:
    """
    Create the FastAPI application.

    If no cache/engine is provided, defaults are built from environment variables:
      CAG_PROVIDER  – LLM provider  (default: ollama)
      CAG_MODEL     – model name    (default: llama3.3)
      CAG_CORPUS    – "sops" | "customer_support" | "" (optional seed)
    """
    app = FastAPI(
        title="CAG Service",
        description="Cache-Augmented Generation as a Service — pre-load your knowledge, query with any LLM.",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---- Bootstrap cache & engine -----------------------------------------
    if cache is None:
        cache = CAGCache(
            name=os.getenv("CAG_CACHE_NAME", "default"),
            preamble=os.getenv("CAG_SYSTEM_PREAMBLE", ""),
        )
        corpus = os.getenv("CAG_CORPUS", "sops")
        if corpus == "sops":
            cache.add_many(load_example_sops())
        elif corpus == "customer_support":
            from .loaders import load_example_customer_support
            cache.add_many(load_example_customer_support())

    if engine is None:
        provider = os.getenv("CAG_PROVIDER", "ollama")
        model    = os.getenv("CAG_MODEL",    "llama3.3")
        try:
            backend = create_backend(provider, model)
            engine  = CAGEngine(cache=cache, backend=backend)
            logger.info("CAG Engine started: provider=%s model=%s", provider, model)
        except Exception as exc:
            logger.warning("Could not initialise LLM backend: %s", exc)
            engine = None  # type: ignore[assignment]

    # Store on app state for access in routes
    app.state.cache  = cache
    app.state.engine = engine

    # ---- Pydantic models ---------------------------------------------------

    class DocumentIn(BaseModel):
        id: str = Field(..., description="Unique document identifier e.g. SOP-042")
        title: str
        content: str
        tags: list[str] = []
        metadata: dict = {}

    class QueryRequest(BaseModel):
        question: str = Field(..., description="Natural-language question to answer from the cache")
        history: list[dict] = Field(default=[], description="Prior conversation turns")
        temperature: float = Field(default=0.2, ge=0.0, le=2.0)

    class QueryResponse(BaseModel):
        answer: str
        documents_used: list[str]
        model: str
        latency_ms: float
        cache_version: str

    # ---- Routes ------------------------------------------------------------

    @app.get("/", tags=["Health"])
    def root():
        return {
            "service": "CAG Service",
            "version": "1.0.0",
            "status":  "ok",
            "cache_docs": app.state.cache.document_count,
        }

    @app.get("/cache/stats", tags=["Cache"])
    def cache_stats():
        return app.state.cache.stats()

    @app.get("/cache/documents", tags=["Cache"])
    def list_documents():
        docs = [
            {
                "id":              d.id,
                "title":           d.title,
                "tags":            d.tags,
                "token_estimate":  d.token_estimate,
                "metadata":        d.metadata,
            }
            for d in app.state.cache._documents.values()
        ]
        return {"count": len(docs), "documents": docs}

    @app.post("/cache/documents", status_code=201, tags=["Cache"])
    def add_document(doc_in: DocumentIn):
        doc = Document(
            id=doc_in.id,
            title=doc_in.title,
            content=doc_in.content,
            tags=doc_in.tags,
            metadata=doc_in.metadata,
        )
        app.state.cache.add(doc)
        return {
            "message":       f"Document '{doc.id}' added.",
            "cache_version": app.state.cache.version,
            "total_docs":    app.state.cache.document_count,
        }

    @app.delete("/cache/documents/{doc_id}", tags=["Cache"])
    def remove_document(doc_id: str):
        if app.state.cache.get(doc_id) is None:
            raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
        app.state.cache.remove(doc_id)
        return {"message": f"Document '{doc_id}' removed.", "total_docs": app.state.cache.document_count}

    @app.post("/query", response_model=QueryResponse, tags=["Query"])
    def query(req: QueryRequest):
        if app.state.engine is None:
            raise HTTPException(status_code=503, detail="LLM backend not initialised.")
        if app.state.cache.document_count == 0:
            raise HTTPException(status_code=400, detail="Cache is empty. Add documents first.")

        result = app.state.engine.query(
            user_message=req.question,
            history=req.history,
            temperature=req.temperature,
        )
        return QueryResponse(
            answer=result.answer,
            documents_used=result.documents_used,
            model=result.model,
            latency_ms=result.latency_ms,
            cache_version=result.cache_version,
        )

    @app.post("/query/stream", tags=["Query"])
    def query_stream(req: QueryRequest):
        """Server-Sent Events streaming endpoint."""
        if app.state.engine is None:
            raise HTTPException(status_code=503, detail="LLM backend not initialised.")

        def event_generator():
            for chunk in app.state.engine.stream_query(
                user_message=req.question,
                history=req.history,
                temperature=req.temperature,
            ):
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    return app


# ---------------------------------------------------------------------------
# Standalone entry-point
# ---------------------------------------------------------------------------
app = create_app()
