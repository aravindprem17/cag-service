"""
examples/multi_backend.py
==========================
Shows how to switch between LLM providers while keeping the same cache.
"""

import os
from cag_service import CAGCache, CAGEngine, create_backend, load_example_sops

# Shared cache
cache = CAGCache(name="erp-sops")
cache.add_many(load_example_sops())
print(cache.stats())

# --- Choose a backend via environment variable ----------------------------
provider = os.getenv("CAG_PROVIDER", "ollama")
model    = os.getenv("CAG_MODEL",    "llama3.3")

# Examples:
#   CAG_PROVIDER=ollama    CAG_MODEL=llama3.3          (local, free)
#   CAG_PROVIDER=openai    CAG_MODEL=gpt-4o            (needs OPENAI_API_KEY)
#   CAG_PROVIDER=anthropic CAG_MODEL=claude-sonnet-4-20250514 (needs ANTHROPIC_API_KEY)
#   CAG_PROVIDER=huggingface CAG_MODEL=mistralai/Mistral-7B-Instruct-v0.3

backend = create_backend(provider, model)
engine  = CAGEngine(cache=cache, backend=backend)

print(f"\nUsing backend: {backend}")

question = "What are the steps to fix ORA-01555?"
response = engine.query(question)

print(f"\nQ: {question}")
print(f"A: {response.answer}")
print(f"\nLatency: {response.latency_ms} ms | Model: {response.model}")
