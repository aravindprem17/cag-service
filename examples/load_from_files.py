"""
examples/load_from_files.py
============================
Demonstrates loading documents from .txt, .md files, JSON, and directories.
"""

from pathlib import Path
from cag_service import CAGCache, CAGEngine, OllamaBackend
from cag_service.loaders import (
    from_text_file,
    from_markdown_file,
    from_json_file,
    from_directory,
    from_string,
)

cache = CAGCache(name="file-demo")

# --- 1. From a plain string -----------------------------------------------
cache.add(from_string(
    content="All support tickets must be acknowledged within 2 business hours.",
    doc_id="POLICY-SLA",
    title="SLA Policy",
    tags=["policy", "sla"],
))

# --- 2. From a .txt file --------------------------------------------------
# cache.add(from_text_file("knowledge/runbook.txt"))

# --- 3. From a Markdown file ----------------------------------------------
# cache.add(from_markdown_file("knowledge/onboarding.md"))

# --- 4. From a JSON file (list of objects) --------------------------------
# docs = from_json_file("knowledge/faqs.json")
# cache.add_many(docs)

# --- 5. From an entire directory ------------------------------------------
# All .txt and .md files under ./knowledge/
# docs = from_directory("./knowledge", extensions=[".txt", ".md"], recursive=True)
# cache.add_many(docs)

print(cache.stats())

# Query
backend = OllamaBackend("llama3.3")
engine  = CAGEngine(cache=cache, backend=backend)
resp    = engine.query("What is the SLA for ticket acknowledgement?")
print(resp.answer)
