"""
examples/basic_usage.py
========================
Minimal example: load documents, run a query.
"""

from cag_service import CAGCache, CAGEngine, OllamaBackend, Document

# 1. Build a cache
cache = CAGCache(name="my-sops", preamble="You are an expert support engineer.")

# 2. Add knowledge documents
cache.add(Document(
    id="SOP-001",
    title="Password Reset Procedure",
    content="""
Users can reset passwords via the self-service portal at /reset.
If MFA is lost, the account must be unlocked by IT via ticket queue IT-ACCESS.
Temporary passwords expire after 8 hours.
    """.strip(),
    tags=["auth", "security"],
))

cache.add(Document(
    id="SOP-002",
    title="VPN Connectivity Issues",
    content="""
Common causes: expired certificate, split-tunnel misconfiguration, DNS conflict.
Resolution:
  1. Ensure VPN client is updated to v5.x or later.
  2. Re-import the latest .ovpn profile from the IT portal.
  3. Flush DNS: ipconfig /flushdns (Windows) or sudo dscacheutil -flushcache (macOS).
  4. Escalate to netops@company.com if unresolved after 30 min.
    """.strip(),
    tags=["network", "vpn"],
))

print(cache.stats())

# 3. Connect an LLM backend
backend = OllamaBackend(model="llama3.3")  # must have Ollama running locally

# 4. Create the engine
engine = CAGEngine(cache=cache, backend=backend)

# 5. Query
response = engine.query("A user can't connect to VPN after renewing their laptop. What should I tell them?")

print(f"\n📚 Cache version : {response.cache_version}")
print(f"⚡ Latency       : {response.latency_ms} ms")
print(f"📄 Docs cited    : {response.documents_used}")
print(f"\n💬 Answer:\n{response.answer}")
