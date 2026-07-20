"""
Compatibility shim for a known ragas bug: ragas/llms/base.py unconditionally
imports ChatVertexAI from langchain_community.chat_models.vertexai — a path
that newer langchain-community versions removed (VertexAI support moved to
the separate langchain-google-vertexai package). We never use VertexAI, so
rather than pin exact package versions against each other, we inject a
harmless stub module so the import succeeds and ragas loads normally.

MUST be imported before any `import ragas` / `from ragas import ...`
anywhere in this package.
"""
import sys
import types

if "langchain_community.chat_models.vertexai" not in sys.modules:
    _stub = types.ModuleType("langchain_community.chat_models.vertexai")
    _stub.ChatVertexAI = type("ChatVertexAI", (), {})
    sys.modules["langchain_community.chat_models.vertexai"] = _stub