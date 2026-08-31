"""Reusable ChromaDB retrieval wrapper for the troubleshooting workflow."""

from __future__ import annotations

import os
from typing import Any

try:
    import chromadb
except ImportError:  # pragma: no cover - dependency guard
    chromadb = None


def _get_collection(collection_name: str = "support_knowledge") -> Any:
    """Return the persisted Chroma collection for knowledge retrieval."""

    if chromadb is None:
        raise RuntimeError("chromadb is not installed")

    persist_dir = os.getenv("CHROMA_PERSIST_DIRECTORY", os.path.join("data", "vector_db"))
    client = chromadb.PersistentClient(path=persist_dir)

    try:
        return client.get_collection(name=collection_name)
    except Exception:
        return client.create_collection(name=collection_name)


def retrieve_documents(query: str, collection_name: str = "support_knowledge", k: int = 5) -> list[dict[str, Any]]:
    """Retrieve relevant support documents for a query using the persisted ChromaDB collection.

    This is the existing Week 2 retrieval logic adapted into a reusable function for
    the LangGraph node wrapper. The caller does not need to know how Chroma is configured.
    """

    if not query or not str(query).strip():
        return []

    try:
        collection = _get_collection(collection_name)
    except Exception:
        return []

    try:
        results = collection.query(query_texts=[str(query)], n_results=k)
    except Exception:
        return []

    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[{}]])[0]
    distances = results.get("distances", [[0.0]])[0]

    retrieved: list[dict[str, Any]] = []
    for index, document in enumerate(documents):
        metadata = metadatas[index] if index < len(metadatas) else {}
        distance = distances[index] if index < len(distances) else 0.0
        retrieved.append(
            {
                "id": ids[index] if index < len(ids) else f"doc_{index}",
                "title": str(metadata.get("title", f"Result {index + 1}")),
                "source": str(metadata.get("source", collection_name)),
                "content": str(document),
                "score": float(max(0.0, 1.0 - float(distance))),
                "metadata": dict(metadata),
            }
        )

    return retrieved


__all__ = ["retrieve_documents"]
