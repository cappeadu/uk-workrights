"""Shared local embedding and Qdrant retrieval helpers."""

from __future__ import annotations

import math
from pathlib import Path

from qdrant_client import QdrantClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STORAGE = PROJECT_ROOT / "qdrant_storage"
DEFAULT_COLLECTION = "employment_guidance_chunks"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5


def load_embedding_model():
    """Load the fixed embedding model on CPU."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME, device="cpu")


def embed_question(question: str, model) -> list[float]:
    """Convert one question to a finite vector for Qdrant."""
    vectors = model.encode([question], show_progress_bar=False)
    if len(vectors) != 1:
        raise ValueError("The model did not return exactly one query vector")
    try:
        vector = [float(value) for value in vectors[0]]
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("The model returned an invalid query vector") from error
    if not vector or not all(math.isfinite(value) for value in vector):
        raise ValueError("The model returned an empty or non-finite query vector")
    return vector


def retrieve_top_five(
    question: str,
    client: QdrantClient,
    model,
    collection_name: str = DEFAULT_COLLECTION,
) -> list[dict[str, object]]:
    """Return the five highest-scoring source-linked chunks for a question."""
    response = client.query_points(
        collection_name=collection_name,
        query=embed_question(question, model),
        limit=TOP_K,
        with_payload=True,
    )

    results: list[dict[str, object]] = []
    for point in response.points:
        payload = point.payload
        if not isinstance(payload, dict):
            raise TypeError("A retrieved Qdrant point has no payload")
        source = payload.get("source")
        if not isinstance(source, dict):
            raise TypeError("A retrieved chunk has no source metadata")
        required_values = (
            payload.get("chunk_id"),
            payload.get("source_id"),
            payload.get("text"),
            source.get("title"),
            source.get("canonical_url"),
        )
        if not all(isinstance(value, str) and value for value in required_values):
            raise ValueError("A retrieved chunk is missing citation metadata")

        results.append(
            {
                "chunk_id": payload["chunk_id"],
                "source_id": payload["source_id"],
                "title": source["title"],
                "canonical_url": source["canonical_url"],
                "text": payload["text"],
                "score": float(point.score),
            }
        )
    return results
