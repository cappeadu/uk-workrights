"""Retrieve the five closest chunks and evaluate them against pilot source labels."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from qdrant_client import QdrantClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STORAGE = PROJECT_ROOT / "qdrant_storage"
DEFAULT_QUESTIONS = PROJECT_ROOT / "tests" / "fixtures" / "pilot_questions.json"
DEFAULT_COLLECTION = "employment_guidance_chunks"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5


def load_embedding_model():
    """Load the fixed embedding model on CPU for question vectors."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME, device="cpu")


def load_questions(path: Path) -> list[dict[str, object]]:
    """Read and validate the labeled pilot questions."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read questions file {path}: {error}") from error

    questions = payload.get("questions") if isinstance(payload, dict) else None
    if not isinstance(questions, list) or not questions:
        raise ValueError("Questions file must contain a non-empty questions list")

    seen_ids: set[str] = set()
    for question in questions:
        if not isinstance(question, dict):
            raise TypeError("Every question must be an object")
        question_id = question.get("id")
        text = question.get("question")
        expected_source_ids = question.get("expected_source_ids")
        if not isinstance(question_id, str) or not question_id:
            raise ValueError("Every question must have a non-empty id")
        if question_id in seen_ids:
            raise ValueError(f"Duplicate question ID: {question_id}")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Question {question_id} has no text")
        if not isinstance(expected_source_ids, list) or not all(
            isinstance(source_id, str) for source_id in expected_source_ids
        ):
            raise ValueError(f"Question {question_id} has invalid expected_source_ids")
        seen_ids.add(question_id)
    return questions


def embed_question(question: str, model) -> list[float]:
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


def evaluate_questions(
    questions: list[dict[str, object]],
    client: QdrantClient,
    model,
    collection_name: str = DEFAULT_COLLECTION,
) -> dict[str, object]:
    """Evaluate top-five source coverage for questions with source labels."""
    results: list[dict[str, object]] = []
    scored_recalls: list[float] = []

    for question in questions:
        question_id = str(question["id"])
        text = str(question["question"])
        expected_sources = question["expected_source_ids"]
        retrieved = retrieve_top_five(text, client, model, collection_name)

        recall: float | None = None
        if expected_sources:
            found_sources = {str(item["source_id"]) for item in retrieved}
            recall = len(set(expected_sources) & found_sources) / len(set(expected_sources))
            scored_recalls.append(recall)

        results.append(
            {
                "id": question_id,
                "question": text,
                "expected_source_ids": expected_sources,
                "source_recall_at_5": recall,
                "retrieved": retrieved,
            }
        )

    return {
        "top_k": TOP_K,
        "source_recall_at_5": (
            sum(scored_recalls) / len(scored_recalls) if scored_recalls else None
        ),
        "scored_questions": len(scored_recalls),
        "unscored_questions": len(questions) - len(scored_recalls),
        "results": results,
    }


def print_report(report: dict[str, object]) -> None:
    for result in report["results"]:
        print(f"\n{result['id']}: {result['question']}")
        recall = result["source_recall_at_5"]
        if recall is None:
            print("  Source Recall@5: not scored (no expected source IDs)")
        else:
            expected = ", ".join(result["expected_source_ids"])
            print(f"  Source Recall@5: {recall:.2f}; expected: {expected}")

        for rank, item in enumerate(result["retrieved"], start=1):
            excerpt = " ".join(item["text"].split())[:180]
            if len(" ".join(item["text"].split())) > 180:
                excerpt += "..."
            print(
                f"  {rank}. {item['chunk_id']} — {item['title']} "
                f"(score {item['score']:.4f})"
            )
            print(f"     {item['canonical_url']}\n     {excerpt}")

    score = report["source_recall_at_5"]
    if score is None:
        print("\nSource Recall@5: not available; no questions have source labels.")
    else:
        print(
            f"\nMacro Source Recall@5: {score:.3f} "
            f"({report['scored_questions']} scored, "
            f"{report['unscored_questions']} unscored)"
        )
        print(
            "This is a source-level proxy: the current pilot labels source IDs. "
            "Chunk-level Recall@5 needs expected chunk IDs."
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run top-five retrieval against the labeled pilot questions."
    )
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--storage", type=Path, default=DEFAULT_STORAGE)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    args = parser.parse_args()

    client: QdrantClient | None = None
    try:
        questions = load_questions(args.questions)
        client = QdrantClient(path=str(args.storage))
        if not client.collection_exists(args.collection):
            raise ValueError(
                f"Qdrant collection {args.collection!r} was not found in {args.storage}; "
                "run scripts/load_qdrant.py first"
            )
        model = load_embedding_model()
        report = evaluate_questions(questions, client, model, args.collection)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    finally:
        if client is not None:
            client.close()

    print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
