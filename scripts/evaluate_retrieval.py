"""Retrieve the five closest chunks and evaluate them against labeled evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from qdrant_client import QdrantClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from workrights_assistant.retrieval import (
    DEFAULT_COLLECTION,
    DEFAULT_STORAGE,
    TOP_K,
    load_embedding_model,
    retrieve_top_five,
)

DEFAULT_QUESTIONS = PROJECT_ROOT / "tests" / "fixtures" / "pilot_questions.json"
DEFAULT_REPORT = PROJECT_ROOT / "data" / "processed" / "retrieval_report.md"
RECALL_TARGET = 0.90


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
        expected_chunk_ids = question.get("expected_chunk_ids")
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
        if not isinstance(expected_chunk_ids, list) or not all(
            isinstance(chunk_id, str) for chunk_id in expected_chunk_ids
        ):
            raise ValueError(f"Question {question_id} has invalid expected_chunk_ids")
        seen_ids.add(question_id)
    return questions


def evaluate_questions(
    questions: list[dict[str, object]],
    client: QdrantClient,
    model,
    collection_name: str = DEFAULT_COLLECTION,
) -> dict[str, object]:
    """Evaluate top-five chunk and source coverage for labeled questions."""
    results: list[dict[str, object]] = []
    chunk_recalls: list[float] = []
    source_recalls: list[float] = []

    for question in questions:
        question_id = str(question["id"])
        text = str(question["question"])
        expected_sources = question["expected_source_ids"]
        expected_chunks = question["expected_chunk_ids"]
        retrieved = retrieve_top_five(text, client, model, collection_name)

        chunk_recall: float | None = None
        if expected_chunks:
            found_chunks = {str(item["chunk_id"]) for item in retrieved}
            expected_chunk_set = set(expected_chunks)
            chunk_recall = len(expected_chunk_set & found_chunks) / len(expected_chunk_set)
            chunk_recalls.append(chunk_recall)

        source_recall: float | None = None
        if expected_sources:
            found_sources = {str(item["source_id"]) for item in retrieved}
            expected_source_set = set(expected_sources)
            source_recall = len(expected_source_set & found_sources) / len(expected_source_set)
            source_recalls.append(source_recall)

        results.append(
            {
                "id": question_id,
                "question": text,
                "category": question.get("category", ""),
                "kind": question.get("kind", ""),
                "expected_source_ids": expected_sources,
                "expected_chunk_ids": expected_chunks,
                "expected_facts": question.get("expected_facts", []),
                "chunk_recall_at_5": chunk_recall,
                "source_recall_at_5": source_recall,
                "retrieved": retrieved,
            }
        )

    return {
        "top_k": TOP_K,
        "chunk_recall_at_5": (
            sum(chunk_recalls) / len(chunk_recalls) if chunk_recalls else None
        ),
        "source_recall_at_5": (
            sum(source_recalls) / len(source_recalls) if source_recalls else None
        ),
        "scored_questions": len(chunk_recalls),
        "unscored_questions": len(questions) - len(chunk_recalls),
        "results": results,
    }


def markdown_cell(value: object) -> str:
    """Make text safe to place in a Markdown table cell."""
    return " ".join(str(value).split()).replace("|", "\\|")


def write_markdown_report(report: dict[str, object], path: Path) -> None:
    """Write a readable retrieval summary and the ranked evidence for review."""
    chunk_score = report["chunk_recall_at_5"]
    source_score = report["source_recall_at_5"]
    target_met = chunk_score is not None and chunk_score >= RECALL_TARGET

    lines = [
        "# Retrieval evaluation report",
        "",
        f"- Top results checked: {report['top_k']}",
        f"- Questions scored: {report['scored_questions']}",
        f"- Questions not scored: {report['unscored_questions']}",
        "- Recall@5 target: 90%",
    ]
    if chunk_score is None:
        lines.append("- **Macro chunk Recall@5:** unavailable")
    else:
        result = "PASS" if target_met else "BELOW TARGET"
        lines.append(f"- **Macro chunk Recall@5:** {chunk_score:.1%} — {result}")
    if source_score is not None:
        lines.append(f"- Macro source Recall@5 (secondary): {source_score:.1%}")

    lines.extend(
        [
            "",
            "## Question summary",
            "",
            "| ID | Topic | Type | Recall@5 | Expected chunks | Found chunks | Missed chunks |",
            "| --- | --- | --- | ---: | --- | --- | --- |",
        ]
    )

    for result in report["results"]:
        retrieved_ids = {str(item["chunk_id"]) for item in result["retrieved"]}
        expected_ids = result["expected_chunk_ids"]
        found_ids = [chunk_id for chunk_id in expected_ids if chunk_id in retrieved_ids]
        missed_ids = [chunk_id for chunk_id in expected_ids if chunk_id not in retrieved_ids]
        recall = result["chunk_recall_at_5"]
        recall_text = "—" if recall is None else f"{recall:.0%}"

        lines.append(
            "| "
            + " | ".join(
                markdown_cell(value)
                for value in (
                    result["id"],
                    result["category"],
                    result["kind"],
                    recall_text,
                    ", ".join(expected_ids) or "—",
                    ", ".join(found_ids) or "—",
                    ", ".join(missed_ids) or "—",
                )
            )
            + " |"
        )

    lines.extend(["", "## Question details", ""])
    for result in report["results"]:
        question_id = markdown_cell(result["id"])
        recall = result["chunk_recall_at_5"]
        recall_text = "not scored" if recall is None else f"{recall:.0%} Recall@5"
        retrieved_ids = {str(item["chunk_id"]) for item in result["retrieved"]}
        missed_ids = [
            chunk_id
            for chunk_id in result["expected_chunk_ids"]
            if chunk_id not in retrieved_ids
        ]

        lines.extend(
            [
                f"### {question_id} — {markdown_cell(result['category'])} ({markdown_cell(result['kind'])})",
                "",
                f"**Question:** {markdown_cell(result['question'])}",
                f"**Result:** {recall_text}",
            ]
        )
        if result["expected_chunk_ids"]:
            lines.append(
                "**Missed expected chunks:** " + (", ".join(missed_ids) or "None")
            )
        if result["expected_facts"]:
            lines.extend(["", "Expected facts:"])
            lines.extend(f"- {markdown_cell(fact)}" for fact in result["expected_facts"])

        lines.extend(
            [
                "",
                "| Rank | Chunk | Source | Score | Excerpt |",
                "| ---: | --- | --- | ---: | --- |",
            ]
        )
        for rank, item in enumerate(result["retrieved"], start=1):
            excerpt = " ".join(item["text"].split())
            if len(excerpt) > 240:
                excerpt = excerpt[:240].rsplit(" ", 1)[0] + "…"
            source = (
                f"[{markdown_cell(item['title'])}]({item['canonical_url']})"
            )
            lines.append(
                f"| {rank} | {markdown_cell(item['chunk_id'])} | {source} "
                f"| {item['score']:.4f} | {markdown_cell(excerpt)} |"
            )
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def print_report(report: dict[str, object]) -> None:
    for result in report["results"]:
        print(f"\n{result['id']}: {result['question']}")
        chunk_recall = result["chunk_recall_at_5"]
        if chunk_recall is None:
            print("  Chunk Recall@5: not scored (no expected supporting chunks)")
        else:
            expected_chunks = ", ".join(result["expected_chunk_ids"])
            print(f"  Chunk Recall@5: {chunk_recall:.2f}; expected: {expected_chunks}")

        source_recall = result["source_recall_at_5"]
        if source_recall is not None:
            expected_sources = ", ".join(result["expected_source_ids"])
            print(f"  Source Recall@5: {source_recall:.2f}; expected: {expected_sources}")

        for rank, item in enumerate(result["retrieved"], start=1):
            excerpt = " ".join(item["text"].split())[:180]
            if len(" ".join(item["text"].split())) > 180:
                excerpt += "..."
            print(
                f"  {rank}. {item['chunk_id']} — {item['title']} "
                f"(score {item['score']:.4f})"
            )
            print(f"     {item['canonical_url']}\n     {excerpt}")

    score = report["chunk_recall_at_5"]
    if score is None:
        print("\nChunk Recall@5: not available; no questions have chunk labels.")
    else:
        print(
            f"\nMacro Chunk Recall@5: {score:.3f} "
            f"({report['scored_questions']} scored, "
            f"{report['unscored_questions']} unscored)"
        )

    source_score = report["source_recall_at_5"]
    if source_score is not None:
        print(f"Macro Source Recall@5: {source_score:.3f} (secondary diagnostic)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run top-five retrieval against the labeled pilot questions."
    )
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--storage", type=Path, default=DEFAULT_STORAGE)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help=f"Markdown report path (default: {DEFAULT_REPORT})",
    )
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
        write_markdown_report(report, args.report)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    finally:
        if client is not None:
            client.close()

    print_report(report)
    print(f"\nMarkdown report written to: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
