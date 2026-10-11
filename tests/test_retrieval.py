import json
import sys

import pytest
from qdrant_client import QdrantClient, models

from scripts import evaluate_retrieval


class FakeModel:
    def encode(self, texts, show_progress_bar=False):
        assert show_progress_bar is False
        return [[1.0, 0.0, 0.0] for _ in texts]


def payload(chunk_id, source_id, title=None):
    return {
        "chunk_id": chunk_id,
        "source_id": source_id,
        "heading_path": ["Topic"],
        "text": f"Evidence for {chunk_id}.",
        "source": {
            "source_id": source_id,
            "title": title or f"{source_id} guide",
            "canonical_url": f"https://www.gov.uk/{source_id}",
        },
    }


def make_client(collection_name="employment_guidance_chunks", storage_path=None):
    if storage_path is None:
        client = QdrantClient(":memory:")
    else:
        client = QdrantClient(path=str(storage_path))
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=3, distance=models.Distance.COSINE),
    )
    client.upsert(
        collection_name=collection_name,
        points=[
            models.PointStruct(
                id=index,
                vector=vector,
                payload=payload(chunk_id, source_id),
            )
            for index, (chunk_id, source_id, vector) in enumerate(
                [
                    ("guide-a-0001", "guide-a", [1.0, 0.0, 0.0]),
                    ("guide-b-0001", "guide-b", [0.8, 0.6, 0.0]),
                    ("guide-c-0001", "guide-c", [0.0, 1.0, 0.0]),
                    ("guide-d-0001", "guide-d", [0.0, 0.0, 1.0]),
                    ("guide-e-0001", "guide-e", [-1.0, 0.0, 0.0]),
                    ("guide-f-0001", "guide-f", [0.0, -1.0, 0.0]),
                ],
                start=1,
            )
        ],
        wait=True,
    )
    return client


def test_retrieve_top_five_returns_ranked_source_linked_chunks():
    client = make_client()
    try:
        results = evaluate_retrieval.retrieve_top_five(
            "What is the guidance?", client, FakeModel()
        )
    finally:
        client.close()

    assert len(results) == 5
    assert results[0]["chunk_id"] == "guide-a-0001"
    assert results[0]["canonical_url"] == "https://www.gov.uk/guide-a"
    assert results[0]["text"] == "Evidence for guide-a-0001."
    assert all(
        results[index]["score"] >= results[index + 1]["score"]
        for index in range(len(results) - 1)
    )


def test_evaluation_scores_source_coverage_and_skips_unlabeled_questions():
    client = make_client()
    questions = [
        {
            "id": "Q1",
            "question": "Find both sources.",
            "expected_source_ids": ["guide-a", "missing-guide"],
        },
        {
            "id": "Q2",
            "question": "No source label.",
            "expected_source_ids": [],
        },
    ]
    try:
        report = evaluate_retrieval.evaluate_questions(questions, client, FakeModel())
    finally:
        client.close()

    assert report["top_k"] == 5
    assert report["source_recall_at_5"] == 0.5
    assert report["scored_questions"] == 1
    assert report["unscored_questions"] == 1
    assert report["results"][1]["source_recall_at_5"] is None


def test_load_questions_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "questions.json"
    question = {
        "id": "Q1",
        "question": "Question text",
        "expected_source_ids": [],
    }
    path.write_text(
        json.dumps({"questions": [question, question]}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="Duplicate question ID"):
        evaluate_retrieval.load_questions(path)


def test_cli_evaluates_using_persistent_qdrant_and_fake_model(
    tmp_path, monkeypatch, capsys
):
    storage_path = tmp_path / "qdrant_storage"
    collection_name = "test_chunks"
    make_client(collection_name, storage_path).close()

    questions_path = tmp_path / "questions.json"
    questions_path.write_text(
        json.dumps(
            {
                "questions": [
                    {
                        "id": "Q1",
                        "question": "Find the closest evidence.",
                        "expected_source_ids": ["guide-a"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(evaluate_retrieval, "load_embedding_model", FakeModel)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_retrieval.py",
            "--questions",
            str(questions_path),
            "--storage",
            str(storage_path),
            "--collection",
            collection_name,
        ],
    )

    assert evaluate_retrieval.main() == 0
    assert "Macro Source Recall@5: 1.000" in capsys.readouterr().out
