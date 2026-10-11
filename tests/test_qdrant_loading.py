import json
import sys
from uuid import NAMESPACE_URL, uuid5

import pytest
from qdrant_client import QdrantClient, models

from scripts import load_qdrant


def source_record(source_id="example-guide"):
    return {
        "source_id": source_id,
        "title": "Example guide",
        "canonical_path": f"/{source_id}",
        "canonical_url": f"https://www.gov.uk/{source_id}",
        "content_id": "content-123",
        "document_type": "guide",
        "schema_name": "guide",
        "attribution": "Contains public sector information.",
        "licence_url": "https://example.test/licence",
    }


def chunk_record(chunk_id="example-guide-0001"):
    source = source_record()
    return {
        "chunk_id": chunk_id,
        "source_id": source["source_id"],
        "heading_path": ["Main topic"],
        "text": "Workers can read the guidance.",
        "source": source,
    }


def test_build_points_matches_vectors_and_preserves_source_payload():
    chunk = chunk_record()
    embeddings = [{"chunk_id": chunk["chunk_id"], "vector": [0.1, 0.2, 0.3]}]

    points, vector_size = load_qdrant.build_points([chunk], embeddings)

    assert vector_size == 3
    assert len(points) == 1
    assert points[0].id == uuid5(
        NAMESPACE_URL, f"govuk-employment-guidance:{chunk['chunk_id']}"
    )
    assert points[0].vector == [0.1, 0.2, 0.3]
    assert points[0].payload == chunk


def test_build_points_rejects_mismatched_ids_and_vector_dimensions():
    first_chunk = chunk_record()
    second_chunk = chunk_record("example-guide-0002")
    first_embedding = {"chunk_id": first_chunk["chunk_id"], "vector": [0.1, 0.2]}

    with pytest.raises(ValueError, match="IDs do not match"):
        load_qdrant.build_points([first_chunk, second_chunk], [first_embedding])

    second_embedding = {"chunk_id": second_chunk["chunk_id"], "vector": [0.3]}
    with pytest.raises(ValueError, match="inconsistent dimensions"):
        load_qdrant.build_points(
            [first_chunk, second_chunk], [first_embedding, second_embedding]
        )


def test_main_persists_chunks_in_local_qdrant(tmp_path, monkeypatch, capsys):
    chunk = chunk_record()
    chunks_path = tmp_path / "chunks.jsonl"
    embeddings_path = tmp_path / "embeddings.jsonl"
    storage_path = tmp_path / "qdrant_storage"
    collection_name = "test_chunks"
    chunks_path.write_text(json.dumps(chunk) + "\n", encoding="utf-8")
    embeddings_path.write_text(
        json.dumps({"chunk_id": chunk["chunk_id"], "vector": [0.1, 0.2, 0.3]})
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "load_qdrant.py",
            "--chunks",
            str(chunks_path),
            "--embeddings",
            str(embeddings_path),
            "--storage",
            str(storage_path),
            "--collection",
            collection_name,
        ],
    )

    assert load_qdrant.main() == 0
    assert "Loaded 1 chunks" in capsys.readouterr().out

    client = QdrantClient(path=str(storage_path))
    try:
        info = client.get_collection(collection_name)
        assert info.config.params.vectors.size == 3
        assert info.config.params.vectors.distance == models.Distance.COSINE
        assert client.count(collection_name).count == 1
        records, _ = client.scroll(collection_name, limit=1, with_payload=True)
        assert records[0].payload == chunk
    finally:
        client.close()


def test_main_rejects_missing_embedding_without_creating_storage(tmp_path, monkeypatch):
    chunk = chunk_record()
    chunks_path = tmp_path / "chunks.jsonl"
    embeddings_path = tmp_path / "embeddings.jsonl"
    storage_path = tmp_path / "qdrant_storage"
    chunks_path.write_text(json.dumps(chunk) + "\n", encoding="utf-8")
    embeddings_path.write_text(
        json.dumps({"chunk_id": "another-guide-0001", "vector": [0.1]}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "load_qdrant.py",
            "--chunks",
            str(chunks_path),
            "--embeddings",
            str(embeddings_path),
            "--storage",
            str(storage_path),
        ],
    )

    assert load_qdrant.main() == 1
    assert not storage_path.exists()
