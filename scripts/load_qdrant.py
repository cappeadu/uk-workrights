"""Load chunk embeddings and their source metadata into local Qdrant storage."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_CHUNKS = PROCESSED_DIR / "chunks.jsonl"
DEFAULT_EMBEDDINGS = PROCESSED_DIR / "embeddings.jsonl"
DEFAULT_STORAGE = PROJECT_ROOT / "qdrant_storage"
DEFAULT_COLLECTION = "employment_guidance_chunks"
REQUIRED_SOURCE_FIELDS = (
    "title",
    "canonical_url",
    "content_id",
    "attribution",
    "licence_url",
)


def load_jsonl(path: Path, record_type: str) -> list[dict[str, object]]:
    """Read JSON object records from a JSON Lines file."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(
            f"Could not read {record_type} file {path}: {error}"
        ) from error

    records: list[dict[str, object]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid JSON in {path} on line {line_number}: {error}"
            ) from error
        if not isinstance(record, dict):
            raise TypeError(f"Record on line {line_number} in {path} must be an object")
        records.append(record)

    if not records:
        raise ValueError(f"No {record_type} records found in {path}")
    return records


def records_by_chunk_id(
    records: list[dict[str, object]], record_type: str
) -> dict[str, dict[str, object]]:
    by_id: dict[str, dict[str, object]] = {}
    for record in records:
        chunk_id = record.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id:
            raise ValueError(f"Every {record_type} record must have a chunk_id")
        if chunk_id in by_id:
            raise ValueError(f"Duplicate {record_type} record for chunk {chunk_id}")
        by_id[chunk_id] = record
    return by_id


def build_points(
    chunks: list[dict[str, object]], embeddings: list[dict[str, object]]
) -> tuple[list[models.PointStruct], int]:
    """Match vectors to chunks and keep their evidence in the Qdrant payload."""
    chunks_by_id = records_by_chunk_id(chunks, "chunk")
    embeddings_by_id = records_by_chunk_id(embeddings, "embedding")
    chunk_ids = set(chunks_by_id)
    embedding_ids = set(embeddings_by_id)
    if chunk_ids != embedding_ids:
        missing_embeddings = sorted(chunk_ids - embedding_ids)
        missing_chunks = sorted(embedding_ids - chunk_ids)
        raise ValueError(
            "Chunk and embedding IDs do not match; "
            f"missing embeddings: {missing_embeddings}; missing chunks: {missing_chunks}"
        )

    points: list[models.PointStruct] = []
    vector_size: int | None = None
    for chunk_id, chunk in chunks_by_id.items():
        text = chunk.get("text")
        source_id = chunk.get("source_id")
        heading_path = chunk.get("heading_path")
        source = chunk.get("source")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Chunk {chunk_id} has no text")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError(f"Chunk {chunk_id} has no source_id")
        if not isinstance(heading_path, list):
            raise TypeError(f"Chunk {chunk_id} has no valid heading_path")
        if not isinstance(source, dict) or source.get("source_id") != source_id:
            raise ValueError(f"Chunk {chunk_id} has invalid source metadata")
        missing_source_fields = [
            field for field in REQUIRED_SOURCE_FIELDS if not source.get(field)
        ]
        if missing_source_fields:
            raise ValueError(
                f"Chunk {chunk_id} has incomplete source metadata: "
                + ", ".join(missing_source_fields)
            )

        embedding = embeddings_by_id[chunk_id]
        raw_vector = embedding.get("vector")
        if not isinstance(raw_vector, list) or not raw_vector:
            raise ValueError(f"Embedding for chunk {chunk_id} has no vector")
        try:
            vector = [float(value) for value in raw_vector]
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                f"Embedding for chunk {chunk_id} has an invalid vector"
            ) from error
        if not all(math.isfinite(value) for value in vector):
            raise ValueError(f"Embedding for chunk {chunk_id} has a non-finite value")
        if vector_size is None:
            vector_size = len(vector)
        elif len(vector) != vector_size:
            raise ValueError("Embedding vectors have inconsistent dimensions")

        point_id = uuid5(NAMESPACE_URL, f"govuk-employment-guidance:{chunk_id}")
        points.append(models.PointStruct(id=point_id, vector=vector, payload=chunk))

    if vector_size is None:
        raise ValueError("No embedding vectors were provided")
    return points, vector_size


def ensure_collection(
    client: QdrantClient, collection_name: str, vector_size: int
) -> None:
    """Create the collection or confirm an existing one has the expected vector setup."""
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size, distance=models.Distance.COSINE
            ),
        )
        return

    collection = client.get_collection(collection_name)
    vector_config = collection.config.params.vectors
    if (
        getattr(vector_config, "size", None) != vector_size
        or getattr(vector_config, "distance", None) != models.Distance.COSINE
    ):
        raise ValueError(
            f"Collection {collection_name} must use {vector_size}-dimension cosine vectors"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Load source-linked chunk vectors into local Qdrant storage."
    )
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--storage", type=Path, default=DEFAULT_STORAGE)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    args = parser.parse_args()

    client: QdrantClient | None = None
    try:
        chunks = load_jsonl(args.chunks, "chunk")
        embeddings = load_jsonl(args.embeddings, "embedding")
        points, vector_size = build_points(chunks, embeddings)

        client = QdrantClient(path=str(args.storage))
        ensure_collection(client, args.collection, vector_size)
        client.upsert(collection_name=args.collection, points=points, wait=True)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    finally:
        if client is not None:
            client.close()

    print(
        f"Loaded {len(points)} chunks into collection {args.collection} "
        f"({vector_size} dimensions, cosine) at {args.storage}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
