"""Embed processed chunks with the project's fixed local model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "embeddings.jsonl"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_embedding_model():
    """Load the project's fixed embedding model on CPU."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME, device="cpu")


def embed_chunks(
    chunks: list[dict[str, object]], model=None
) -> list[dict[str, object]]:
    """Return each chunk ID paired with its embedding vector."""
    if not chunks:
        raise ValueError("No chunks were provided")

    chunk_ids: list[str] = []
    texts: list[str] = []
    for chunk in chunks:
        chunk_id = chunk.get("chunk_id")
        text = chunk.get("text")
        if not isinstance(chunk_id, str) or not chunk_id:
            raise ValueError("Every chunk must have a non-empty chunk_id")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Chunk {chunk_id} must have non-empty text")
        chunk_ids.append(chunk_id)
        texts.append(text)

    if model is None:
        model = load_embedding_model()

    vectors = model.encode(texts, show_progress_bar=False)
    if len(vectors) != len(chunks):
        raise ValueError("The model returned a different number of vectors than chunks")

    vector_size = len(vectors[0])
    if not vector_size or any(len(vector) != vector_size for vector in vectors):
        raise ValueError("The model returned empty or inconsistent vectors")

    return [
        {"chunk_id": chunk_id, "vector": [float(value) for value in vector]}
        for chunk_id, vector in zip(chunk_ids, vectors, strict=True)
    ]


def load_chunks(path: Path) -> list[dict[str, object]]:
    """Read and validate chunk records from JSON Lines."""
    chunks: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"Could not read chunks file {path}: {error}") from error

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            chunk = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid JSON in {path} on line {line_number}: {error}"
            ) from error
        if not isinstance(chunk, dict):
            raise TypeError(f"Chunk on line {line_number} must be a JSON object")

        chunk_id = chunk.get("chunk_id")
        text = chunk.get("text")
        if not isinstance(chunk_id, str) or not chunk_id:
            raise ValueError(f"Chunk on line {line_number} has no valid chunk_id")
        if chunk_id in seen_ids:
            raise ValueError(f"Duplicate chunk ID on line {line_number}: {chunk_id}")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Chunk {chunk_id} has no text")

        chunks.append(chunk)
        seen_ids.add(chunk_id)

    if not chunks:
        raise ValueError(f"No chunks found in {path}")
    return chunks


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create CPU embeddings for the processed source-linked chunks."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    try:
        chunks = load_chunks(args.input)
        embeddings = embed_chunks(chunks)
        output = "\n".join(
            json.dumps(embedding, ensure_ascii=False) for embedding in embeddings
        )
        args.output.write_text(output + "\n", encoding="utf-8")
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    dimensions = len(embeddings[0]["vector"])
    print(
        f"Embedded {len(embeddings)} chunks with {MODEL_NAME} "
        f"on CPU ({dimensions} dimensions). Wrote {args.output}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
