import json
import sys
from types import ModuleType

import pytest

from scripts import embed_chunks as embed_chunks_script

embed_chunks = embed_chunks_script.embed_chunks
load_embedding_model = embed_chunks_script.load_embedding_model

class FakeModel:
    def encode(self, texts, show_progress_bar=False):
        assert show_progress_bar is False
        return [[float(index), float(len(text))] for index, text in enumerate(texts)]


def test_embed_chunks_preserves_chunk_ids_and_returns_vectors():
    chunks = [
        {"chunk_id": "guide-0001", "text": "First chunk text"},
        {"chunk_id": "guide-0002", "text": "Second chunk text"},
    ]

    embeddings = embed_chunks(chunks, model=FakeModel())

    assert embeddings == [
        {"chunk_id": "guide-0001", "vector": [0.0, 16.0]},
        {"chunk_id": "guide-0002", "vector": [1.0, 17.0]},
    ]


def test_model_is_loaded_with_cpu_device(monkeypatch):
    calls = []
    fake_module = ModuleType("sentence_transformers")

    def fake_sentence_transformer(model_name, device):
        calls.append((model_name, device))
        return FakeModel()

    fake_module.SentenceTransformer = fake_sentence_transformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    model = load_embedding_model()

    assert isinstance(model, FakeModel)
    assert calls == [("sentence-transformers/all-MiniLM-L6-v2", "cpu")]


def test_embedding_command_writes_one_record_per_chunk(tmp_path, monkeypatch):
    input_path = tmp_path / "chunks.jsonl"
    output_path = tmp_path / "embeddings.jsonl"
    input_path.write_text(
        '\n'.join(
            [
                json.dumps({"chunk_id": "guide-0001", "text": "First chunk"}),
                json.dumps({"chunk_id": "guide-0002", "text": "Second chunk"}),
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "embed_chunks.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
    )
    monkeypatch.setattr(
        embed_chunks_script,
        "embed_chunks",
        lambda chunks: [
            {"chunk_id": chunk["chunk_id"], "vector": [0.1, 0.2]}
            for chunk in chunks
        ],
    )

    assert embed_chunks_script.main() == 0
    records = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert [record["chunk_id"] for record in records] == [
        "guide-0001",
        "guide-0002",
    ]
    assert all(len(record["vector"]) == 2 for record in records)


def test_embedding_command_reports_invalid_chunks_without_output(tmp_path, monkeypatch):
    input_path = tmp_path / "chunks.jsonl"
    output_path = tmp_path / "embeddings.jsonl"
    input_path.write_text(
        json.dumps({"chunk_id": "guide-0001", "text": "First chunk"})
        + "\n"
        + json.dumps({"chunk_id": "guide-0001", "text": "Duplicate"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "embed_chunks.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
    )

    assert embed_chunks_script.main() == 1
    assert not output_path.exists()


def test_embed_chunks_rejects_empty_input():
    with pytest.raises(ValueError, match="No chunks were provided"):
        embed_chunks([], model=FakeModel())
