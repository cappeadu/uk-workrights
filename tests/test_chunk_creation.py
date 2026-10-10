import json

from scripts.create_chunks import (
    chunk_markdown,
    main,
    split_list_block,
    split_table_block,
    split_text_with_overlap,
)


def source_record(source_id="example-guide"):
    return {
        "source_id": source_id,
        "title": "Example guide",
        "canonical_path": f"/{source_id}",
        "canonical_url": f"https://www.gov.uk/{source_id}",
        "content_id": "content-123",
        "document_type": "guide",
        "schema_name": "guide",
        "phase": "live",
        "withdrawn": False,
        "first_published_at": "2020-01-01T00:00:00Z",
        "public_updated_at": "2024-01-02T00:00:00Z",
        "updated_at": "2024-01-02T01:00:00Z",
        "fetched_at": "2026-10-10T12:00:00+00:00",
        "content_sha256": "abc123",
        "attribution": "Contains public sector information.",
        "licence_url": "https://example.test/licence",
        "missing_fields": [],
    }


def processed_markdown(source):
    url = source["canonical_url"]
    source_id = source["source_id"]
    return (
        "# Example guide\n\n"
        f"Source: [{url}]({url})\n"
        f"Source ID: `{source_id}`\n"
        "Content ID: `content-123`\n"
        "Document type: `guide`\n"
        "Schema: `guide`\n\n"
        "## Main topic\n\n"
        "### Eligibility\n\n"
        "Workers can read the guidance.\n\n"
        "- First rule\n"
        "  - A nested detail\n\n"
        "| Requirement | Evidence |\n"
        "| --- | --- |\n"
        "| Example | Source text |\n"
    )


def test_chunks_keep_heading_context_lists_tables_and_source_metadata():
    source = source_record()

    chunks = chunk_markdown(processed_markdown(source), source)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk["chunk_id"] == "example-guide-0001"
    assert chunk["heading_path"] == ["Main topic", "Eligibility"]
    assert "### Eligibility" in chunk["text"]
    assert "- First rule\n  - A nested detail" in chunk["text"]
    assert "| Requirement | Evidence |" in chunk["text"]
    assert chunk["source"]["canonical_url"] == source["canonical_url"]
    assert chunk["source"]["attribution"] == source["attribution"]


def test_oversized_paragraph_uses_overlap_and_chunk_ids_are_repeatable():
    source = source_record()
    text = " ".join(f"word{index}" for index in range(450))
    chunks = split_text_with_overlap(text, target_words=200, overlap_words=30)
    assert len(chunks) == 3
    assert chunks[0].split()[-30:] == chunks[1].split()[:30]
    assert chunks[1].split()[-30:] == chunks[2].split()[:30]

    markdown = processed_markdown(source).replace(
        "Workers can read the guidance.", text
    )
    first_run = chunk_markdown(markdown, source)
    second_run = chunk_markdown(markdown, source)

    assert [chunk["chunk_id"] for chunk in first_run] == [
        chunk["chunk_id"] for chunk in second_run
    ]
    assert first_run == second_run


def test_large_lists_split_between_items_and_tables_repeat_their_headers():
    list_text = "\n".join(f"- Rule {index} has detail" for index in range(6))
    list_chunks = split_list_block(list_text, target_words=11)
    assert len(list_chunks) == 3
    assert all(chunk.startswith("- Rule") for chunk in list_chunks)

    table_text = (
        "| Day | Rule |\n"
        "| --- | --- |\n"
        "| Monday | First rule |\n"
        "| Tuesday | Second rule |\n"
        "| Wednesday | Third rule |"
    )
    table_chunks = split_table_block(table_text, target_words=18)
    assert len(table_chunks) == 3
    assert all("| Day | Rule |\n| --- | --- |" in chunk for chunk in table_chunks)
    assert "| Wednesday | Third rule |" in table_chunks[-1]


def test_cli_does_not_write_partial_chunks_when_a_source_file_is_missing(
    tmp_path, monkeypatch
):
    first_source = source_record()
    missing_source = source_record("missing-guide")
    metadata_path = tmp_path / "source_metadata.json"
    metadata_path.write_text(
        json.dumps({"sources": [first_source, missing_source]}), encoding="utf-8"
    )
    (tmp_path / f"{first_source['source_id']}.md").write_text(
        processed_markdown(first_source), encoding="utf-8"
    )
    output_path = tmp_path / "chunks.jsonl"
    review_path = tmp_path / "chunks_review.md"
    monkeypatch.setattr(
        "sys.argv",
        [
            "create_chunks.py",
            "--input-dir",
            str(tmp_path),
            "--metadata",
            str(metadata_path),
            "--output",
            str(output_path),
            "--review-output",
            str(review_path),
        ],
    )

    assert main() == 1
    assert not output_path.exists()
    assert not review_path.exists()


def test_cli_writes_source_linked_jsonl_for_valid_inputs(tmp_path, monkeypatch):
    source = source_record()
    metadata_path = tmp_path / "source_metadata.json"
    metadata_path.write_text(json.dumps({"sources": [source]}), encoding="utf-8")
    (tmp_path / f"{source['source_id']}.md").write_text(
        processed_markdown(source), encoding="utf-8"
    )
    output_path = tmp_path / "chunks.jsonl"
    review_path = tmp_path / "chunks_review.md"
    monkeypatch.setattr(
        "sys.argv",
        [
            "create_chunks.py",
            "--input-dir",
            str(tmp_path),
            "--metadata",
            str(metadata_path),
            "--output",
            str(output_path),
            "--review-output",
            str(review_path),
        ],
    )

    assert main() == 0
    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    chunk = json.loads(lines[0])
    assert chunk["source_id"] == source["source_id"]
    assert chunk["source"]["content_id"] == source["content_id"]

    review = review_path.read_text(encoding="utf-8")
    assert "## Example guide" in review
    assert f"Source: [{source['title']}]({source['canonical_url']})" in review
    assert "### example-guide-0001" in review
    assert "**Section:** Main topic / Eligibility" in review
    assert "> - First rule" in review
    assert "> | Requirement | Evidence |" in review
