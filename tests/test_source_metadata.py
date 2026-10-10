import hashlib
import json

import pytest

from scripts.fetch_sources import FetchResult, write_fetch_manifest
from scripts.record_source_metadata import (
    build_source_metadata,
    metadata_for_source,
)

SOURCE = {
    "id": "example-guide",
    "title": "Example guide",
    "canonical_path": "/example-guide",
    "content_id": "content-123",
    "document_type": "guide",
    "schema_name": "guide",
}

API_ITEM = {
    "title": "Example guide",
    "base_path": "/example-guide",
    "content_id": "content-123",
    "document_type": "guide",
    "schema_name": "guide",
    "phase": "live",
    "withdrawn": False,
    "first_published_at": "2020-01-01T00:00:00Z",
    "public_updated_at": "2024-01-02T00:00:00Z",
    "updated_at": "2024-01-02T01:00:00Z",
}


def test_build_source_metadata_records_api_and_attribution_fields():
    metadata = build_source_metadata(
        item=API_ITEM,
        source=SOURCE,
        attribution="Contains public sector information.",
        licence_url="https://example.test/licence",
        fetched_at="2026-10-10T12:00:00+00:00",
        fetched_at_source="fetch_manifest",
        content_sha256="abc123",
    )

    assert metadata["canonical_url"] == "https://www.gov.uk/example-guide"
    assert metadata["content_id"] == "content-123"
    assert metadata["first_published_at"] == "2020-01-01T00:00:00Z"
    assert metadata["withdrawn"] is False
    assert metadata["content_sha256"] == "abc123"
    assert metadata["attribution"] == "Contains public sector information."
    assert metadata["missing_fields"] == []


def test_metadata_for_source_checks_manifest_hash(tmp_path):
    raw_bytes = json.dumps(API_ITEM).encode("utf-8")
    (tmp_path / "example-guide.json").write_bytes(raw_bytes)
    manifest = {
        "example-guide": {
            "status": "ok",
            "fetched_at": "2026-10-10T12:00:00+00:00",
            "content_sha256": "wrong-hash",
        }
    }

    with pytest.raises(ValueError, match="hash does not match"):
        metadata_for_source(
            source=SOURCE,
            raw_dir=tmp_path,
            fetch_manifest=manifest,
            manifest_exists=True,
            attribution="Attribution",
            licence_url="https://example.test/licence",
        )


def test_metadata_uses_file_time_for_responses_without_fetch_manifest(tmp_path):
    raw_bytes = json.dumps(API_ITEM).encode("utf-8")
    (tmp_path / "example-guide.json").write_bytes(raw_bytes)

    metadata = metadata_for_source(
        source=SOURCE,
        raw_dir=tmp_path,
        fetch_manifest={},
        manifest_exists=False,
        attribution="Attribution",
        licence_url="https://example.test/licence",
    )

    assert metadata["fetched_at_source"] == "raw_file_modified_time"
    assert metadata["content_sha256"] == hashlib.sha256(raw_bytes).hexdigest()
    assert metadata["fetched_at"]


def test_write_fetch_manifest_records_fetch_time_and_hash(tmp_path):
    result = FetchResult(
        source_id="example-guide",
        status="ok",
        detail="Fetched",
        fetched_at="2026-10-10T12:00:00+00:00",
        content_sha256="abc123",
    )

    path = write_fetch_manifest(
        output_dir=tmp_path,
        api_base_url="https://www.gov.uk/api/content",
        sources=[{"id": "example-guide", "api_path": "/example-guide"}],
        results=[result],
    )

    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["sources"][0]["fetched_at"] == "2026-10-10T12:00:00+00:00"
    assert manifest["sources"][0]["content_sha256"] == "abc123"
