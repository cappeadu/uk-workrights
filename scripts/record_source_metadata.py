"""Record source and fetch metadata for the approved GOV.UK responses."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_LIST = PROJECT_ROOT / "config" / "sources.toml"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
FETCH_MANIFEST = RAW_DATA_DIR / "fetch_manifest.json"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "source_metadata.json"


def load_source_config(path: Path) -> dict[str, object]:
    """Load the approved source list and its attribution details."""
    try:
        config = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"Could not read source list {path}: {error}") from error

    sources = config.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("The source list must contain at least one [[sources]] entry")
    if not isinstance(config.get("attribution"), str) or not isinstance(
        config.get("licence_url"), str
    ):
        raise TypeError("The source list must include attribution and licence_url")
    for source in sources:
        if not isinstance(source, dict) or source.get("review_status") != "approved":
            raise ValueError("Every source must be an approved TOML source entry")
    return config


def load_fetch_manifest(path: Path) -> dict[str, dict[str, object]]:
    """Load fetch times and hashes written by fetch_sources.py."""
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read fetch manifest {path}: {error}") from error

    entries = manifest.get("sources") if isinstance(manifest, dict) else None
    if not isinstance(entries, list):
        raise TypeError("Fetch manifest must contain a sources list")

    by_id: dict[str, dict[str, object]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("source_id"), str):
            raise TypeError("Every fetch manifest entry must have a source_id")
        source_id = entry["source_id"]
        if source_id in by_id:
            raise ValueError(f"Duplicate source in fetch manifest: {source_id}")
        by_id[source_id] = entry
    return by_id


def build_source_metadata(
    item: dict[str, object],
    source: dict[str, object],
    attribution: str,
    licence_url: str,
    fetched_at: str,
    fetched_at_source: str,
    content_sha256: str,
) -> dict[str, object]:
    """Combine the API record, approved allowlist, and local fetch details."""
    expected_fields = {
        "base_path": source.get("canonical_path"),
        "content_id": source.get("content_id"),
        "document_type": source.get("document_type"),
        "schema_name": source.get("schema_name"),
    }
    mismatches = [
        f"{field} expected {expected!r}, got {item.get(field)!r}"
        for field, expected in expected_fields.items()
        if expected is not None and item.get(field) != expected
    ]
    if mismatches:
        raise ValueError("API response does not match approved source: " + "; ".join(mismatches))

    base_path = item.get("base_path")
    if not isinstance(base_path, str) or not base_path.startswith("/"):
        raise ValueError("API response is missing a valid canonical base_path")

    timestamps = {
        key: item.get(key)
        for key in ("first_published_at", "public_updated_at", "updated_at")
    }
    missing_fields = [key for key, value in timestamps.items() if not value]
    title = item.get("title")
    content_id = item.get("content_id")
    document_type = item.get("document_type")
    schema_name = item.get("schema_name")
    for key, value in (
        ("title", title),
        ("content_id", content_id),
        ("document_type", document_type),
        ("schema_name", schema_name),
    ):
        if not value:
            missing_fields.append(key)

    return {
        "source_id": source["id"],
        "title": title,
        "canonical_path": base_path,
        "canonical_url": f"https://www.gov.uk{base_path}",
        "content_id": content_id,
        "document_type": document_type,
        "schema_name": schema_name,
        "phase": item.get("phase"),
        "withdrawn": bool(item.get("withdrawn") or item.get("phase") == "withdrawn"),
        "withdrawn_notice": item.get("withdrawn_notice"),
        **timestamps,
        "fetched_at": fetched_at,
        "fetched_at_source": fetched_at_source,
        "content_sha256": content_sha256,
        "attribution": attribution,
        "licence_url": licence_url,
        "missing_fields": missing_fields,
    }


def metadata_for_source(
    source: dict[str, object],
    raw_dir: Path,
    fetch_manifest: dict[str, dict[str, object]],
    manifest_exists: bool,
    attribution: str,
    licence_url: str,
) -> dict[str, object]:
    source_id = str(source["id"])
    raw_path = raw_dir / f"{source_id}.json"
    raw_bytes = raw_path.read_bytes()
    item = json.loads(raw_bytes)
    if not isinstance(item, dict):
        raise TypeError("Raw API JSON must contain an object")

    digest = hashlib.sha256(raw_bytes).hexdigest()
    fetch_record = fetch_manifest.get(source_id)
    if fetch_record is not None:
        if fetch_record.get("status") != "ok":
            raise ValueError(
                f"Latest fetch status is {fetch_record.get('status')!r}; review before recording metadata"
            )
        if fetch_record.get("content_sha256") != digest:
            raise ValueError("Fetch manifest hash does not match the saved API response")
        fetched_at = fetch_record.get("fetched_at")
        if not isinstance(fetched_at, str) or not fetched_at:
            raise ValueError("Fetch manifest is missing the fetch timestamp")
        time_source = "fetch_manifest"
    elif manifest_exists:
        raise ValueError("Source is missing from the fetch manifest")
    else:
        fetched_at = datetime.fromtimestamp(
            raw_path.stat().st_mtime, tz=UTC
        ).isoformat()
        time_source = "raw_file_modified_time"

    return build_source_metadata(
        item=item,
        source=source,
        attribution=attribution,
        licence_url=licence_url,
        fetched_at=fetched_at,
        fetched_at_source=time_source,
        content_sha256=digest,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Record metadata for the approved saved GOV.UK source responses."
    )
    parser.add_argument("--sources", type=Path, default=SOURCE_LIST, help="TOML source allowlist")
    parser.add_argument("--input-dir", type=Path, default=RAW_DATA_DIR, help="Directory with raw API JSON")
    parser.add_argument("--fetch-manifest", type=Path, default=FETCH_MANIFEST, help="Local fetch manifest")
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE, help="Output metadata JSON file")
    args = parser.parse_args()

    try:
        config = load_source_config(args.sources)
        fetch_manifest = load_fetch_manifest(args.fetch_manifest)
    except (TypeError, ValueError) as error:
        print(f"Metadata input error: {error}", file=sys.stderr)
        return 2

    manifest_exists = args.fetch_manifest.exists()
    records: list[dict[str, object]] = []
    failures = 0
    incomplete = 0
    sources = config["sources"]
    for source in sources:
        source_id = str(source.get("id", "unknown"))
        try:
            record = metadata_for_source(
                source=source,
                raw_dir=args.input_dir,
                fetch_manifest=fetch_manifest,
                manifest_exists=manifest_exists,
                attribution=str(config["attribution"]),
                licence_url=str(config["licence_url"]),
            )
        except FileNotFoundError:
            failures += 1
            print(f"MISSING  {source_id}: no saved API response")
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
            failures += 1
            print(f"FAILED   {source_id}: {error}")
        else:
            records.append(record)
            if record["missing_fields"]:
                incomplete += 1
                fields = ", ".join(record["missing_fields"])
                print(f"REVIEW   {source_id}: missing metadata fields: {fields}")
            else:
                print(f"RECORDED {source_id}: metadata complete")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"sources": records}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nRecorded metadata for {len(records)} of {len(sources)} approved sources.")
    print(f"Metadata saved to {args.output}.")
    return 0 if failures == 0 and incomplete == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
