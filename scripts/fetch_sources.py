"""Fetch the approved GOV.UK Content API pages in config/sources.toml."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_LIST = PROJECT_ROOT / "config" / "sources.toml"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


class DoNotFollowRedirects(HTTPRedirectHandler):
    """Return redirects to the caller so they can be reviewed and reported."""

    def redirect_request(self, request, response, code, message, headers, new_url):
        return None


@dataclass
class FetchResult:
    source_id: str
    status: str
    detail: str
    http_status: int | None = None
    saved_path: Path | None = None
    fetched_at: str | None = None
    content_sha256: str | None = None


def load_sources(path: Path) -> tuple[str, list[dict[str, object]]]:
    """Read and validate the approved, page-specific source allowlist."""
    try:
        config = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"Could not read source list {path}: {error}") from error

    api_base_url = config.get("api_base_url")
    sources = config.get("sources")
    if not isinstance(api_base_url, str) or not api_base_url.startswith(
        "https://www.gov.uk/api/content"
    ):
        raise ValueError("api_base_url must use https://www.gov.uk/api/content")
    if not isinstance(sources, list) or not sources:
        raise ValueError("The source list must contain at least one [[sources]] entry")

    seen_ids: set[str] = set()
    for source in sources:
        if not isinstance(source, dict):
            raise TypeError("Every source entry must be a TOML table")

        source_id = source.get("id")
        api_path = source.get("api_path")
        if not isinstance(source_id, str) or not re.fullmatch(r"[a-z0-9-]+", source_id):
            raise ValueError(
                "Each source id must contain only lowercase letters, digits, and hyphens"
            )
        if source_id in seen_ids:
            raise ValueError(f"Duplicate source id: {source_id}")
        seen_ids.add(source_id)

        if not isinstance(api_path, str) or not api_path.startswith("/"):
            raise ValueError(f"Source {source_id} must have an absolute api_path")
        if (
            ".." in api_path.split("/")
            or "\\" in api_path
            or "?" in api_path
            or "#" in api_path
        ):
            raise ValueError(f"Source {source_id} has an unsafe api_path")
        if source.get("review_status") != "approved":
            raise ValueError(f"Source {source_id} is not marked approved")

    return api_base_url.rstrip("/"), sources


def contains_page_text(details: object) -> bool:
    """Return whether API details contain a body or a guide part with a body."""
    if not isinstance(details, dict):
        return False

    body = details.get("body")
    if isinstance(body, str) and body.strip():
        return True

    parts = details.get("parts")
    return isinstance(parts, list) and any(contains_page_text(part) for part in parts)


def save_response(output_dir: Path, source_id: str, suffix: str, body: bytes) -> Path:
    path = output_dir / f"{source_id}{suffix}"
    path.write_bytes(body)
    return path


def write_fetch_manifest(
    output_dir: Path,
    api_base_url: str,
    sources: list[dict[str, object]],
    results: list[FetchResult],
) -> Path:
    """Save the local fetch time and hash for each response in this run."""
    result_by_id = {result.source_id: result for result in results}
    manifest = {
        "sources": [
            {
                "source_id": str(source["id"]),
                "requested_url": f"{api_base_url}{source['api_path']}",
                "status": result_by_id[str(source["id"])].status,
                "fetched_at": result_by_id[str(source["id"])].fetched_at,
                "content_sha256": result_by_id[str(source["id"])].content_sha256,
            }
            for source in sources
        ]
    }
    path = output_dir / "fetch_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def fetch_source(
    api_base_url: str,
    source: dict[str, object],
    output_dir: Path,
    opener,
    timeout: float,
) -> FetchResult:
    source_id = str(source["id"])
    api_path = str(source["api_path"])
    url = f"{api_base_url}{api_path}"
    request = Request(url, headers={"Accept": "application/json"})

    try:
        with opener.open(request, timeout=timeout) as response:
            http_status = response.status
            body = response.read()
    except HTTPError as error:
        body = error.read()
        suffix = f".http-{error.code}.body"
        saved_path = (
            save_response(output_dir, source_id, suffix, body) if body else None
        )
        if 300 <= error.code < 400:
            location = error.headers.get("Location", "not provided")
            return FetchResult(
                source_id,
                "redirected",
                f"HTTP {error.code}; Location: {location}",
                error.code,
                saved_path,
            )
        if error.code == 404:
            return FetchResult(source_id, "missing", "HTTP 404", error.code, saved_path)
        return FetchResult(
            source_id, "failed", f"HTTP {error.code}", error.code, saved_path
        )
    except (URLError, TimeoutError, OSError) as error:
        return FetchResult(source_id, "failed", str(error))

    if not 200 <= http_status < 300:
        saved_path = (
            save_response(output_dir, source_id, ".response.body", body)
            if body
            else None
        )
        return FetchResult(
            source_id,
            "failed",
            f"Unexpected HTTP {http_status}",
            http_status,
            saved_path,
        )

    if not body.strip():
        return FetchResult(
            source_id, "empty", "The API returned an empty response", http_status
        )

    saved_path = save_response(output_dir, source_id, ".json", body)
    try:
        item = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        return FetchResult(
            source_id,
            "failed",
            f"The response was not valid JSON: {error}",
            http_status,
            saved_path,
        )

    if not isinstance(item, dict):
        return FetchResult(
            source_id,
            "failed",
            "The JSON response was not an object",
            http_status,
            saved_path,
        )

    if item.get("withdrawn") or item.get("phase") == "withdrawn":
        return FetchResult(
            source_id,
            "withdrawn",
            "The API marks this page as withdrawn",
            http_status,
            saved_path,
        )

    if item.get("schema_name") == "redirect" or item.get("document_type") == "redirect":
        return FetchResult(
            source_id,
            "redirected",
            "The API returned a redirect document",
            http_status,
            saved_path,
        )

    if (
        item.get("schema_name") == "placeholder"
        or item.get("document_type") == "placeholder"
    ):
        return FetchResult(
            source_id,
            "placeholder",
            "The API returned a placeholder document",
            http_status,
            saved_path,
        )

    if not contains_page_text(item.get("details")):
        return FetchResult(
            source_id,
            "empty",
            "No readable body or guide parts were found",
            http_status,
            saved_path,
        )

    expected_fields = {
        "base_path": source.get("canonical_path"),
        "content_id": source.get("content_id"),
        "document_type": source.get("document_type"),
        "schema_name": source.get("schema_name"),
    }
    changed = [
        f"{field}: expected {expected!r}, got {item.get(field)!r}"
        for field, expected in expected_fields.items()
        if expected is not None and item.get(field) != expected
    ]
    if changed:
        return FetchResult(
            source_id, "changed", "; ".join(changed), http_status, saved_path
        )

    if item.get("phase") != "live":
        return FetchResult(
            source_id,
            "changed",
            f"Expected a live page, got phase {item.get('phase')!r}",
            http_status,
            saved_path,
        )

    return FetchResult(
        source_id,
        "ok",
        "Fetched and matched the approved source record",
        http_status,
        saved_path,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch the approved GOV.UK Content API source list."
    )
    parser.add_argument(
        "--sources", type=Path, default=SOURCE_LIST, help="TOML source allowlist"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RAW_DATA_DIR,
        help="Directory for raw API responses",
    )
    parser.add_argument(
        "--timeout", type=float, default=20, help="Request timeout in seconds"
    )
    args = parser.parse_args()

    try:
        api_base_url, sources = load_sources(args.sources)
    except ValueError as error:
        print(f"Source list error: {error}", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    opener = build_opener(DoNotFollowRedirects)
    results = [
        fetch_source(api_base_url, source, args.output_dir, opener, args.timeout)
        for source in sources
    ]

    for result in results:
        if result.saved_path and result.saved_path.exists():
            response_bytes = result.saved_path.read_bytes()
            result.fetched_at = datetime.fromtimestamp(
                result.saved_path.stat().st_mtime, tz=UTC
            ).isoformat()
            result.content_sha256 = hashlib.sha256(response_bytes).hexdigest()

    manifest_path = write_fetch_manifest(args.output_dir, api_base_url, sources, results)

    for result in results:
        status_text = (
            f"HTTP {result.http_status}"
            if result.http_status is not None
            else "no HTTP response"
        )
        saved_text = f"; saved {result.saved_path}" if result.saved_path else ""
        print(
            f"{result.status.upper():<11} {result.source_id}: {status_text}; {result.detail}{saved_text}"
        )

    successful = sum(result.status == "ok" for result in results)
    print(
        f"\nFetched {successful} of {len(results)} approved sources. Responses are in {args.output_dir}."
    )
    print(f"Fetch metadata saved to {manifest_path}.")
    return 0 if successful == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
