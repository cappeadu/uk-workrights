"""Create deterministic, source-linked chunks from processed GOV.UK pages."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
SOURCE_METADATA_FILE = PROCESSED_DATA_DIR / "source_metadata.json"
CHUNKS_FILE = PROCESSED_DATA_DIR / "chunks.jsonl"
CHUNK_REVIEW_FILE = PROCESSED_DATA_DIR / "chunks_review.md"
DEFAULT_TARGET_WORDS = 200
DEFAULT_OVERLAP_WORDS = 30

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
LIST_ITEM_RE = re.compile(r"^( *)(?:[-+*]|\d+[.)])\s+")
TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
SOURCE_ID_RE = re.compile(r"^[a-z0-9-]+$")


@dataclass(frozen=True)
class ContentBlock:
    """A paragraph, list, or table together with its GOV.UK heading context."""

    text: str
    kind: str
    heading_path: tuple[tuple[int, str], ...]


def word_count(text: str) -> int:
    return len(text.split())


def split_text_with_overlap(
    text: str, target_words: int, overlap_words: int = DEFAULT_OVERLAP_WORDS
) -> list[str]:
    """Split an oversized paragraph at sentence ends where practical."""
    words = text.split()
    if len(words) <= target_words:
        return [text]

    overlap = min(overlap_words, max(0, target_words // 4))
    pieces: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + target_words, len(words))
        if end < len(words):
            earliest_end = start + max(1, target_words // 2)
            sentence_ends = [
                index + 1
                for index in range(earliest_end, end)
                if re.search(r"[.!?][\])'\"]?$", words[index])
            ]
            if sentence_ends:
                end = sentence_ends[-1]

        pieces.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = max(start + 1, end - overlap)
    return pieces


def split_list_block(text: str, target_words: int) -> list[str]:
    """Split a long list between top-level items, keeping nested items with them."""
    lines = text.splitlines()
    first_item = next(
        (LIST_ITEM_RE.match(line) for line in lines if LIST_ITEM_RE.match(line)), None
    )
    if first_item is None:
        return [text]
    root_indent = len(first_item.group(1))

    items: list[list[str]] = []
    for line in lines:
        match = LIST_ITEM_RE.match(line)
        if match and len(match.group(1)) == root_indent:
            items.append([line])
        elif items:
            items[-1].append(line)
        else:
            items.append([line])

    pieces: list[str] = []
    current: list[str] = []
    current_words = 0
    for item in items:
        item_text = "\n".join(item)
        item_words = word_count(item_text)
        if current and current_words + item_words > target_words:
            pieces.append("\n".join(current))
            current = []
            current_words = 0
        current.extend(item)
        current_words += item_words
    if current:
        pieces.append("\n".join(current))
    return pieces


def split_table_block(text: str, target_words: int) -> list[str]:
    """Split a Markdown table between rows and repeat its header in each piece."""
    lines = text.splitlines()
    if len(lines) < 3:
        return [text]

    header = lines[:2]
    rows = lines[2:]
    pieces: list[str] = []
    current_rows: list[str] = []
    current_words = word_count("\n".join(header))
    for row in rows:
        row_words = word_count(row)
        if current_rows and current_words + row_words > target_words:
            pieces.append("\n".join(header + current_rows))
            current_rows = []
            current_words = word_count("\n".join(header))
        current_rows.append(row)
        current_words += row_words
    if current_rows:
        pieces.append("\n".join(header + current_rows))
    return pieces or [text]


def parse_markdown_blocks(markdown: str) -> list[ContentBlock]:
    """Read extracted Markdown into content blocks with a heading path."""
    blocks: list[ContentBlock] = []
    heading_by_level: dict[int, str] = {}
    current_kind: str | None = None
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_kind, current_lines
        text = "\n".join(current_lines).strip()
        if text and current_kind:
            path = tuple(sorted(heading_by_level.items()))
            blocks.append(ContentBlock(text, current_kind, path))
        current_kind = None
        current_lines = []

    for line in markdown.splitlines():
        heading = HEADING_RE.match(line)
        if heading:
            flush()
            level = len(heading.group(1))
            for old_level in list(heading_by_level):
                if old_level >= level:
                    del heading_by_level[old_level]
            heading_by_level[level] = heading.group(2).strip()
            continue

        if not line.strip():
            flush()
            continue

        if TABLE_ROW_RE.match(line):
            if current_kind != "table":
                flush()
                current_kind = "table"
            current_lines.append(line.rstrip())
            continue

        list_item = LIST_ITEM_RE.match(line)
        if list_item or (current_kind == "list" and line.startswith((" ", "\t"))):
            if current_kind != "list":
                flush()
                current_kind = "list"
            current_lines.append(line.rstrip())
            continue

        if current_kind != "paragraph":
            flush()
            current_kind = "paragraph"
        current_lines.append(line.rstrip())

    flush()
    return blocks


def split_block(block: ContentBlock, target_words: int) -> list[ContentBlock]:
    if word_count(block.text) <= target_words:
        return [block]

    if block.kind == "table":
        pieces = split_table_block(block.text, target_words)
    elif block.kind == "list":
        pieces = split_list_block(block.text, target_words)
    else:
        pieces = split_text_with_overlap(block.text, target_words)
    return [ContentBlock(piece, block.kind, block.heading_path) for piece in pieces]


def source_body(markdown: str, source: dict[str, object]) -> str:
    """Validate the Markdown provenance header and return only page content."""
    lines = markdown.splitlines()
    if not lines or not lines[0].startswith("# "):
        raise ValueError("Extracted Markdown is missing its page title")

    source_id = str(source["source_id"])
    source_id_lines = [line for line in lines if line.startswith("Source ID:")]
    if source_id_lines != [f"Source ID: `{source_id}`"]:
        raise ValueError("Markdown Source ID does not match the metadata record")

    source_url = str(source["canonical_url"])
    expected_source_line = f"Source: [{source_url}]({source_url})"
    if expected_source_line not in lines:
        raise ValueError("Markdown source link does not match the metadata record")

    schema_indices = [
        index for index, line in enumerate(lines) if line.startswith("Schema:")
    ]
    if len(schema_indices) != 1:
        raise ValueError("Markdown provenance header is missing its schema line")

    content_lines = lines[schema_indices[0] + 1 :]
    while content_lines and not content_lines[0].strip():
        content_lines.pop(0)
    body = "\n".join(content_lines).strip()
    if not body:
        raise ValueError("Extracted Markdown has no page content")
    return body


def load_source_metadata(path: Path) -> list[dict[str, object]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read source metadata {path}: {error}") from error

    sources = payload.get("sources") if isinstance(payload, dict) else None
    if not isinstance(sources, list) or not sources:
        raise ValueError("Source metadata must contain a non-empty sources list")

    required_fields = (
        "title",
        "canonical_path",
        "canonical_url",
        "content_id",
        "document_type",
        "schema_name",
        "attribution",
        "licence_url",
    )
    seen_ids: set[str] = set()
    for source in sources:
        if not isinstance(source, dict):
            raise TypeError("Every source metadata entry must be an object")
        source_id = source.get("source_id")
        if not isinstance(source_id, str) or not SOURCE_ID_RE.fullmatch(source_id):
            raise ValueError("Every source metadata entry needs a safe source_id")
        if source_id in seen_ids:
            raise ValueError(f"Duplicate source metadata record: {source_id}")
        seen_ids.add(source_id)
        missing = [field for field in required_fields if not source.get(field)]
        recorded_missing = source.get("missing_fields")
        if not isinstance(recorded_missing, list):
            raise TypeError(f"Source {source_id} has no valid missing_fields list")
        missing.extend(recorded_missing)
        if missing:
            raise ValueError(
                f"Source {source_id} has incomplete metadata: {', '.join(map(str, missing))}"
            )
    return sources


def common_heading_path(blocks: list[ContentBlock]) -> list[str]:
    if not blocks:
        return []
    common = list(blocks[0].heading_path)
    for block in blocks[1:]:
        limit = min(len(common), len(block.heading_path))
        index = 0
        while index < limit and common[index] == block.heading_path[index]:
            index += 1
        common = common[:index]
    return [title for _, title in common]


def render_chunk_text(blocks: list[ContentBlock]) -> str:
    lines: list[str] = []
    previous_path: tuple[tuple[int, str], ...] = ()
    for block in blocks:
        common_length = 0
        while (
            common_length < min(len(previous_path), len(block.heading_path))
            and previous_path[common_length] == block.heading_path[common_length]
        ):
            common_length += 1
        for level, title in block.heading_path[common_length:]:
            lines.append(f"{'#' * level} {title}")
            lines.append("")
        lines.append(block.text)
        lines.append("")
        previous_path = block.heading_path
    return "\n".join(lines).strip()


def added_word_count(current: list[ContentBlock], unit: ContentBlock) -> int:
    previous_path = current[-1].heading_path if current else ()
    common_length = 0
    while (
        common_length < min(len(previous_path), len(unit.heading_path))
        and previous_path[common_length] == unit.heading_path[common_length]
    ):
        common_length += 1
    return word_count(unit.text) + sum(
        word_count(title) for _, title in unit.heading_path[common_length:]
    )


def chunk_markdown(
    markdown: str,
    source: dict[str, object],
    target_words: int = DEFAULT_TARGET_WORDS,
) -> list[dict[str, object]]:
    if target_words < 1:
        raise ValueError("target_words must be greater than zero")

    body = source_body(markdown, source)
    blocks = parse_markdown_blocks(body)
    if not blocks:
        raise ValueError("No readable content blocks were found")

    units = [unit for block in blocks for unit in split_block(block, target_words)]
    grouped: list[list[ContentBlock]] = []
    current: list[ContentBlock] = []
    current_words = 0
    current_scope: str | None = None

    def flush_group() -> None:
        nonlocal current, current_words, current_scope
        if current:
            grouped.append(current)
        current = []
        current_words = 0
        current_scope = None

    for unit in units:
        scope = next(
            (title for level, title in unit.heading_path if level == 2),
            "__document__",
        )
        rendered_words = added_word_count(current, unit)
        if current and (
            scope != current_scope or current_words + rendered_words > target_words
        ):
            flush_group()
            rendered_words = added_word_count(current, unit)
        if not current:
            current_scope = scope
        current.append(unit)
        current_words += rendered_words
        if rendered_words > target_words:
            flush_group()
    flush_group()

    records: list[dict[str, object]] = []
    source_id = str(source["source_id"])
    for index, group in enumerate(grouped, start=1):
        records.append(
            {
                "chunk_id": f"{source_id}-{index:04d}",
                "source_id": source_id,
                "heading_path": common_heading_path(group),
                "text": render_chunk_text(group),
                "source": source,
            }
        )
    return records


def build_all_chunks(
    sources: list[dict[str, object]],
    input_dir: Path,
    target_words: int = DEFAULT_TARGET_WORDS,
) -> tuple[list[dict[str, object]], list[str]]:
    records: list[dict[str, object]] = []
    failures: list[str] = []
    for source in sources:
        source_id = str(source["source_id"])
        markdown_path = input_dir / f"{source_id}.md"
        try:
            markdown = markdown_path.read_text(encoding="utf-8")
            chunks = chunk_markdown(markdown, source, target_words)
        except FileNotFoundError:
            failures.append(
                f"MISSING {source_id}: no extracted Markdown at {markdown_path}"
            )
        except (OSError, TypeError, ValueError) as error:
            failures.append(f"FAILED {source_id}: {error}")
        else:
            if not chunks:
                failures.append(f"FAILED {source_id}: no chunks were created")
            else:
                records.extend(chunks)
    return records, failures


def render_review_markdown(records: list[dict[str, object]]) -> str:
    """Create a readable Markdown view grouped by source and chunk."""
    lines = [
        "# GOV.UK chunk review",
        "",
        "Generated alongside `chunks.jsonl` for human review.",
        "",
    ]
    current_source_id: str | None = None
    for record in records:
        source = record["source"]
        source_id = str(record["source_id"])
        if source_id != current_source_id:
            if current_source_id is not None:
                lines.extend(["", "---", ""])
            title = str(source["title"])
            url = str(source["canonical_url"])
            lines.extend(
                [
                    f"## {title}",
                    "",
                    f"Source: [{title}]({url})",
                    f"Attribution: {source['attribution']}",
                    "",
                ]
            )
            current_source_id = source_id

        section = " / ".join(record["heading_path"]) or "Page content"
        text_lines = str(record["text"]).splitlines()
        quoted_text = [f"> {line}" if line else ">" for line in text_lines]
        lines.extend(
            [
                f"### {record['chunk_id']}",
                "",
                f"**Section:** {section}  ",
                f"**Words:** {word_count(str(record['text']))}",
                "",
                *quoted_text,
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create deterministic chunks from extracted GOV.UK source pages."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROCESSED_DATA_DIR,
        help="Directory with extracted Markdown",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=SOURCE_METADATA_FILE,
        help="Source metadata JSON",
    )
    parser.add_argument(
        "--output", type=Path, default=CHUNKS_FILE, help="Output JSONL file"
    )
    parser.add_argument(
        "--review-output",
        type=Path,
        default=CHUNK_REVIEW_FILE,
        help="Human-readable Markdown review file",
    )
    parser.add_argument(
        "--target-words",
        type=int,
        default=DEFAULT_TARGET_WORDS,
        help="Approximate maximum chunk size in words (default: 200)",
    )
    args = parser.parse_args()

    try:
        sources = load_source_metadata(args.metadata)
    except (OSError, TypeError, ValueError) as error:
        print(f"Metadata error: {error}", file=sys.stderr)
        return 2

    records, failures = build_all_chunks(sources, args.input_dir, args.target_words)
    for failure in failures:
        print(failure, file=sys.stderr)
    if failures:
        print(
            "No chunk file was written because one or more sources failed.",
            file=sys.stderr,
        )
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(json.dumps(record, ensure_ascii=False) for record in records)
    args.output.write_text(content + "\n", encoding="utf-8")
    review = render_review_markdown(records)
    args.review_output.parent.mkdir(parents=True, exist_ok=True)
    args.review_output.write_text(review, encoding="utf-8")
    print(
        f"Created {len(records)} chunks from {len(sources)} approved sources at {args.output}."
    )
    print(f"Human-readable review saved to {args.review_output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
