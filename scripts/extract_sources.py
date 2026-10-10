"""Extract readable Markdown from the approved raw GOV.UK API responses."""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_LIST = PROJECT_ROOT / "config" / "sources.toml"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


class GovUkHtmlParser(HTMLParser):
    """Convert the HTML blocks used in GOV.UK content into simple Markdown."""

    formatting = {
        "b": "**",
        "strong": "**",
        "i": "*",
        "em": "*",
        "code": "`",
        "del": "~~",
        "s": "~~",
    }

    def __init__(self, source_url: str, heading_offset: int = 0) -> None:
        super().__init__(convert_charrefs=True)
        self.source_url = source_url
        self.heading_offset = heading_offset
        self.lines: list[str] = []
        self.buffer: list[str] = []
        self.block_kind = "paragraph"
        self.heading_level = 0
        self.list_stack: list[dict[str, object]] = []
        self.list_items: list[dict[str, object]] = []
        self.links: list[str | None] = []
        self.quote_depth = 0
        self.skip_depth = 0
        self.in_table = False
        self.table_rows: list[tuple[list[str], bool]] = []
        self.table_row: list[str] | None = None
        self.table_cell: list[str] | None = None
        self.table_cell_is_header = False

    def _append(self, text: str) -> None:
        if self.in_table:
            if self.table_cell is not None:
                self.table_cell.append(text)
            return
        self.buffer.append(text)

    def _flush_block(self) -> None:
        text = re.sub(r"\s+", " ", "".join(self.buffer)).strip()
        self.buffer.clear()
        if not text:
            return

        if self.block_kind == "heading":
            level = min(6, self.heading_level + self.heading_offset)
            self.lines.append(f"{'#' * max(level, 1)} {text}")
            self.lines.append("")
        elif self.list_items:
            item = self.list_items[-1]
            prefix = item["prefix"] if not item["written"] else item["continuation"]
            self.lines.append(f"{prefix}{text}")
            item["written"] = True
        else:
            prefix = "> " * self.quote_depth
            self.lines.append(f"{prefix}{text}")
            self.lines.append("")

        self.block_kind = "paragraph"

    def _start_link(self, attrs: list[tuple[str, str | None]]) -> None:
        href = next((value for name, value in attrs if name == "href"), None)
        if href:
            target = urljoin(self.source_url, href)
            self.links.append(target)
            self._append("[")
        else:
            self.links.append(None)

    def _end_link(self) -> None:
        if not self.links:
            return
        target = self.links.pop()
        if target:
            self._append(f"]({target})")

    def _start_table(self) -> None:
        self._flush_block()
        self.in_table = True
        self.table_rows = []
        self.table_row = None
        self.table_cell = None

    def _finish_table_row(self) -> None:
        if self.table_row is not None and self.table_row:
            self.table_rows.append((self.table_row, self.table_row_is_header))
        self.table_row = None

    def _render_table(self) -> None:
        if not self.table_rows:
            return

        column_count = max(len(row) for row, _ in self.table_rows)
        first_row, first_is_header = self.table_rows[0]
        if first_is_header:
            header = first_row
            body_rows = [row for row, _ in self.table_rows[1:]]
        else:
            header = [f"Column {index + 1}" for index in range(column_count)]
            body_rows = [row for row, _ in self.table_rows]

        def format_row(row: list[str]) -> str:
            cells = [cell.replace("|", "\\|") for cell in row]
            cells.extend([""] * (column_count - len(cells)))
            return "| " + " | ".join(cells) + " |"

        self.lines.append(format_row(header))
        self.lines.append("| " + " | ".join(["---"] * column_count) + " |")
        self.lines.extend(format_row(row) for row in body_rows)
        self.lines.append("")

    @property
    def table_row_is_header(self) -> bool:
        return bool(getattr(self, "_table_row_is_header", False))

    @table_row_is_header.setter
    def table_row_is_header(self, value: bool) -> None:
        self._table_row_is_header = value

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style"}:
            self.skip_depth += 1
            return
        if self.skip_depth:
            return

        if self.in_table:
            if tag == "tr":
                self.table_row = []
                self.table_row_is_header = False
            elif tag in {"td", "th"}:
                self.table_cell = []
                self.table_cell_is_header = tag == "th"
                self.table_row_is_header |= self.table_cell_is_header
            elif tag == "a":
                self._start_link(attrs)
            elif tag in self.formatting:
                self._append(self.formatting[tag])
            elif tag == "br":
                self._append(" ")
            return

        if tag == "table":
            self._start_table()
        elif tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._flush_block()
            self.block_kind = "heading"
            self.heading_level = int(tag[1])
        elif tag in {"ul", "ol"}:
            self._flush_block()
            self.list_stack.append({"tag": tag, "next": 1})
        elif tag == "li":
            self._flush_block()
            list_info = (
                self.list_stack[-1] if self.list_stack else {"tag": "ul", "next": 1}
            )
            depth = len(self.list_items)
            if list_info["tag"] == "ol":
                marker = f"{list_info['next']}."
                list_info["next"] = int(list_info["next"]) + 1
            else:
                marker = "-"
            indent = "  " * depth
            self.list_items.append(
                {
                    "prefix": f"{indent}{marker} ",
                    "continuation": f"{indent}  ",
                    "written": False,
                }
            )
        elif tag == "p":
            self._flush_block()
        elif tag == "a":
            self._start_link(attrs)
        elif tag in self.formatting:
            self._append(self.formatting[tag])
        elif tag == "br":
            self._append(" ")
        elif tag == "blockquote":
            self._flush_block()
            self.quote_depth += 1
        elif tag == "hr":
            self._flush_block()
            self.lines.extend(["---", ""])
        elif tag == "img":
            alt = next((value for name, value in attrs if name == "alt"), None)
            if alt:
                self._append(alt)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self.skip_depth:
            self.skip_depth -= 1
            return
        if self.skip_depth:
            return

        if self.in_table:
            if tag in {"td", "th"} and self.table_cell is not None:
                text = re.sub(r"\s+", " ", "".join(self.table_cell)).strip()
                if self.table_row is not None:
                    self.table_row.append(text)
                self.table_cell = None
            elif tag == "tr":
                self._finish_table_row()
            elif tag == "table":
                self._finish_table_row()
                self.in_table = False
                self._render_table()
                self.table_rows = []
            elif tag == "a":
                self._end_link()
            elif tag in self.formatting:
                self._append(self.formatting[tag])
            return

        if tag in {"h1", "h2", "h3", "h4", "h5", "h6", "p"}:
            self._flush_block()
        elif tag == "li":
            self._flush_block()
            if self.list_items:
                self.list_items.pop()
        elif tag in {"ul", "ol"}:
            self._flush_block()
            if self.list_stack:
                self.list_stack.pop()
            if not self.list_stack and not self.list_items:
                self.lines.append("")
        elif tag == "a":
            self._end_link()
        elif tag in self.formatting:
            self._append(self.formatting[tag])
        elif tag == "blockquote":
            self._flush_block()
            self.quote_depth = max(0, self.quote_depth - 1)

    def handle_data(self, data: str) -> None:
        if not self.skip_depth:
            self._append(data)


def html_to_markdown(html: str, source_url: str, heading_offset: int = 0) -> str:
    parser = GovUkHtmlParser(source_url, heading_offset)
    parser.feed(html)
    parser.close()
    parser._flush_block()
    return "\n".join(parser.lines).strip()


def load_approved_sources(path: Path) -> list[dict[str, object]]:
    try:
        config = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"Could not read source list {path}: {error}") from error

    sources = config.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("The source list must contain at least one [[sources]] entry")
    for source in sources:
        if not isinstance(source, dict) or source.get("review_status") != "approved":
            raise ValueError("Every source must be an approved TOML source entry")
    return sources


def extract_source(item: dict[str, object], source: dict[str, object]) -> str:
    """Extract one API item and include its source identity in the Markdown."""
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
        raise ValueError(
            "API response does not match the approved source: " + "; ".join(mismatches)
        )

    details = item.get("details")
    if not isinstance(details, dict):
        raise ValueError("API response has no content details")

    title = str(item.get("title") or source.get("title") or source["id"])
    source_url = f"https://www.gov.uk{item['base_path']}"
    content_id = str(item.get("content_id", "unknown"))
    document_type = str(item.get("document_type", "unknown"))
    schema_name = str(item.get("schema_name", "unknown"))
    sections: list[str] = []

    body = details.get("body")
    if isinstance(body, str) and body.strip():
        markdown = html_to_markdown(body, source_url)
        if markdown:
            sections.append(markdown)

    parts = details.get("parts")
    if isinstance(parts, list):
        for part in parts:
            if not isinstance(part, dict):
                continue
            part_title = str(part.get("title") or "Section")
            part_body = part.get("body")
            if not isinstance(part_body, str) or not part_body.strip():
                continue
            markdown = html_to_markdown(part_body, source_url, heading_offset=1)
            if markdown:
                sections.append(f"## {part_title}\n\n{markdown}")

    if not sections:
        raise ValueError("API response contains no readable body or guide parts")

    header = [
        f"# {title}",
        "",
        f"Source: [{source_url}]({source_url})",
        f"Source ID: `{source['id']}`",
        f"Content ID: `{content_id}`",
        f"Document type: `{document_type}`",
        f"Schema: `{schema_name}`",
        "",
    ]
    return "\n".join(header + sections).strip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract readable Markdown from saved, approved GOV.UK API responses."
    )
    parser.add_argument(
        "--sources", type=Path, default=SOURCE_LIST, help="TOML source allowlist"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=RAW_DATA_DIR,
        help="Directory with raw API JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSED_DATA_DIR,
        help="Directory for extracted Markdown",
    )
    args = parser.parse_args()

    try:
        sources = load_approved_sources(args.sources)
    except ValueError as error:
        print(f"Source list error: {error}", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    succeeded = 0
    for source in sources:
        source_id = str(source.get("id", "unknown"))
        input_path = args.input_dir / f"{source_id}.json"
        output_path = args.output_dir / f"{source_id}.md"
        try:
            item = json.loads(input_path.read_text(encoding="utf-8"))
            if not isinstance(item, dict):
                raise ValueError("Raw API JSON must contain an object")
            markdown = extract_source(item, source)
            output_path.write_text(markdown, encoding="utf-8")
        except FileNotFoundError:
            print(f"MISSING  {source_id}: no saved response at {input_path}")
        except (OSError, json.JSONDecodeError, ValueError) as error:
            print(f"FAILED   {source_id}: {error}")
        else:
            succeeded += 1
            print(f"EXTRACTED {source_id}: saved {output_path}")

    print(f"\nExtracted {succeeded} of {len(sources)} approved sources.")
    return 0 if succeeded == len(sources) else 1


if __name__ == "__main__":
    raise SystemExit(main())
